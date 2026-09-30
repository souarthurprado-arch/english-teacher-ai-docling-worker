#!/usr/bin/env python3
"""GitHub Actions Docling worker.

The workflow claims a single queued Library file from the Supabase worker
gateway. This script sends the signed original URL to a localhost Docling Serve
container, compacts each page batch, then submits it back to Supabase. GitHub
OIDC is used for every callback; no Supabase key is stored in GitHub.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pymupdf
from PIL import Image, ImageOps
from pypdf import PdfReader

from hybrid_extraction import attach as attach_hybrid_evidence

WORKER_URL = os.environ.get(
    "DOCLING_WORKER_URL",
    "https://gxyufydpltpidzbeytxv.supabase.co/functions/v1/docling-worker",
)
DOCLING_URL = os.environ.get("DOCLING_LOCAL_URL", "http://127.0.0.1:5001")
OIDC_AUDIENCE = "english-teacher-ai-docling"
PREVIEW_MAX_EDGE = 1440
PREVIEW_WEBP_QUALITY = 76
PREVIEW_MAX_BYTES = 1_200_000


def http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout: int = 60,
) -> Any:
    payload = None
    merged = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        payload = json.dumps(body, separators=(",", ":")).encode()
        merged["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=merged, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw.decode()) if raw else None


def github_oidc_token() -> str:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
    if not request_url or not request_token:
        raise RuntimeError("GitHub OIDC environment is unavailable.")
    separator = "&" if "?" in request_url else "?"
    payload = http_json(
        f"{request_url}{separator}audience={urllib.parse.quote(OIDC_AUDIENCE)}",
        headers={"Authorization": f"bearer {request_token}"},
        timeout=20,
    )
    token = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("GitHub OIDC token was not returned.")
    return token


def worker_call(action: str, payload: dict[str, Any], timeout: int = 70) -> Any:
    token = github_oidc_token()
    return http_json(
        WORKER_URL,
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
        body={"action": action, **payload},
        timeout=timeout,
    )


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def page_size(document: dict[str, Any], page_no: int) -> tuple[float, float] | None:
    pages = document.get("pages")
    if not isinstance(pages, dict):
        return None
    raw = pages.get(str(page_no)) or pages.get(str(page_no - 1))
    if not isinstance(raw, dict):
        return None
    size = raw.get("size")
    if not isinstance(size, dict):
        return None
    width, height = finite(size.get("width")), finite(size.get("height"))
    if not width or not height or width <= 0 or height <= 0:
        return None
    return width, height


def normalize_bbox(value: Any, size: tuple[float, float]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    coords = [finite(value.get(key)) for key in ("l", "t", "r", "b")]
    if any(v is None for v in coords):
        return None
    l, t, r, b = [float(v) for v in coords]
    page_width, page_height = size
    left, right = min(l, r), max(l, r)
    origin = str(value.get("coord_origin", "TOPLEFT")).upper()
    if origin == "BOTTOMLEFT":
        top = page_height - max(t, b)
        bottom = page_height - min(t, b)
    else:
        top, bottom = min(t, b), max(t, b)

    def clamp(n: float) -> float:
        return max(0.0, min(1.0, n))

    x, y = clamp(left / page_width), clamp(top / page_height)
    x2, y2 = clamp(right / page_width), clamp(bottom / page_height)
    width, height = clamp(x2 - x), clamp(y2 - y)
    if width <= 0 or height <= 0:
        return None
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "coordinate_space": "RELATIVE_TOP_LEFT",
    }


def absolute_page(raw_page: int, page_range: tuple[int, int] | None) -> int:
    if not page_range or page_range[0] <= 1:
        return raw_page
    start, end = page_range
    batch_len = end - start + 1
    if 1 <= raw_page <= batch_len:
        return start + raw_page - 1
    return raw_page


def compact_table_data(raw: dict[str, Any]) -> dict[str, Any] | None:
    data = raw.get("data")
    if not isinstance(data, dict):
        return None

    try:
        num_rows = max(0, int(data.get("num_rows") or 0))
        num_cols = max(0, int(data.get("num_cols") or 0))
    except (TypeError, ValueError):
        num_rows, num_cols = 0, 0

    compact_cells: list[dict[str, Any]] = []
    cells = data.get("table_cells")
    if isinstance(cells, list):
        for cell in cells[:240]:
            if not isinstance(cell, dict):
                continue
            text = cell.get("text")
            compact: dict[str, Any] = {
                "text": " ".join(text.split()).strip()[:500]
                if isinstance(text, str)
                else "",
            }
            for key in (
                "start_row_offset_idx",
                "end_row_offset_idx",
                "start_col_offset_idx",
                "end_col_offset_idx",
                "row_span",
                "col_span",
            ):
                value = cell.get(key)
                try:
                    compact[key] = int(value)
                except (TypeError, ValueError):
                    pass
            for key in ("column_header", "row_header", "row_section", "fillable"):
                if isinstance(cell.get(key), bool):
                    compact[key] = cell[key]
            compact_cells.append(compact)

    rows: list[list[str]] = []
    if num_rows > 0 and num_cols > 0 and num_rows * num_cols <= 400:
        rows = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in compact_cells:
            row = cell.get("start_row_offset_idx")
            col = cell.get("start_col_offset_idx")
            if (
                isinstance(row, int)
                and isinstance(col, int)
                and 0 <= row < num_rows
                and 0 <= col < num_cols
                and cell.get("text")
            ):
                rows[row][col] = str(cell["text"])

    if not compact_cells and not rows:
        return None
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "cells": compact_cells,
        "rows": rows,
    }


def compact_document(
    document: dict[str, Any],
    text_content: str,
    *,
    page_range: tuple[int, int] | None,
    total_pages: int | None,
    preserve_text_geometry: bool = False,
) -> dict[str, Any]:
    page_fragments: dict[int, list[str]] = {}
    seen_fragments: dict[int, set[str]] = {}

    texts = document.get("texts")
    if isinstance(texts, list):
        for raw in texts[:900]:
            if not isinstance(raw, dict):
                continue
            provs = raw.get("prov")
            prov = next((item for item in provs or [] if isinstance(item, dict)), None)
            if not prov:
                continue
            raw_page_float = finite(prov.get("page_no"))
            if raw_page_float is None or raw_page_float < 1:
                continue
            raw_page = int(raw_page_float)
            page_number = absolute_page(raw_page, page_range)
            candidate = raw.get("text") if isinstance(raw.get("text"), str) else raw.get("orig")
            if not isinstance(candidate, str):
                continue
            fragment = " ".join(candidate.split()).strip()
            if not fragment:
                continue
            fragment = fragment[:500]
            seen = seen_fragments.setdefault(page_number, set())
            key = fragment.lower()
            if key in seen or len(page_fragments.setdefault(page_number, [])) >= 32:
                continue
            seen.add(key)
            page_fragments[page_number].append(fragment)

    sections: list[str] = []
    for page_number in sorted(page_fragments):
        sections.append(f"[[PAGE {page_number}]]\n" + "\n".join(page_fragments[page_number]))
    compact_text = "\n\n".join(sections)
    if not compact_text and isinstance(text_content, str):
        compact_text = text_content
    compact_text = compact_text[:20000]

    elements: list[dict[str, Any]] = []
    for key, kind, limit in (("pictures", "PICTURE", 120), ("tables", "TABLE", 80)):
        items = document.get(key)
        if not isinstance(items, list):
            continue
        for index, raw in enumerate(items[:limit]):
            if not isinstance(raw, dict):
                continue
            provs = raw.get("prov")
            prov = next((item for item in provs or [] if isinstance(item, dict)), None)
            if not prov:
                continue
            raw_page_float = finite(prov.get("page_no"))
            if raw_page_float is None or raw_page_float < 1:
                continue
            raw_page = int(raw_page_float)
            size = page_size(document, raw_page)
            if not size:
                continue
            bbox = normalize_bbox(prov.get("bbox"), size)
            if not bbox:
                continue
            page_number = absolute_page(raw_page, page_range)
            local_ref = raw.get("self_ref")
            if not isinstance(local_ref, str) or not local_ref.strip():
                local_ref = f"#/{key}/{index}"
            element: dict[str, Any] = {
                "source_ref": f"p{page_number}:{local_ref.strip()}",
                "kind": kind,
                "page_number": page_number,
                "bbox": bbox,
            }
            candidate_text = raw.get("text") if isinstance(raw.get("text"), str) else raw.get("orig")
            if isinstance(candidate_text, str) and candidate_text.strip():
                element["text"] = " ".join(candidate_text.split()).strip()[:700]
            if isinstance(raw.get("label"), str) and raw["label"].strip():
                element["label"] = raw["label"].strip()[:80]
            if kind == "TABLE":
                table_data = compact_table_data(raw)
                if table_data:
                    element["table_data"] = table_data
            elements.append(element)

    if preserve_text_geometry and isinstance(texts, list):
        for index, raw in enumerate(texts[:500]):
            if not isinstance(raw, dict):
                continue
            provs = raw.get("prov")
            prov = next((item for item in provs or [] if isinstance(item, dict)), None)
            if not prov:
                continue
            raw_page_float = finite(prov.get("page_no"))
            if raw_page_float is None or raw_page_float < 1:
                continue
            raw_page = int(raw_page_float)
            size = page_size(document, raw_page)
            if not size:
                continue
            bbox = normalize_bbox(prov.get("bbox"), size)
            if not bbox:
                continue
            candidate = raw.get("text") if isinstance(raw.get("text"), str) else raw.get("orig")
            if not isinstance(candidate, str):
                continue
            fragment = " ".join(candidate.split()).strip()
            if not fragment:
                continue
            page_number = absolute_page(raw_page, page_range)
            local_ref = raw.get("self_ref")
            if not isinstance(local_ref, str) or not local_ref.strip():
                local_ref = f"#/texts/{index}"
            element: dict[str, Any] = {
                "source_ref": f"p{page_number}:{local_ref.strip()}",
                "kind": "TEXT",
                "page_number": page_number,
                "bbox": bbox,
                "text": fragment[:1000],
            }
            if isinstance(raw.get("label"), str) and raw["label"].strip():
                element["label"] = raw["label"].strip()[:80]
            elements.append(element)

    return {
        "engine": "docling",
        "available": True,
        "status": "SUCCESS",
        "page_count": total_pages or len(document.get("pages") or {}),
        "text_content": compact_text,
        "elements": elements,
        "visual_assets": [],
        "page_range": list(page_range) if page_range else None,
        "compaction": {
            "mode": "MANUAL_CURATION_V2",
            "text_chars": len(compact_text),
            "visual_elements": len(
                [item for item in elements if item.get("kind") in {"PICTURE", "TABLE"}]
            ),
            "text_geometry_elements": len(
                [item for item in elements if item.get("kind") == "TEXT"]
            ),
        },
    }


def extract_batch(
    source_url: str,
    page_range: tuple[int, int] | None,
    total_pages: int | None,
    *,
    force_ocr: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "to_formats": ["json", "text"],
        "image_export_mode": "placeholder",
        "include_images": False,
        "include_page_images": False,
        "do_ocr": True,
        "force_ocr": force_ocr,
        "do_table_structure": True,
        "table_mode": "accurate",
        "abort_on_error": False,
    }
    if page_range:
        options["page_range"] = list(page_range)

    raw = http_json(
        f"{DOCLING_URL}/v1/convert/source",
        method="POST",
        body={"sources": [{"kind": "http", "url": source_url}], "options": options},
        timeout=420,
    )
    if not isinstance(raw, dict):
        raise RuntimeError("Docling returned an invalid response.")
    result = raw.get("document")
    if not isinstance(result, dict):
        raise RuntimeError(f"Docling did not return a document ({raw.get('status')}).")
    document_json = result.get("json_content")
    if isinstance(document_json, str):
        document = json.loads(document_json)
    elif isinstance(document_json, dict):
        document = document_json
    else:
        raise RuntimeError("Docling did not return structured JSON.")
    if not isinstance(document, dict):
        raise RuntimeError("Docling document JSON is invalid.")
    text_content = result.get("text_content")
    return compact_document(
        document,
        text_content if isinstance(text_content, str) else "",
        page_range=page_range,
        total_pages=total_pages,
        preserve_text_geometry=force_ocr,
    )


def download_source_file(source_url: str, suffix: str) -> str:
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "english-teacher-ai-docling-worker/1.0"},
    )
    temp_path = ""
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_path = temp_file.name
                shutil.copyfileobj(response, temp_file)
        return temp_path
    except Exception:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
        raise


def detect_pdf_page_count(temp_path: str) -> int:
    pages = len(PdfReader(temp_path).pages)
    if pages < 1:
        raise RuntimeError("PDF contains no readable pages.")
    return pages


def encode_webp(image: Image.Image) -> bytes:
    working = image.convert("RGB")
    attempts = [
        (PREVIEW_WEBP_QUALITY, PREVIEW_MAX_EDGE),
        (68, 1280),
        (60, 1120),
    ]
    for quality, edge in attempts:
        candidate = working.copy()
        candidate.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        candidate.save(buffer, format="WEBP", quality=quality, method=4)
        payload = buffer.getvalue()
        candidate.close()
        if len(payload) <= PREVIEW_MAX_BYTES:
            return payload
    return payload


def render_pdf_page_preview(document: pymupdf.Document, page_number: int) -> tuple[bytes, int, int]:
    page = document.load_page(page_number - 1)
    rect = page.rect
    longest = max(float(rect.width), float(rect.height), 1.0)
    scale = min(3.0, PREVIEW_MAX_EDGE / longest)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    mode = "RGB" if pix.n == 3 else "RGBA"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    try:
        payload = encode_webp(image)
        probe = Image.open(io.BytesIO(payload))
        try:
            return payload, int(probe.width), int(probe.height)
        finally:
            probe.close()
    finally:
        image.close()


def render_image_preview(source_path: str) -> tuple[bytes, int, int]:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            payload = encode_webp(image)
        finally:
            image.close()
    probe = Image.open(io.BytesIO(payload))
    try:
        return payload, int(probe.width), int(probe.height)
    finally:
        probe.close()


def submit_preview(
    *,
    file_id: str,
    page_number: int,
    payload: bytes,
    width: int,
    height: int,
) -> None:
    response = worker_call(
        "submit_preview",
        {
            "file_id": file_id,
            "page_number": page_number,
            "format": "webp",
            "width": width,
            "height": height,
            "byte_size": len(payload),
            "data_base64": base64.b64encode(payload).decode("ascii"),
        },
        timeout=70,
    )
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise RuntimeError(f"Supabase rejected page preview {page_number}: {response!r}")


def ranges_for(page_count: int | None, batch_pages: int) -> list[tuple[int, int] | None]:
    if not page_count:
        return [None]
    return [
        (start, min(page_count, start + batch_pages - 1))
        for start in range(1, page_count + 1, batch_pages)
    ]



def write_style_audit(document: pymupdf.Document, *, file_id: str) -> None:
    """Persist a compact source-style inventory for editorial fidelity auditing."""
    pages: dict[str, list[dict[str, Any]]] = {}
    totals = {"bold": 0, "italic": 0, "bold_italic": 0}
    for page_index in range(len(document)):
        page = document.load_page(page_index)
        payload = page.get_text("dict")
        styled: list[dict[str, Any]] = []
        for block in payload.get("blocks", []):
            if not isinstance(block, dict) or block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                if not isinstance(line, dict):
                    continue
                for span in line.get("spans", []):
                    if not isinstance(span, dict):
                        continue
                    text_value = " ".join(str(span.get("text") or "").split()).strip()
                    if not text_value:
                        continue
                    flags = int(span.get("flags") or 0)
                    font = str(span.get("font") or "")
                    font_lc = font.lower()
                    bold = bool(flags & 16) or "bold" in font_lc or "black" in font_lc
                    italic = bool(flags & 2) or "italic" in font_lc or "oblique" in font_lc
                    if not (bold or italic):
                        continue
                    bbox = span.get("bbox")
                    item = {
                        "text": text_value[:500],
                        "font": font[:120],
                        "size": round(float(span.get("size") or 0.0), 2),
                        "flags": flags,
                        "bold": bold,
                        "italic": italic,
                    }
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        item["bbox"] = [round(float(value), 2) for value in bbox]
                    styled.append(item)
                    if bold and italic:
                        totals["bold_italic"] += 1
                    elif bold:
                        totals["bold"] += 1
                    elif italic:
                        totals["italic"] += 1
        if styled:
            pages[str(page_index + 1)] = styled
    result = {
        "file_id": file_id,
        "engine": "PyMuPDF",
        "method": "SOURCE_STYLE_SPANS_V1",
        "page_count": len(document),
        "limitations": [
            "Bold and italic are derived from PDF font flags/font names.",
            "Underline is not reliably encoded as a font span and is not certified by this audit."
        ],
        "totals": totals,
        "pages": pages,
    }
    with open("/tmp/source-style-audit.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
    print(
        "Stored source style audit: "
        f"{len(pages)} page(s), {totals['bold']} bold, "
        f"{totals['italic']} italic, {totals['bold_italic']} bold+italic span(s)."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-file", required=True)
    args = parser.parse_args()

    with open(args.claim_file, "r", encoding="utf-8") as handle:
        claim = json.load(handle)
    job = claim.get("job") if isinstance(claim, dict) else None
    if not isinstance(job, dict):
        print("No queued Docling job.")
        return 0

    file_id = str(job.get("file_id") or "")
    source_url = str(job.get("source_url") or "")
    page_count = int(job["page_count"]) if job.get("page_count") else None
    batch_pages = max(1, int(job.get("batch_pages") or 8))
    mime_type = str(job.get("mime_type") or "").lower()
    file_name = str(job.get("file_name") or "").lower()
    is_pdf = mime_type == "application/pdf" or file_name.endswith(".pdf")
    is_image = mime_type.startswith("image/")
    suffix = ".pdf" if is_pdf else os.path.splitext(file_name)[1] or ".img"
    source_path = download_source_file(source_url, suffix) if (is_pdf or is_image) else ""
    pdf_document: pymupdf.Document | None = None
    if is_pdf:
        if page_count is None:
            page_count = detect_pdf_page_count(source_path)
            print(f"Detected {page_count} PDF page(s) from the preserved original.")
        pdf_document = pymupdf.open(source_path)
    ranges = ranges_for(page_count, batch_pages)
    completed_batches = {
        int(value)
        for value in (job.get("completed_batches") or [])
        if str(value).isdigit() and int(value) > 0
    }
    existing_previews = {
        int(key)
        for key, value in (job.get("page_previews") or {}).items()
        if str(key).isdigit() and isinstance(value, dict) and value.get("path")
    }
    preview_backfill_only = job.get("preview_backfill_only") is True
    preview_pages = sorted(
        {
            int(value)
            for value in (job.get("preview_pages") or [])
            if str(value).isdigit() and int(value) > 0
        }
    )

    reconstruction_only = job.get("reconstruction_only") is True
    reconstruction_pages = sorted(
        {
            int(value)
            for value in (job.get("reconstruction_pages") or [])
            if str(value).isdigit() and int(value) > 0
        }
    )
    reconstruction_completed_pages = {
        int(value)
        for value in (job.get("reconstruction_completed_pages") or [])
        if str(value).isdigit() and int(value) > 0
    }

    if reconstruction_only and pdf_document is not None:
        write_style_audit(pdf_document, file_id=file_id)

    if not file_id or not source_url:
        raise RuntimeError("Claimed job is missing file_id/source_url.")

    resumed = bool(completed_batches)
    print(
        f"Processing {job.get('file_name') or file_id}: {len(ranges)} batch(es)"
        + (f", resuming after {len(completed_batches)} completed." if resumed else ".")
    )
    try:
        if reconstruction_only:
            print(
                f"Docling reconstruction pass: {len(reconstruction_pages)} page(s) "
                f"requested for {job.get('file_name') or file_id}."
            )
            for page_number in reconstruction_pages:
                if page_number in reconstruction_completed_pages:
                    print(f"Skipping reconstructed page {page_number}.")
                    continue
                if page_count is not None and not (1 <= page_number <= page_count):
                    raise RuntimeError(
                        f"Reconstruction page {page_number} is outside the document."
                    )
                print(
                    f"Reconstructing page {page_number} with Docling "
                    "(directed OCR enabled)."
                )
                extraction = extract_batch(
                    source_url,
                    (page_number, page_number),
                    page_count,
                    force_ocr=True,
                )
                if is_pdf and source_path:
                    extraction = attach_hybrid_evidence(
                        extraction,
                        source_path=source_path,
                        page_range=(page_number, page_number),
                        total_pages=page_count,
                    )
                    consensus_state = (
                        extraction.get("hybrid_evidence", {})
                        .get("consensus", {})
                        .get("status", "SINGLE_ENGINE")
                    )
                    print(
                        f"Hybrid reconstruction consensus page {page_number}: "
                        f"{consensus_state}."
                    )
                response = worker_call(
                    "submit_reconstruction",
                    {
                        "file_id": file_id,
                        "page_number": page_number,
                        "extraction": extraction,
                    },
                    timeout=70,
                )
                if not isinstance(response, dict) or response.get("ok") is not True:
                    raise RuntimeError(
                        f"Supabase rejected reconstruction page {page_number}: {response!r}"
                    )
                reconstruction_completed_pages.add(page_number)

            response = worker_call(
                "finish_reconstruction",
                {"file_id": file_id},
                timeout=30,
            )
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise RuntimeError(
                    f"Supabase could not finalize Docling reconstruction: {response!r}"
                )
            print("Docling reconstruction pass completed; awaiting ChatGPT curation.")
            return 0

        if preview_backfill_only:
            print(
                f"Preview-only backfill: {len(preview_pages)} page(s) requested "
                f"for {job.get('file_name') or file_id}."
            )
            for page_number in preview_pages:
                if page_number in existing_previews:
                    continue
                if pdf_document is not None and 1 <= page_number <= len(pdf_document):
                    preview, width, height = render_pdf_page_preview(
                        pdf_document,
                        page_number,
                    )
                elif is_image and page_number == 1 and source_path:
                    preview, width, height = render_image_preview(source_path)
                else:
                    continue
                submit_preview(
                    file_id=file_id,
                    page_number=page_number,
                    payload=preview,
                    width=width,
                    height=height,
                )
                existing_previews.add(page_number)
                print(
                    f"Stored compact page preview {page_number}: "
                    f"{width}x{height}, {len(preview)} bytes."
                )

            response = worker_call(
                "finish_preview_backfill",
                {"file_id": file_id},
                timeout=30,
            )
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise RuntimeError(
                    f"Supabase could not finalize preview backfill: {response!r}"
                )
            print("Preview backfill completed.")
            return 0

        for index, page_range in enumerate(ranges, start=1):
            if index in completed_batches:
                print(f"Skipping already completed batch {index}/{len(ranges)}.")
                continue
            label = f"{page_range[0]}-{page_range[1]}" if page_range else "full"
            print(f"Docling batch {index}/{len(ranges)} pages {label}.")
            extraction = extract_batch(source_url, page_range, page_count)
            if is_pdf and source_path:
                extraction = attach_hybrid_evidence(
                    extraction,
                    source_path=source_path,
                    page_range=page_range,
                    total_pages=page_count,
                )
                consensus_state = (
                    extraction.get("hybrid_evidence", {})
                    .get("consensus", {})
                    .get("status", "SINGLE_ENGINE")
                )
                print(
                    f"Hybrid extraction consensus pages {label}: "
                    f"{consensus_state}."
                )

            visual_pages = sorted(
                {
                    int(element.get("page_number"))
                    for element in (extraction.get("elements") or [])
                    if isinstance(element, dict)
                    and str(element.get("kind") or "").upper() in {"PICTURE", "TABLE"}
                    and str(element.get("page_number") or "").isdigit()
                }
            )
            for page_number in visual_pages:
                if page_number in existing_previews:
                    continue
                if pdf_document is not None and 1 <= page_number <= len(pdf_document):
                    preview, width, height = render_pdf_page_preview(
                        pdf_document,
                        page_number,
                    )
                elif is_image and page_number == 1 and source_path:
                    preview, width, height = render_image_preview(source_path)
                else:
                    continue
                submit_preview(
                    file_id=file_id,
                    page_number=page_number,
                    payload=preview,
                    width=width,
                    height=height,
                )
                existing_previews.add(page_number)
                print(
                    f"Stored compact page preview {page_number}: "
                    f"{width}x{height}, {len(preview)} bytes."
                )

            response = worker_call(
                "submit_batch",
                {
                    "file_id": file_id,
                    "batch_index": index,
                    "batch_total": len(ranges),
                    "extraction": extraction,
                },
                timeout=70,
            )
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise RuntimeError(f"Supabase rejected batch {index}: {response!r}")
        print("Docling extraction and curation completed.")
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:480]
        print(message, file=sys.stderr)
        try:
            worker_call("fail", {"file_id": file_id, "message": message}, timeout=30)
        except Exception as callback_exc:
            print(f"Failure callback also failed: {callback_exc}", file=sys.stderr)
        return 1
    finally:
        if pdf_document is not None:
            pdf_document.close()
        if source_path:
            try:
                os.unlink(source_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())