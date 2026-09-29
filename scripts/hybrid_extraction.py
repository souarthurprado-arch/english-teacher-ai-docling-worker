"""Optional PyMuPDF4LLM evidence for the Docling import worker.

This module never decides pedagogical correctness. It only normalizes an
independent geometric extraction and records agreement with Docling.

PyMuPDF4LLM is dual licensed AGPL-3.0 / Artifex Commercial. Execution is
therefore explicitly gated by PYMUPDF4LLM_LICENSE_MODE.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any

ALLOWED_LICENSE_MODES = {"AGPL", "COMMERCIAL"}


def license_mode() -> str:
    return os.environ.get("PYMUPDF4LLM_LICENSE_MODE", "DISABLED").strip().upper()


def enabled() -> bool:
    return license_mode() in ALLOWED_LICENSE_MODES


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bbox(
    value: Any,
    *,
    page_width: float,
    page_height: float,
) -> dict[str, Any] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or page_width <= 0
        or page_height <= 0
    ):
        return None
    coords = [_finite(v) for v in value]
    if any(v is None for v in coords):
        return None
    x0, y0, x1, y1 = [float(v) for v in coords]
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)

    def clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    x = clamp(left / page_width)
    y = clamp(top / page_height)
    r = clamp(right / page_width)
    b = clamp(bottom / page_height)
    if r <= x or b <= y:
        return None
    return {
        "x": x,
        "y": y,
        "width": r - x,
        "height": b - y,
        "coordinate_space": "RELATIVE_TOP_LEFT",
    }


def _kind(raw: Any) -> str:
    value = str(raw or "TEXT").strip().upper().replace("-", "_")
    aliases = {
        "SECTION_HEADER": "SECTION_HEADER",
        "PAGE_HEADER": "PAGE_HEADER",
        "PAGE_FOOTER": "PAGE_FOOTER",
        "LIST_ITEM": "LIST_ITEM",
        "PICTURE": "PICTURE",
        "TABLE": "TABLE",
        "CAPTION": "CAPTION",
        "TITLE": "TITLE",
        "FOOTNOTE": "FOOTNOTE",
        "FORMULA": "FORMULA",
        "TEXT": "TEXT",
    }
    return aliases.get(value, "TEXT")


def _page_numbers(
    page_range: tuple[int, int] | None,
    total_pages: int | None,
) -> list[int]:
    if page_range:
        return list(range(page_range[0], page_range[1] + 1))
    if total_pages:
        return list(range(1, total_pages + 1))
    return []


def extract_geometry(
    source_path: str,
    *,
    page_range: tuple[int, int] | None,
    total_pages: int | None,
) -> dict[str, Any]:
    """Extract text/layout boxes with PyMuPDF4LLM.

    Failure is returned as evidence instead of aborting Docling, because this
    engine is corroborating evidence during the safe hybrid rollout.
    """
    mode = license_mode()
    if mode not in ALLOWED_LICENSE_MODES:
        return {
            "engine": "pymupdf4llm",
            "available": False,
            "status": "DISABLED",
            "license_mode": mode,
            "elements": [],
            "text_content": "",
        }

    try:
        import pymupdf
        import pymupdf4llm
    except Exception as exc:
        return {
            "engine": "pymupdf4llm",
            "available": False,
            "status": "IMPORT_ERROR",
            "license_mode": mode,
            "error": f"{type(exc).__name__}: {exc}"[:240],
            "elements": [],
            "text_content": "",
        }

    pages_one_based = _page_numbers(page_range, total_pages)
    pages_zero_based = [page - 1 for page in pages_one_based] or None

    try:
        chunks = pymupdf4llm.to_text(
            source_path,
            pages=pages_zero_based,
            page_chunks=True,
            use_ocr=True,
            force_text=True,
            show_progress=False,
        )
        if not isinstance(chunks, list):
            raise RuntimeError("PyMuPDF4LLM page_chunks output is not a list.")

        document = pymupdf.open(source_path)
        elements: list[dict[str, Any]] = []
        text_parts: list[str] = []
        try:
            for position, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    continue
                metadata = chunk.get("metadata")
                raw_page = (
                    metadata.get("page_number")
                    if isinstance(metadata, dict)
                    else None
                )
                try:
                    page_number = int(raw_page)
                except (TypeError, ValueError):
                    page_number = (
                        pages_one_based[position]
                        if position < len(pages_one_based)
                        else position + 1
                    )
                if not 1 <= page_number <= len(document):
                    continue

                page = document[page_number - 1]
                width, height = float(page.rect.width), float(page.rect.height)
                page_text = str(chunk.get("text") or "")
                text_parts.append(f"[[PAGE {page_number}]]\n{page_text}".strip())

                boxes = chunk.get("page_boxes")
                if not isinstance(boxes, list):
                    continue
                for box_index, box in enumerate(boxes):
                    if not isinstance(box, dict):
                        continue
                    normalized_bbox = _bbox(
                        box.get("bbox"),
                        page_width=width,
                        page_height=height,
                    )
                    if not normalized_bbox:
                        continue

                    fragment = ""
                    pos = box.get("pos")
                    if (
                        isinstance(pos, (list, tuple))
                        and len(pos) == 2
                    ):
                        try:
                            start, stop = int(pos[0]), int(pos[1])
                            fragment = page_text[max(0, start):max(start, stop)].strip()
                        except (TypeError, ValueError):
                            fragment = ""

                    item: dict[str, Any] = {
                        "engine": "pymupdf4llm",
                        "source_ref": (
                            f"p{page_number}:pymupdf4llm:#/page_boxes/{box_index}"
                        ),
                        "kind": _kind(box.get("class")),
                        "page_number": page_number,
                        "bbox": normalized_bbox,
                    }
                    if fragment:
                        item["text"] = re.sub(r"\s+", " ", fragment)[:1200]
                    elements.append(item)
        finally:
            document.close()

        version = getattr(pymupdf4llm, "__version__", None)
        if not isinstance(version, str):
            version = "unknown"
        return {
            "engine": "pymupdf4llm",
            "engine_version": version,
            "available": True,
            "status": "SUCCESS",
            "license_mode": mode,
            "page_range": list(page_range) if page_range else None,
            "text_content": "\n\n".join(text_parts),
            "elements": elements,
            "compaction": {
                "mode": "HYBRID_GEOMETRY_V1",
                "element_count": len(elements),
                "text_geometry_elements": sum(
                    1 for item in elements if item.get("kind") == "TEXT"
                ),
                "visual_elements": sum(
                    1
                    for item in elements
                    if item.get("kind") in {"PICTURE", "TABLE"}
                ),
            },
        }
    except Exception as exc:
        return {
            "engine": "pymupdf4llm",
            "available": False,
            "status": "EXTRACTION_ERROR",
            "license_mode": mode,
            "error": f"{type(exc).__name__}: {exc}"[:240],
            "elements": [],
            "text_content": "",
        }


def _tokens(text: str) -> Counter[str]:
    words = re.findall(r"[\w'-]+", text.lower(), flags=re.UNICODE)
    return Counter(word for word in words if len(word) > 1)


def _weighted_jaccard(left: str, right: str) -> float | None:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return None
    keys = set(a) | set(b)
    union = sum(max(a[key], b[key]) for key in keys)
    if not union:
        return None
    intersection = sum(min(a[key], b[key]) for key in keys)
    return round(intersection / union, 4)


def consensus(docling: dict[str, Any], pymupdf4llm: dict[str, Any]) -> dict[str, Any]:
    if pymupdf4llm.get("status") != "SUCCESS":
        return {
            "status": "SINGLE_ENGINE",
            "material_disagreement": False,
            "engines": ["docling"],
            "secondary_status": pymupdf4llm.get("status", "UNAVAILABLE"),
        }

    similarity = _weighted_jaccard(
        str(docling.get("text_content") or ""),
        str(pymupdf4llm.get("text_content") or ""),
    )
    doc_elements = [
        item for item in (docling.get("elements") or []) if isinstance(item, dict)
    ]
    pymu_elements = [
        item
        for item in (pymupdf4llm.get("elements") or [])
        if isinstance(item, dict)
    ]

    def counts(items: list[dict[str, Any]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in items:
            kind = str(item.get("kind") or "UNKNOWN").upper()
            result[kind] = result.get(kind, 0) + 1
        return result

    doc_counts, pymu_counts = counts(doc_elements), counts(pymu_elements)
    if similarity is None:
        status = "COMPLEMENTARY"
        material = False
    elif similarity >= 0.82:
        status = "AGREED"
        material = False
    elif similarity >= 0.48:
        status = "COMPLEMENTARY"
        material = False
    else:
        status = "DISAGREED"
        material = True

    return {
        "status": status,
        "material_disagreement": material,
        "engines": ["docling", "pymupdf4llm"],
        "text_similarity": similarity,
        "element_counts": {
            "docling": doc_counts,
            "pymupdf4llm": pymu_counts,
        },
    }


def attach(
    docling: dict[str, Any],
    *,
    source_path: str,
    page_range: tuple[int, int] | None,
    total_pages: int | None,
) -> dict[str, Any]:
    secondary = extract_geometry(
        source_path,
        page_range=page_range,
        total_pages=total_pages,
    )
    merged = dict(docling)
    merged["hybrid_evidence"] = {
        "contract": "EXTRACTION_CONTRACT_V1",
        "pymupdf4llm": secondary,
        "consensus": consensus(docling, secondary),
    }
    return merged
