#!/usr/bin/env python3
"""Materialize curated Docling visual regions as persisted Supabase assets.

The source PDF is obtained only through a short-lived signed URL returned by the
OIDC-authenticated Edge Function. No Supabase secret is stored in GitHub.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import fitz

WORKER_URL = os.environ.get("DOCLING_WORKER_URL", "").strip()
OIDC_AUDIENCE = "english-teacher-ai-docling"
MAX_BYTES = 6_000_000
MAX_DIMENSION = 4_000


def request_json(url: str, *, method: str = "GET", payload=None, headers=None, timeout=120):
    data = None
    merged = {"Accept": "application/json"}
    if headers:
        merged.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        merged["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=merged, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def oidc_token() -> str:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
    if not request_url or not request_token:
        raise RuntimeError("GitHub OIDC environment is unavailable.")
    separator = "&" if "?" in request_url else "?"
    payload = request_json(
        f"{request_url}{separator}audience={urllib.parse.quote(OIDC_AUDIENCE)}",
        headers={"Authorization": f"bearer {request_token}"},
        timeout=30,
    )
    token = payload.get("value")
    if not isinstance(token, str) or not token:
        raise RuntimeError("GitHub OIDC token was not returned.")
    return token


def worker_post(payload: dict, timeout: int = 180):
    if not WORKER_URL:
        raise RuntimeError("DOCLING_WORKER_URL is required.")
    token = oidc_token()
    return request_json(
        WORKER_URL,
        method="POST",
        payload=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )


def download_source(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "eta-visual-worker/1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        with destination.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)


def normalized_clip(page: fitz.Page, bbox: dict) -> fitz.Rect:
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    width = float(bbox.get("width", 0))
    height = float(bbox.get("height", 0))
    if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0):
        raise RuntimeError("Invalid relative visual bbox.")
    page_rect = page.rect
    clip = fitz.Rect(
        page_rect.x0 + x * page_rect.width,
        page_rect.y0 + y * page_rect.height,
        page_rect.x0 + min(1.0, x + width) * page_rect.width,
        page_rect.y0 + min(1.0, y + height) * page_rect.height,
    )
    clip &= page_rect
    if clip.is_empty or clip.width < 2 or clip.height < 2:
        raise RuntimeError("Visual bbox produced an empty crop.")
    return clip


def render_png(page: fitz.Page, bbox: dict):
    clip = normalized_clip(page, bbox)
    last = None
    for scale in (3.0, 2.5, 2.0, 1.5, 1.0):
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        data = pix.tobytes("png")
        last = (data, pix.width, pix.height)
        if (
            pix.width <= MAX_DIMENSION
            and pix.height <= MAX_DIMENSION
            and len(data) <= MAX_BYTES
        ):
            return last
    assert last is not None
    if (
        last[1] > MAX_DIMENSION
        or last[2] > MAX_DIMENSION
        or len(last[0]) > MAX_BYTES
    ):
        raise RuntimeError("Visual crop remains too large after downscaling.")
    return last


def process_claim(claim: dict) -> None:
    job = claim.get("job")
    if not isinstance(job, dict):
        return
    extraction_id = str(job.get("extraction_id") or "")
    source_url = str(job.get("source_url") or "")
    regions = job.get("regions")
    if not extraction_id or not source_url or not isinstance(regions, list) or not regions:
        raise RuntimeError("Invalid visual claim.")

    with tempfile.TemporaryDirectory(prefix="eta-visual-") as temp:
        pdf_path = Path(temp) / "source.pdf"
        download_source(source_url, pdf_path)
        document = fitz.open(pdf_path)
        try:
            for region in regions:
                if not isinstance(region, dict):
                    raise RuntimeError("Invalid visual region.")
                page_number = int(region.get("page_number") or 0)
                source_ref = str(region.get("source_ref") or "").strip()
                bbox = region.get("bbox")
                if page_number < 1 or page_number > document.page_count or not source_ref or not isinstance(bbox, dict):
                    raise RuntimeError("Visual region is not traceable to the source.")
                page = document.load_page(page_number - 1)
                png, width, height = render_png(page, bbox)
                response = worker_post(
                    {
                        "action": "submit_visual_asset",
                        "extraction_id": extraction_id,
                        "source_ref": source_ref,
                        "format": "png",
                        "width": width,
                        "height": height,
                        "byte_size": len(png),
                        "data_base64": base64.b64encode(png).decode("ascii"),
                    },
                    timeout=180,
                )
                if not response.get("ok"):
                    raise RuntimeError(f"Visual submit failed for {source_ref}.")
        finally:
            document.close()

    response = worker_post(
        {"action": "finish_visual", "extraction_id": extraction_id},
        timeout=120,
    )
    if not response.get("ok"):
        raise RuntimeError("Visual materialization did not finish cleanly.")
    print(f"Materialized {len(regions)} visual region(s) for extraction {extraction_id}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-file", required=True)
    args = parser.parse_args()
    with open(args.claim_file, encoding="utf-8") as handle:
        claim = json.load(handle)
    process_claim(claim)


if __name__ == "__main__":
    main()
