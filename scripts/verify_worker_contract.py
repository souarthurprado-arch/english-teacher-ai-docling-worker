from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")

workflow = read(".github/workflows/docling-worker.yml")
worker = read("scripts/docling_actions_worker.py")
hybrid = read("scripts/hybrid_extraction.py")

require("workflow_dispatch:" in workflow, "worker must be explicitly dispatchable")
require("\n  schedule:" not in workflow and "\n  push:" not in workflow, "worker must not poll or run on push")
require("id-token: write" in workflow, "worker must use short-lived GitHub OIDC")
require("quay.io/docling-project/docling-serve-cpu:v1.29.0" in workflow, "Docling image must stay pinned")
require("secrets." not in workflow, "workflow must not depend on repository Actions secrets")
require("pypdf==6.19.0" in workflow, "PDF metadata reader must stay pinned")
require("Pillow==11.3.0" in workflow, "Pillow must stay pinned")
require("pymupdf4llm==1.28.2" in workflow and "PyMuPDF==1.26.4" in workflow, "hybrid PDF dependencies must stay pinned")
require("AGPL|COMMERCIAL" in workflow, "PyMuPDF4LLM must remain license-gated")

for needle, label in [
    ("Skipping already completed batch", "resume completed batches"),
    ("PREVIEW_MAX_EDGE = 1440", "bounded preview dimensions"),
    ("PREVIEW_WEBP_QUALITY = 76", "bounded preview quality"),
    ('"table_mode": "accurate"', "accurate table extraction"),
    ("MANUAL_CURATION_V2", "manual curation contract"),
    ("attach_hybrid_evidence", "hybrid evidence"),
    ("force_ocr=True", "forced OCR reconstruction"),
    ("preserve_text_geometry", "text geometry reconstruction"),
    ("submit_reconstruction", "reconstruction submission"),
    ("finish_reconstruction", "reconstruction completion"),
]:
    require(needle in worker, f"worker must preserve {label}")

for needle, label in [
    ("PYMUPDF4LLM_LICENSE_MODE", "explicit PyMuPDF4LLM license mode"),
    ("AGREED", "agreed consensus state"),
    ("COMPLEMENTARY", "complementary consensus state"),
    ("DISAGREED", "disagreed consensus state"),
    ("page_boxes", "layout boxes"),
]:
    require(needle in hybrid, f"hybrid extractor must preserve {label}")

print("PASS: Docling worker contract")
