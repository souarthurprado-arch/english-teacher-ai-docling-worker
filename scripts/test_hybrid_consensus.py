#!/usr/bin/env python3
from hybrid_extraction import consensus


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


base = {
    "elements": [{"kind": "TEXT", "page_number": 1}],
    "text_content": "[[PAGE 1]]\nComplete exercise 1 with 4 words.",
}
matching = {
    "status": "SUCCESS",
    "elements": [{"kind": "TEXT", "page_number": 1}],
    "text_content": "[[PAGE 1]]\nComplete exercise 1 with 4 words.",
}
result = consensus(base, matching)
require(result["status"] == "AGREED", "matching page must agree")
require(result["page_consensus"][0]["status"] == "AGREED", "page must agree")

numeric_conflict = {
    "status": "SUCCESS",
    "elements": [{"kind": "TEXT", "page_number": 1}],
    "text_content": "[[PAGE 1]]\nComplete exercise 1 with 5 words.",
}
result = consensus(base, numeric_conflict)
require(result["status"] == "DISAGREED", "numeric conflict must disagree")
require(result["material_disagreement"] is True, "numeric conflict must be material")
require("p1:numeric_token_mismatch" in result["material_reasons"], "reason must identify page")

print("PASS: hybrid consensus tests")
