#!/usr/bin/env python3
"""
Parse the first AGO REF PDF in:
  /home/malay/Documents/Belladonna/belladonna/AGO_REF_PDF/E_REF_PDF

Extract:
- page text (PyMuPDF)
- tables (pdfplumber; vision fallback)
- flowcharts/figures (Qwen3-VL via OpenAI-compatible local endpoint)

Write JSON to:
  /home/malay/Documents/Belladonna/belladonna/artifacts/ago/<pdf_stem>.json

Env (as you specified):
  export BASE_URL="http://localhost:8000/v1"
  export VIRTUAL_API_KEY="..."
Model (fixed below):
  Qwen3-VL-235B-A22B-Thinking-FP8
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
import requests
from PIL import Image


# ----------------------------
# Config
# ----------------------------
PDF_DIR = Path("/home/malay/Documents/Belladonna/belladonna/AGO_REF_PDF/E_REF_PDF")
OUT_DIR = Path("/home/malay/Documents/Belladonna/belladonna/artifacts/ago")

# Use your env variable names:
BASE_URL = (os.environ.get("BASE_URL") or "").rstrip("/")
API_KEY = os.environ.get("VIRTUAL_API_KEY") or ""

MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking-FP8"

# Heuristics for deciding whether to run vision on a page
VISION_MIN_IMAGES = 5
VISION_KEYWORDS = [
    r"\bflow\b", r"\balgorithm\b", r"\bdecision\b", r"\bworkflow\b",
    r"\bIHC\b", r"\bISH\b", r"\bHER2\b", r"\bratio\b", r"\bGroup\s*\d\b",
]
VISION_MIN_TEXT_CHARS = 300  # if very low text, likely figure-heavy

HEADER_FOOTER_JUNK = [
    "© AGO e. V.",
    "in der DGGG e.V.",
    "in der DKG e.V.",
    "www.ago-online.de",
]

SECTION_PREFIXES = [
    "Preanalytics",
    "Workup",
    "Reporting",
    "Predictive Pathology",
    "Additional Special Studies",
    "Special Studies",
    "Quality",
    "Diagnosis",
    "Therapy",
]


# ----------------------------
# Helpers
# ----------------------------
def find_first_pdf(pdf_dir: Path) -> Path:
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")
    return pdfs[0]


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty
    lines = [ln for ln in lines if ln not in HEADER_FOOTER_JUNK]
    return lines


def detect_page_heading(lines: List[str]) -> Optional[str]:
    if not lines:
        return None

    # Prefer explicit section prefix + colon
    for ln in lines[:50]:
        for prefix in SECTION_PREFIXES:
            if ln.startswith(prefix + ":"):
                return ln

    # Next: any "Something: Something" that looks like a heading
    heading_like = re.compile(r"^[A-Za-z][A-Za-z0-9 \-/&().]{2,60}:\s+.+$")
    for ln in lines[:80]:
        if heading_like.match(ln) and "http" not in ln.lower():
            return ln

    return None


def section_category_from_heading(heading: str) -> str:
    for prefix in SECTION_PREFIXES:
        if heading.startswith(prefix + ":"):
            return prefix.lower().replace(" ", "_")
    return "uncategorized"


def extract_doc_metadata(doc: fitz.Document) -> Dict[str, Any]:
    first = doc[0].get_text("text")
    lines = clean_lines(first)

    title = None
    version = None
    for ln in lines:
        if "Guidelines" in ln and "Breast" in ln:
            title = ln.strip()
        m = re.search(r"Version\s+([0-9]{4}\.[0-9A-Za-z]+)", ln)
        if m:
            version = m.group(1)

    if not title:
        for ln in lines:
            if not re.fullmatch(r"\d+", ln):
                title = ln
                break

    return {
        "title": title,
        "version": version,
        "pages": len(doc),
    }


def extract_tables_pdfplumber(pdf_path: Path, page_index: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index]
        tables = None

        # Try line-based first
        try:
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 15,
                }
            )
        except Exception:
            tables = None

        # Fallback: text-based
        if not tables:
            try:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "min_words_vertical": 3,
                        "min_words_horizontal": 1,
                        "intersection_tolerance": 5,
                    }
                )
            except Exception:
                tables = None

        if not tables:
            return out

        for t in tables:
            norm_rows = []
            for row in t:
                if not row:
                    continue
                norm_rows.append([("" if c is None else str(c).strip()) for c in row])

            if len(norm_rows) < 2 or max(len(r) for r in norm_rows) < 2:
                continue

            out.append(
                {
                    "page": page_index + 1,
                    "rows": norm_rows,
                    "extraction": "pdfplumber",
                }
            )
    return out


def render_page_png(doc: fitz.Document, page_index: int, zoom: float = 2.0) -> bytes:
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")

    # Downscale if enormous
    try:
        im = Image.open(BytesIO(png_bytes))
        max_dim = max(im.size)
        if max_dim > 2600:
            scale = 2600 / max_dim
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size)
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass

    return png_bytes


def should_run_vision(page_text: str, image_count: int, tables_found: int) -> bool:
    t = (page_text or "").strip()
    if image_count >= VISION_MIN_IMAGES:
        return True
    if len(t) < VISION_MIN_TEXT_CHARS:
        return True
    if tables_found == 0 and re.search(r"\b(Oxford|LoE|GR|AGO)\b", t):
        return True
    for pat in VISION_KEYWORDS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# ----------------------------
# Qwen-VL (OpenAI-compatible) client
# ----------------------------
@dataclass
class QwenVLClient:
    base_url: str
    model: str
    api_key: str

    def is_configured(self) -> bool:
        return bool(self.base_url) and bool(self.api_key) and self.base_url.startswith("http")

    def available(self) -> bool:
        if not self.is_configured():
            return False
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return r.status_code < 500
        except Exception:
            return False

    def analyze_page(self, page_png: bytes) -> Tuple[Optional[Dict[str, Any]], str]:
        b64 = base64.b64encode(page_png).decode("utf-8")
        prompt = (
            "You are analyzing a medical guideline PDF page.\n"
            "Extract structured content. Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "tables": [ { "title": null|string, "headers": [string], "rows": [[string]] } ],\n'
            '  "flowcharts": [ { "topic": null|string, "nodes": [ { "id": string, "text": string } ],\n'
            '                  "edges": [ { "from": string, "to": string, "condition": null|string } ] } ],\n'
            '  "figures": [ { "type": string, "description": string } ]\n'
            "}\n"
            "Rules:\n"
            "- If none found, use empty lists.\n"
            "- For tables: do not invent rows.\n"
            "- For flowcharts: capture decision logic and cutoffs.\n"
        )

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        }

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        raw = data["choices"][0]["message"]["content"]
        parsed = extract_json_from_text(raw)
        return parsed, raw


# ----------------------------
# Main pipeline
# ----------------------------
def process_pdf(pdf_path: Path, out_dir: Path) -> Path:
    doc = fitz.open(str(pdf_path))
    meta = extract_doc_metadata(doc)

    qwen = QwenVLClient(BASE_URL, MODEL_NAME, API_KEY)
    qwen_ok = qwen.available()

    if not qwen_ok:
        # Helpful but not fatal
        cfg_state = f"BASE_URL set={bool(BASE_URL)}, API_KEY set={bool(API_KEY)}"
        print(f"[WARN] Qwen-VL not available; vision steps will be skipped. ({cfg_state})", file=sys.stderr)

    sections: List[Dict[str, Any]] = []
    vision_runs: List[Dict[str, Any]] = []
    warnings: List[str] = []

    current_section: Optional[Dict[str, Any]] = None

    for i in range(len(doc)):
        page = doc[i]
        page_text = page.get_text("text") or ""
        lines = clean_lines(page_text)
        heading = detect_page_heading(lines)

        if heading:
            if (current_section is None) or (current_section.get("name") != heading):
                if current_section is not None:
                    sections.append(current_section)
                current_section = {
                    "name": heading,
                    "category": section_category_from_heading(heading),
                    "pages": [],
                    "text_blocks": [],
                    "tables": [],
                    "figures": [],
                }

        if current_section is None:
            current_section = {
                "name": "front_matter",
                "category": "front_matter",
                "pages": [],
                "text_blocks": [],
                "tables": [],
                "figures": [],
            }

        current_section["pages"].append(i + 1)

        if page_text.strip():
            current_section["text_blocks"].append({"page": i + 1, "text": page_text.strip()})

        # tables
        tables: List[Dict[str, Any]] = []
        try:
            tables = extract_tables_pdfplumber(pdf_path, i)
        except Exception as e:
            warnings.append(f"pdfplumber table extraction failed on page {i+1}: {e}")

        if tables:
            current_section["tables"].extend(tables)

        # vision
        img_count = len(page.get_images(full=True))
        run_vision = should_run_vision(page_text, img_count, tables_found=len(tables))

        if run_vision and qwen_ok:
            try:
                png = render_page_png(doc, i, zoom=2.0)
                parsed, raw = qwen.analyze_page(png)

                vision_runs.append(
                    {
                        "page": i + 1,
                        "image_count": img_count,
                        "parsed_ok": parsed is not None,
                        "raw_if_parse_failed": raw if parsed is None else None,
                    }
                )

                if parsed:
                    for t in parsed.get("tables", []) or []:
                        current_section["tables"].append(
                            {"page": i + 1, "extraction": "qwen_vl", **t}
                        )
                    for fc in parsed.get("flowcharts", []) or []:
                        current_section["figures"].append(
                            {
                                "page": i + 1,
                                "type": "decision_flowchart",
                                "extraction": "qwen_vl",
                                "confidence": 0.75,
                                "flowchart": fc,
                            }
                        )
                    for fig in parsed.get("figures", []) or []:
                        current_section["figures"].append({"page": i + 1, "extraction": "qwen_vl", **fig})

            except Exception as e:
                warnings.append(f"Vision failed on page {i+1}: {e}")

        elif run_vision and not qwen_ok:
            vision_runs.append(
                {
                    "page": i + 1,
                    "image_count": img_count,
                    "parsed_ok": False,
                    "skipped": True,
                    "reason": "Qwen-VL endpoint unavailable or not configured",
                }
            )

    if current_section is not None:
        sections.append(current_section)

    output = {
        "source_pdf": str(pdf_path),
        "processed_at_utc": now_utc_iso(),
        "doc_meta": meta,
        "vision_model": {
            "enabled": qwen_ok,
            "base_url": BASE_URL,
            "model": MODEL_NAME,
        },
        "sections": sections,
        "vision_runs": vision_runs,
        "warnings": warnings,
    }

    out_path = out_dir / f"{pdf_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return out_path


def main() -> None:
    ensure_out_dir(OUT_DIR)
    pdf_path = find_first_pdf(PDF_DIR)
    out_path = process_pdf(pdf_path, OUT_DIR)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
