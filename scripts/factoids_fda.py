#!/usr/bin/env python3
"""
Create self-sufficient factoids from merged FDA label JSON files
using a local OpenAI-compatible LLM.

Behavior:
- Reads merged FDA JSON files from artifacts/fda/merged
- Processes each merged source file separately
- Writes one combined output JSON file
- Optionally writes per-record JSON files
- Stores relevant metadata alongside factoids
- Logs the full run

Expected merged input structure:
{
  "metadata_link_status": "...",
  "metadata_link_method": "...",
  "metadata_match_candidates": 0,
  "metadata": {...} | null,
  "label": {...}
}

Requirements:
    pip install openai

Environment variables expected:
    VIRTUAL_API_KEY
    BASE_URL
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

from openai import AsyncOpenAI

INPUT_DIR = Path("artifacts/fda/merged")
OUTPUT_DIR = Path("artifacts/fda/factoids")
LOG_DIR = Path("logs")

OUTPUT_FILE = OUTPUT_DIR / "fda_factoids.json"

MODEL_NAME = "GPT-OSS-120B"
CONCURRENT_REQUESTS = 50
MAX_COMPLETION_TOKENS = 1200
MAX_FILES = 50          # set to an int for testing, e.g. 50
MAX_CHARS_PER_SECTION = 4000
WRITE_PER_RECORD_FILES = False
ONLY_WITH_METADATA = False   # True = skip unmatched metadata records
ONLY_BREAST_CANCER = False   # True = filter to likely breast-cancer-relevant FDA files only

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

BREAST_CANCER_KEYWORDS = [
    "breast cancer",
    "breast carcinoma",
    "her2-positive",
    "her2 positive",
    "triple negative",
    "tnbc",
    "hr-positive",
    "hr positive",
    "hormone receptor",
    "metastatic breast",
    "early breast",
    "breast neoplasm",
]

LABEL_FIELDS_FOR_PROMPT = [
    "indications_and_usage",
    "dosage_and_administration",
    "boxed_warning",
    "warnings",
    "warnings_and_cautions",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
    "pregnancy",
    "pregnancy_or_breast_feeding",
    "breastfeeding",
    "pediatric_use",
    "geriatric_use",
    "renal_impairment",
    "hepatic_impairment",
    "use_in_specific_populations",
    "special_populations",
    "clinical_pharmacology",
    "description",
]


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

class Tee:
    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def get_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"factoids_fda_{timestamp}.log"


# -------------------------------------------------------------------
# Client
# -------------------------------------------------------------------

def get_client() -> AsyncOpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError(
            "Missing environment variable VIRTUAL_API_KEY. "
            "Make sure it is exported in your shell."
        )
    if not base_url:
        raise RuntimeError(
            "Missing environment variable BASE_URL. "
            "Make sure it is exported in your shell."
        )

    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html_entities(text: str) -> str:
    text = text.replace("&nbsp;", " ")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")
    return normalize_whitespace(text)


def slugify(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return strip_html_entities(value)
    if isinstance(value, (list, tuple, set)):
        return normalize_whitespace(" ".join(norm_text(v) for v in value if v is not None))
    if isinstance(value, dict):
        return normalize_whitespace(" ".join(norm_text(v) for v in value.values() if v is not None))
    return normalize_whitespace(str(value))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = norm_text(value)
        if text:
            return text
    return ""


def truncate(text: str, max_chars: int = MAX_CHARS_PER_SECTION) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def join_label_field(label: dict[str, Any], key: str) -> str:
    value = label.get(key)
    if value is None:
        return ""
    if isinstance(value, list):
        return truncate(" ".join(norm_text(v) for v in value if v is not None))
    return truncate(norm_text(value))


def infer_document_year(record: dict[str, Any]) -> int:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}

    candidates = [
        metadata.get("effective_time"),
        label.get("effective_time"),
    ]

    for value in candidates:
        text = norm_text(value)
        if not text:
            continue
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return int(m.group(0))
        if re.fullmatch(r"\d{8}", text):
            return int(text[:4])

    return datetime.now().year


def safe_filename_from_record(record: dict[str, Any], index: int) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}

    generic_name = first_nonempty(
        metadata.get("generic_name"),
        metadata.get("brand_name"),
        label.get("openfda", {}).get("generic_name") if isinstance(label.get("openfda"), dict) else None,
        label.get("openfda", {}).get("brand_name") if isinstance(label.get("openfda"), dict) else None,
    )

    set_id = first_nonempty(
        metadata.get("source_label_set_id"),
        label.get("set_id"),
        label.get("id"),
        f"record_{index}",
    )

    return f"{slugify(generic_name)}__{slugify(set_id)}.json"


def is_likely_breast_cancer_relevant(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}

    haystack_parts: list[str] = []

    for key in ("generic_name", "brand_name", "product_type", "route"):
        haystack_parts.append(norm_text(metadata.get(key)))

    for key in ("substance_names",):
        haystack_parts.append(norm_text(metadata.get(key)))

    for key in ("indications_and_usage", "description", "clinical_pharmacology"):
        haystack_parts.append(join_label_field(label, key))

    haystack = " ".join(x.lower() for x in haystack_parts if x)

    return any(keyword in haystack for keyword in BREAST_CANCER_KEYWORDS)


# -------------------------------------------------------------------
# Record summarisation
# -------------------------------------------------------------------

def extract_summary(record: dict[str, Any], file_name: str) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

    generic_name = first_nonempty(
        metadata.get("generic_name"),
        openfda.get("generic_name"),
        metadata.get("brand_name"),
        openfda.get("brand_name"),
    )

    brand_name = first_nonempty(
        metadata.get("brand_name"),
        openfda.get("brand_name"),
    )

    application_number = first_nonempty(
        metadata.get("application_number"),
        openfda.get("application_number"),
    )

    manufacturer_name = first_nonempty(
        metadata.get("manufacturer_name"),
        openfda.get("manufacturer_name"),
        metadata.get("sponsor_name"),
    )

    route = first_nonempty(
        metadata.get("route"),
        openfda.get("route"),
    )

    substance_names = norm_text(
        metadata.get("substance_names") or openfda.get("substance_name")
    )

    summary = {
        "file_name": file_name,
        "metadata_link_status": record.get("metadata_link_status", ""),
        "metadata_link_method": record.get("metadata_link_method", ""),
        "metadata_match_candidates": record.get("metadata_match_candidates", 0),

        "generic_name": generic_name,
        "brand_name": brand_name,
        "application_number": application_number,
        "manufacturer_name": manufacturer_name,
        "product_type": first_nonempty(metadata.get("product_type"), openfda.get("product_type")),
        "route": route,
        "substance_names": substance_names,

        "source_label_set_id": first_nonempty(metadata.get("source_label_set_id"), label.get("set_id")),
        "source_label_id": first_nonempty(metadata.get("source_label_id"), label.get("id")),
        "spl_id": first_nonempty(metadata.get("spl_id"), openfda.get("spl_id")),
        "spl_set_id": first_nonempty(metadata.get("spl_set_id"), openfda.get("spl_set_id")),

        "effective_time": first_nonempty(metadata.get("effective_time"), label.get("effective_time")),
        "document_year": infer_document_year(record),

        "indications_and_usage": join_label_field(label, "indications_and_usage"),
        "dosage_and_administration": join_label_field(label, "dosage_and_administration"),
        "boxed_warning": join_label_field(label, "boxed_warning"),
        "warnings": join_label_field(label, "warnings") or join_label_field(label, "warnings_and_cautions"),
        "contraindications": join_label_field(label, "contraindications"),
        "adverse_reactions": join_label_field(label, "adverse_reactions"),
        "drug_interactions": join_label_field(label, "drug_interactions"),
        "pregnancy_breastfeeding": " ".join(
            x for x in [
                join_label_field(label, "pregnancy"),
                join_label_field(label, "pregnancy_or_breast_feeding"),
                join_label_field(label, "breastfeeding"),
            ] if x
        ).strip(),
        "special_populations": " ".join(
            x for x in [
                join_label_field(label, "pediatric_use"),
                join_label_field(label, "geriatric_use"),
                join_label_field(label, "renal_impairment"),
                join_label_field(label, "hepatic_impairment"),
                join_label_field(label, "use_in_specific_populations"),
                join_label_field(label, "special_populations"),
            ] if x
        ).strip(),
        "clinical_pharmacology": join_label_field(label, "clinical_pharmacology"),
        "description": join_label_field(label, "description"),
        "how_supplied": join_label_field(label, "how_supplied"),
    }

    return summary


# -------------------------------------------------------------------
# Prompting
# -------------------------------------------------------------------

def build_prompt(summary: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a biomedical factoid generator working from FDA drug label records. "
        "Your job is to create short, self-sufficient factual statements based only on the supplied FDA metadata and label text. "
        "Each factoid must stand on its own without requiring the source record for context. "
        "Always name the drug explicitly in every factoid using the generic name or brand name. "
        "Do not invent facts. Do not use outside medical knowledge. "
        "Prefer breast-cancer-relevant facts when present, but if the record is not breast-cancer-specific, still generate factual label-based factoids from the FDA text provided. "
        "Focus on indication, treatment setting, biomarker or subtype if explicitly stated, dosage context if clearly label-based, major warnings, contraindications, adverse reactions, pregnancy or breastfeeding, special populations, approval identifiers, and manufacturer/application metadata when present. "
        "Do not quote huge passages. Summarize tightly. "
        "Return between 2 and 8 factoids. "
        f"Wrap each factoid exactly like this: {FACTOID_START}fact text{FACTOID_END} "
        "Return nothing except the factoids."
    )

    user_prompt = (
        f"File name: {summary['file_name'] or 'N/A'}\n"
        f"Metadata link status: {summary['metadata_link_status'] or 'N/A'}\n"
        f"Metadata link method: {summary['metadata_link_method'] or 'N/A'}\n"
        f"Metadata match candidates: {summary['metadata_match_candidates']}\n"
        f"Generic name: {summary['generic_name'] or 'N/A'}\n"
        f"Brand name: {summary['brand_name'] or 'N/A'}\n"
        f"Application number: {summary['application_number'] or 'N/A'}\n"
        f"Manufacturer name: {summary['manufacturer_name'] or 'N/A'}\n"
        f"Product type: {summary['product_type'] or 'N/A'}\n"
        f"Route: {summary['route'] or 'N/A'}\n"
        f"Substance names: {summary['substance_names'] or 'N/A'}\n"
        f"Label effective time: {summary['effective_time'] or 'N/A'}\n"
        f"Source label set ID: {summary['source_label_set_id'] or 'N/A'}\n"
        f"Source label ID: {summary['source_label_id'] or 'N/A'}\n"
        f"SPL ID: {summary['spl_id'] or 'N/A'}\n"
        f"SPL set ID: {summary['spl_set_id'] or 'N/A'}\n\n"

        f"INDICATIONS AND USAGE:\n{summary['indications_and_usage'] or 'N/A'}\n\n"
        f"DOSAGE AND ADMINISTRATION:\n{summary['dosage_and_administration'] or 'N/A'}\n\n"
        f"BOXED WARNING:\n{summary['boxed_warning'] or 'N/A'}\n\n"
        f"WARNINGS:\n{summary['warnings'] or 'N/A'}\n\n"
        f"CONTRAINDICATIONS:\n{summary['contraindications'] or 'N/A'}\n\n"
        f"ADVERSE REACTIONS:\n{summary['adverse_reactions'] or 'N/A'}\n\n"
        f"DRUG INTERACTIONS:\n{summary['drug_interactions'] or 'N/A'}\n\n"
        f"PREGNANCY / BREASTFEEDING:\n{summary['pregnancy_breastfeeding'] or 'N/A'}\n\n"
        f"SPECIAL POPULATIONS:\n{summary['special_populations'] or 'N/A'}\n\n"
        f"CLINICAL PHARMACOLOGY:\n{summary['clinical_pharmacology'] or 'N/A'}\n\n"
        f"DESCRIPTION:\n{summary['description'] or 'N/A'}\n\n"
        f"HOW SUPPLIED:\n{summary['how_supplied'] or 'N/A'}\n\n"

        "Generate self-sufficient factoids only from this FDA record."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_factoids(text: str) -> list[str]:
    # First pass: normal extraction
    raw = re.findall(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        text,
        flags=re.DOTALL,
    )

    chunks: list[str] = []

    # If model nested additional markers inside one capture,
    # split again conservatively.
    for item in raw:
        inner = re.split(re.escape(FACTOID_END) + r"\s*" + re.escape(FACTOID_START), item)
        for part in inner:
            part = part.strip()
            if part:
                chunks.append(part)

    factoids: list[str] = []
    seen: set[str] = set()

    for item in chunks:
        fact = normalize_whitespace(strip_html_entities(item))
        fact = re.sub(rf"^{re.escape(FACTOID_START)}", "", fact).strip()
        fact = re.sub(rf"{re.escape(FACTOID_END)}$", "", fact).strip()
        fact = fact.strip(" -•\t\r\n")
        if not fact:
            continue

        if fact[-1] not in ".!?":
            fact += "."

        key = fact.lower()
        if key not in seen:
            seen.add(key)
            factoids.append(fact)

    return factoids


# -------------------------------------------------------------------
# Output builders
# -------------------------------------------------------------------

def build_per_record_file_payload(
    summary: dict[str, Any],
    factoids: list[str],
    out_name: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "source_family": "FDA",
            "document_title": summary.get("generic_name") or summary.get("brand_name") or "FDA Label Record",
            "document_type": "Drug Label",
            "document_year": summary.get("document_year"),
            "file_name": out_name,
            "model_name": MODEL_NAME,
            "classification_date": datetime.now().strftime("%Y-%m-%d"),
        },
        "record_metadata": summary,
        "factoids": [
            {"id": i + 1, "factoid_text": fact}
            for i, fact in enumerate(factoids)
        ],
    }


# -------------------------------------------------------------------
# LLM call
# -------------------------------------------------------------------

async def generate_factoids_for_record(
    client: AsyncOpenAI,
    summary: dict[str, Any],
    model_name: str = MODEL_NAME,
) -> list[str]:
    last_error = None

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                messages=build_prompt(summary),
                model=model_name,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            content = response.choices[0].message.content or ""
            print(f"[DEBUG RAW MODEL OUTPUT] {repr(content[:1000])}")

            factoids = parse_factoids(content)
            if factoids:
                return factoids

            return []

        except Exception as exc:
            last_error = exc
            print(
                f"[WARN] attempt {attempt + 1}/3 failed for "
                f"{summary.get('generic_name') or summary.get('brand_name') or 'UNKNOWN'}: {exc}"
            )
            await asyncio.sleep(2)

    raise RuntimeError(f"LLM request failed after 3 attempts: {last_error}")


# -------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------

async def process_item(
    index: int,
    path: Path,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
) -> None:
    try:
        record = load_json(path)
    except Exception as exc:
        print(f"[ERROR] Could not read {path.name}: {exc}")
        return

    if not isinstance(record, dict):
        print(f"[WARN] Skipping non-dict record: {path.name}")
        return

    if ONLY_WITH_METADATA and not isinstance(record.get("metadata"), dict):
        print(f"[SKIP] No metadata: {path.name}")
        return

    if ONLY_BREAST_CANCER and not is_likely_breast_cancer_relevant(record):
        print(f"[SKIP] Not breast-cancer-relevant: {path.name}")
        return

    summary = extract_summary(record, path.name)
    display_name = summary["generic_name"] or summary["brand_name"] or path.stem

    print("\n==============================")
    print(f"INDEX: {index}")
    print(f"FILE: {path.name}")
    print(f"DRUG: {display_name}")
    print(f"APPLICATION: {summary['application_number']}")
    print(f"MANUFACTURER: {summary['manufacturer_name']}")
    print(f"EFFECTIVE TIME: {summary['effective_time']}")
    print(f"INDICATIONS: {summary['indications_and_usage'][:1000]}")
    print("==============================\n")

    try:
        async with semaphore:
            factoids = await generate_factoids_for_record(
                client=client,
                summary=summary,
            )
    except Exception as exc:
        print(f"[ERROR] Factoid generation failed for {display_name}: {exc}")
        factoids = []

    result = {
        "record_index": index,
        "file_name": path.name,
        "generic_name": summary["generic_name"],
        "brand_name": summary["brand_name"],
        "application_number": summary["application_number"],
        "manufacturer_name": summary["manufacturer_name"],
        "effective_time": summary["effective_time"],
        "source_label_set_id": summary["source_label_set_id"],
        "source_label_id": summary["source_label_id"],
        "spl_id": summary["spl_id"],
        "spl_set_id": summary["spl_set_id"],
        "metadata_link_status": summary["metadata_link_status"],
        "metadata_link_method": summary["metadata_link_method"],
        "metadata_match_candidates": summary["metadata_match_candidates"],
        "source_summary": summary,
        "factoids": [
            {"id": i + 1, "factoid_text": fact}
            for i, fact in enumerate(factoids)
        ],
    }

    async with results_lock:
        results.append(result)

    if WRITE_PER_RECORD_FILES and factoids:
        out_name = safe_filename_from_record(record, index)
        payload = build_per_record_file_payload(summary, factoids, out_name)

        per_record_dir = OUTPUT_DIR / "per_record"
        per_record_dir.mkdir(parents=True, exist_ok=True)
        (per_record_dir / out_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[DONE] {display_name} | factoids={len(factoids)}")


async def async_main() -> int:
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = get_log_file()
    sys.stdout = Tee(log_file)

    print(f"Log file: {log_file}")
    print(f"Model: {MODEL_NAME}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"Only with metadata: {ONLY_WITH_METADATA}")
    print(f"Only breast cancer: {ONLY_BREAST_CANCER}")
    print(f"Write per record files: {WRITE_PER_RECORD_FILES}")

    try:
        client = get_client()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input dir does not exist: {INPUT_DIR}")
        return 1

    paths = [
        p for p in sorted(INPUT_DIR.glob("*.json"))
        if p.name != "summary.json"
    ]

    if MAX_FILES is not None:
        paths = paths[:MAX_FILES]

    if not paths:
        print("[WARN] No input JSON files found.")
        return 0

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    results_lock = asyncio.Lock()
    results: list[dict[str, Any]] = []

    tasks = [
        process_item(
            index=i + 1,
            path=path,
            client=client,
            semaphore=semaphore,
            results=results,
            results_lock=results_lock,
        )
        for i, path in enumerate(paths)
    ]

    await asyncio.gather(*tasks)

    results.sort(
        key=lambda x: (
            (x.get("generic_name") or "").lower(),
            (x.get("brand_name") or "").lower(),
            (x.get("file_name") or "").lower(),
        )
    )

    flat_factoids: list[dict[str, Any]] = []
    global_id = 1

    for record in results:
        for fact in record["factoids"]:
            flat_factoids.append(
                {
                    "id": global_id,
                    "factoid_text": fact["factoid_text"],
                    "file_name": record.get("file_name", ""),
                    "generic_name": record.get("generic_name", ""),
                    "brand_name": record.get("brand_name", ""),
                    "application_number": record.get("application_number", ""),
                    "manufacturer_name": record.get("manufacturer_name", ""),
                    "effective_time": record.get("effective_time", ""),
                    "source_label_set_id": record.get("source_label_set_id", ""),
                    "source_label_id": record.get("source_label_id", ""),
                    "spl_id": record.get("spl_id", ""),
                    "spl_set_id": record.get("spl_set_id", ""),
                    "metadata_link_status": record.get("metadata_link_status", ""),
                    "metadata_link_method": record.get("metadata_link_method", ""),
                    "metadata_match_candidates": record.get("metadata_match_candidates", 0),
                }
            )
            global_id += 1

    final_payload = {
        "metadata": {
            "source_family": "FDA",
            "document_title": "FDA Merged Drug Labels",
            "document_type": "Drug Labels",
            "document_year": datetime.now().year,
            "file_name": OUTPUT_FILE.name,
            "model_name": MODEL_NAME,
            "classification_date": datetime.now().strftime("%Y-%m-%d"),
            "input_dir": str(INPUT_DIR),
            "records_processed": len(results),
            "factoids_created": len(flat_factoids),
        },
        "factoids": flat_factoids,
        "record_level_outputs": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_time = time.time() - start_time

    print("\n======================================")
    print("FDA DATASET DONE")
    print(f"Records processed: {len(results)}")
    print(f"Factoids created: {len(flat_factoids)}")
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print("======================================")

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())