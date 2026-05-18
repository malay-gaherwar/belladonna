#!/usr/bin/env python3
"""
Create self-sufficient factoids from seeded FDA Belladonna records
using a local OpenAI-compatible LLM.

Input:
    artifacts/fda/seed_from_csv/belladonna_fda_seed.jsonl

Output:
    artifacts/fda/factoids/fda_factoids.json

Behavior:
- Reads the seeded FDA JSONL
- Processes each seeded FDA record separately
- Uses only the seeded Belladonna/FDA fields
- Writes one combined output JSON file
- Optionally writes per-record JSON files
- Stores relevant metadata alongside factoids
- Logs the full run
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

def resolve_artifact_root() -> Path:
    configured = os.getenv("FDA_ARTIFACT_ROOT")
    if configured:
        return Path(configured)

    lowercase = Path("artifacts/fda")
    uppercase = Path("artifacts/FDA")
    if lowercase.exists():
        return lowercase
    if uppercase.exists():
        return uppercase
    return lowercase


ARTIFACT_ROOT = resolve_artifact_root()
INPUT_JSONL = ARTIFACT_ROOT / "seed_from_csv" / "belladonna_fda_seed.jsonl"
OUTPUT_DIR = ARTIFACT_ROOT / "factoids"
LOG_DIR = Path("logs")

OUTPUT_FILE = OUTPUT_DIR / "fda_factoids.json"
OUTPUT_JSONL = OUTPUT_DIR / "fda_factoids.jsonl"

MODEL_NAME = "GPT-OSS-120B"
CONCURRENT_REQUESTS = 50
MAX_COMPLETION_TOKENS = 1200
MAX_RECORDS = None          # set to int for testing
MAX_CHARS_PER_SECTION = 4000
WRITE_PER_RECORD_FILES = False
DEBUG_MODEL_OUTPUT = False

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

LOW_VALUE_PATTERNS = [
    r"ask a health professional before use",
    r"children younger than",
    r"does not list any boxed warning",
    r"does not list any contraindications",
    r"does not list any adverse reactions",
    r"does not list any drug interactions",
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
    return LOG_DIR / f"factoids_fda_seed_{timestamp}.log"


# -------------------------------------------------------------------
# Client
# -------------------------------------------------------------------

def get_client() -> AsyncOpenAI:
    from openai import AsyncOpenAI

    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("Missing environment variable VIRTUAL_API_KEY.")
    if not base_url:
        raise RuntimeError("Missing environment variable BASE_URL.")

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Bad JSONL line {line_num}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


# -------------------------------------------------------------------
# Record summarisation
# -------------------------------------------------------------------

def extract_summary(record: dict[str, Any], record_index: int) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    belladonna = record.get("belladonna_fields") if isinstance(record.get("belladonna_fields"), dict) else {}

    generic_name = first_nonempty(belladonna.get("generic_name"))
    brand_names = belladonna.get("brand_names") if isinstance(belladonna.get("brand_names"), list) else []
    brand_name = first_nonempty(brand_names)

    file_name = first_nonempty(
        metadata.get("file_name"),
        f"record_{record_index}.json",
    )

    return {
        "record_index": record_index,
        "file_name": file_name,
        "generic_name": generic_name,
        "brand_name": brand_name,
        "brand_names": brand_names,
        "drug_class": first_nonempty(belladonna.get("drug_class")),
        "application_number": first_nonempty(metadata.get("application_number")),
        "manufacturer_name": first_nonempty(metadata.get("manufacturer_name")),
        "source_label_set_id": first_nonempty(metadata.get("source_label_set_id")),
        "source_label_id": first_nonempty(metadata.get("source_label_id")),
        "spl_id": first_nonempty(metadata.get("spl_id")),
        "spl_set_id": first_nonempty(metadata.get("spl_set_id")),
        "label_date": first_nonempty(belladonna.get("label_date")),
        "fda_bc_indication": first_nonempty(belladonna.get("fda_bc_indication")),
        "ema_bc_indication": first_nonempty(belladonna.get("ema_bc_indication")),
        "fda_ema_status": first_nonempty(belladonna.get("fda_ema_status")),
        "use_type": first_nonempty(belladonna.get("use_type")),
        "bc_notes": first_nonempty(belladonna.get("bc_notes")),
        "indications_and_usage": truncate(first_nonempty(belladonna.get("indications_and_usage"))),
        "dosage_and_administration": truncate(first_nonempty(belladonna.get("dosage_and_administration"))),
        "warnings": truncate(first_nonempty(belladonna.get("warnings"))),
        "black_box_warning": truncate(first_nonempty(belladonna.get("black_box_warning"))),
        "pregnancy_or_breastfeeding": truncate(first_nonempty(belladonna.get("pregnancy_or_breastfeeding"))),
        "adverse_reactions": truncate(first_nonempty(belladonna.get("adverse_reactions"))),
        "special_populations": truncate(first_nonempty(belladonna.get("special_populations"))),
        "contraindications": truncate(first_nonempty(belladonna.get("contraindications"))),
        "drug_interactions": truncate(first_nonempty(belladonna.get("drug_interactions"))),
        "last_updated": first_nonempty(metadata.get("last_updated")),
        "reviewed": metadata.get("reviewed"),
        "data_source": first_nonempty(metadata.get("data_source")),
        "source_family": first_nonempty(metadata.get("source_family")),
        "source_format": first_nonempty(metadata.get("source_format")),
        "metadata_link_status": first_nonempty(metadata.get("metadata_link_status")),
        "metadata_link_method": first_nonempty(metadata.get("metadata_link_method")),
        "metadata_match_candidates": metadata.get("metadata_match_candidates", 0),
    }


# -------------------------------------------------------------------
# Prompting
# -------------------------------------------------------------------

def build_prompt(summary: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a biomedical factoid generator working from FDA breast-cancer seed records. "
        "Create only high-value, self-sufficient factual statements. "
        "Each factoid must explicitly name the drug. "
        "Use only the supplied metadata and FDA/Belladonna fields. "
        "Do not invent facts. "
        "Do not generate factoids about missing sections or the absence of warnings/interactions. "
        "Do not generate trivial consumer-use instructions unless they are clinically important. "
        "Prioritize breast-cancer-relevant facts, especially indication, biomarker, disease setting, "
        "treatment line, combination therapy, major safety warnings, pregnancy/lactation, special populations, "
        "drug interactions, and regulatory identifiers. "
        f"Return 2 to 6 factoids, each wrapped exactly as {FACTOID_START}fact text{FACTOID_END}. "
        "Return nothing except factoids."
    )

    user_prompt = (
        f"File name: {summary['file_name'] or 'N/A'}\n"
        f"Generic name: {summary['generic_name'] or 'N/A'}\n"
        f"Brand name: {summary['brand_name'] or 'N/A'}\n"
        f"Brand names: {', '.join(summary['brand_names']) if summary['brand_names'] else 'N/A'}\n"
        f"Drug class: {summary['drug_class'] or 'N/A'}\n"
        f"Application number: {summary['application_number'] or 'N/A'}\n"
        f"Manufacturer name: {summary['manufacturer_name'] or 'N/A'}\n"
        f"Label date: {summary['label_date'] or 'N/A'}\n"
        f"FDA BC indication: {summary['fda_bc_indication'] or 'N/A'}\n"
        f"EMA BC indication: {summary['ema_bc_indication'] or 'N/A'}\n"
        f"FDA EMA status: {summary['fda_ema_status'] or 'N/A'}\n"
        f"Use type: {summary['use_type'] or 'N/A'}\n"
        f"BC notes: {summary['bc_notes'] or 'N/A'}\n"
        f"Source label set ID: {summary['source_label_set_id'] or 'N/A'}\n"
        f"Source label ID: {summary['source_label_id'] or 'N/A'}\n"
        f"SPL ID: {summary['spl_id'] or 'N/A'}\n"
        f"SPL set ID: {summary['spl_set_id'] or 'N/A'}\n\n"
        f"INDICATIONS AND USAGE:\n{summary['indications_and_usage'] or 'N/A'}\n\n"
        f"DOSAGE AND ADMINISTRATION:\n{summary['dosage_and_administration'] or 'N/A'}\n\n"
        f"WARNINGS:\n{summary['warnings'] or 'N/A'}\n\n"
        f"BLACK BOX WARNING:\n{summary['black_box_warning'] or 'N/A'}\n\n"
        f"PREGNANCY OR BREASTFEEDING:\n{summary['pregnancy_or_breastfeeding'] or 'N/A'}\n\n"
        f"ADVERSE REACTIONS:\n{summary['adverse_reactions'] or 'N/A'}\n\n"
        f"SPECIAL POPULATIONS:\n{summary['special_populations'] or 'N/A'}\n\n"
        f"CONTRAINDICATIONS:\n{summary['contraindications'] or 'N/A'}\n\n"
        f"DRUG INTERACTIONS:\n{summary['drug_interactions'] or 'N/A'}\n\n"
        "Generate self-sufficient factoids only from this seeded FDA breast-cancer record."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_factoids(text: str) -> list[str]:
    raw = re.findall(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        text,
        flags=re.DOTALL,
    )

    factoids: list[str] = []
    seen: set[str] = set()

    for item in raw:
        fact = normalize_whitespace(strip_html_entities(item))
        fact = fact.strip(" -•\t\r\n")
        if not fact:
            continue

        if fact[-1] not in ".!?":
            fact += "."

        if any(re.search(pat, fact, flags=re.I) for pat in LOW_VALUE_PATTERNS):
            continue

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
            "document_title": summary.get("generic_name") or summary.get("brand_name") or "FDA Seed Record",
            "document_type": "Drug Label Seed Record",
            "document_year": datetime.now().year,
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
            if DEBUG_MODEL_OUTPUT:
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
    record: dict[str, Any],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
) -> None:
    summary = extract_summary(record, index)
    display_name = summary["generic_name"] or summary["brand_name"] or f"record_{index}"

    print("\n==============================")
    print(f"INDEX: {index}")
    print(f"DRUG: {display_name}")
    print(f"APPLICATION: {summary['application_number']}")
    print(f"MANUFACTURER: {summary['manufacturer_name']}")
    print(f"LABEL DATE: {summary['label_date']}")
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
        "file_name": summary["file_name"],
        "generic_name": summary["generic_name"],
        "brand_name": summary["brand_name"],
        "drug_class": summary["drug_class"],
        "application_number": summary["application_number"],
        "manufacturer_name": summary["manufacturer_name"],
        "label_date": summary["label_date"],
        "source_label_set_id": summary["source_label_set_id"],
        "source_label_id": summary["source_label_id"],
        "spl_id": summary["spl_id"],
        "spl_set_id": summary["spl_set_id"],
        "fda_bc_indication": summary["fda_bc_indication"],
        "ema_bc_indication": summary["ema_bc_indication"],
        "fda_ema_status": summary["fda_ema_status"],
        "use_type": summary["use_type"],
        "bc_notes": summary["bc_notes"],
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
        out_name = f"{slugify(display_name)}__{slugify(summary['source_label_set_id'] or str(index))}.json"
        payload = build_per_record_file_payload(summary, factoids, out_name)

        per_record_dir = OUTPUT_DIR / "per_record"
        per_record_dir.mkdir(parents=True, exist_ok=True)
        (per_record_dir / out_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[DONE] {display_name} | factoids={len(factoids)}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

async def async_main() -> int:
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = get_log_file()
    sys.stdout = Tee(log_file)

    print(f"Log file: {log_file}")
    print(f"Model: {MODEL_NAME}")
    print(f"Input JSONL: {INPUT_JSONL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Output JSONL: {OUTPUT_JSONL}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"Write per record files: {WRITE_PER_RECORD_FILES}")

    try:
        client = get_client()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not INPUT_JSONL.exists():
        print(f"[ERROR] Input JSONL does not exist: {INPUT_JSONL}")
        return 1

    records = load_jsonl(INPUT_JSONL)

    if MAX_RECORDS is not None:
        records = records[:MAX_RECORDS]

    if not records:
        print("[WARN] No input records found.")
        return 0

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    results_lock = asyncio.Lock()
    results: list[dict[str, Any]] = []

    tasks = [
        process_item(
            index=i + 1,
            record=record,
            client=client,
            semaphore=semaphore,
            results=results,
            results_lock=results_lock,
        )
        for i, record in enumerate(records)
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
                    "drug_class": record.get("drug_class", ""),
                    "application_number": record.get("application_number", ""),
                    "manufacturer_name": record.get("manufacturer_name", ""),
                    "label_date": record.get("label_date", ""),
                    "source_label_set_id": record.get("source_label_set_id", ""),
                    "source_label_id": record.get("source_label_id", ""),
                    "spl_id": record.get("spl_id", ""),
                    "spl_set_id": record.get("spl_set_id", ""),
                    "fda_bc_indication": record.get("fda_bc_indication", ""),
                    "ema_bc_indication": record.get("ema_bc_indication", ""),
                    "fda_ema_status": record.get("fda_ema_status", ""),
                    "use_type": record.get("use_type", ""),
                    "bc_notes": record.get("bc_notes", ""),
                    "metadata_link_status": record.get("metadata_link_status", ""),
                    "metadata_link_method": record.get("metadata_link_method", ""),
                    "metadata_match_candidates": record.get("metadata_match_candidates", 0),
                }
            )
            global_id += 1

    final_payload = {
        "metadata": {
            "source_family": "FDA",
            "document_title": "FDA Seeded Breast Cancer Drug Labels",
            "document_type": "Drug Label Seed Records",
            "document_year": datetime.now().year,
            "file_name": OUTPUT_FILE.name,
            "model_name": MODEL_NAME,
            "classification_date": datetime.now().strftime("%Y-%m-%d"),
            "input_jsonl": str(INPUT_JSONL),
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
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for factoid in flat_factoids:
            f.write(json.dumps(factoid, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time

    print("\n======================================")
    print("FDA SEEDED DATASET DONE")
    print(f"Records processed: {len(results)}")
    print(f"Factoids created: {len(flat_factoids)}")
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Wrote: {OUTPUT_JSONL}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print("======================================")

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
