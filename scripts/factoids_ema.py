#!/usr/bin/env python3
"""
Create self-sufficient factoids from all EMA breast-cancer-related filtered files
using a local GPT-OSS model.

Behavior:
- Reads all known EMA filtered JSON files
- Processes each source separately
- Writes one output JSON file per source
- Metadata reflects the source-specific document title
- Optionally writes per-record JSON files under subfolders
- Logs the full run to logs/factoids_ema_all_<datetime>.log

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

from openai import AsyncOpenAI


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

INPUT_DIR = Path("artifacts/ema/filtered")
OUTPUT_DIR = Path("artifacts/ema/factoids")
LOG_DIR = Path("logs")

MODEL_NAME = "GPT-OSS-120B"
CONCURRENT_REQUESTS = 50
MAX_COMPLETION_TOKENS = 1200
MAX_ITEMS_PER_FILE = None  # set to e.g. 10 for testing

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

WRITE_PER_RECORD_FILES = False

DATASET_CONFIGS = [
    {
        "input_file": INPUT_DIR / "medicines_breast_cancer_related.json",
        "source_key": "medicines",
        "source_family": "EMA",
        "document_title": "EMA Medicines",
        "document_type": "Medicines",
        "output_file": OUTPUT_DIR / "ema_medicines_factoids.json",
        "record_label": "medicine",
    },
    {
        "input_file": INPUT_DIR / "dhpcs_breast_cancer_related.json",
        "source_key": "dhpcs",
        "source_family": "EMA",
        "document_title": "EMA DHPCs",
        "document_type": "DHPCs",
        "output_file": OUTPUT_DIR / "ema_dhpcs_factoids.json",
        "record_label": "dhpc",
    },
    {
        "input_file": INPUT_DIR / "post_authorisation_breast_cancer_related.json",
        "source_key": "post_authorisation",
        "source_family": "EMA",
        "document_title": "EMA Post Authorisation",
        "document_type": "Post Authorisation",
        "output_file": OUTPUT_DIR / "ema_post_authorisation_factoids.json",
        "record_label": "post_authorisation_record",
    },
    {
        "input_file": INPUT_DIR / "psusas_breast_cancer_related.json",
        "source_key": "psusas",
        "source_family": "EMA",
        "document_title": "EMA PSUSAs",
        "document_type": "PSUSAs",
        "output_file": OUTPUT_DIR / "ema_psusas_factoids.json",
        "record_label": "psusa",
    },
    {
        "input_file": INPUT_DIR / "referrals_breast_cancer_related.json",
        "source_key": "referrals",
        "source_family": "EMA",
        "document_title": "EMA Referrals",
        "document_type": "Referrals",
        "output_file": OUTPUT_DIR / "ema_referrals_factoids.json",
        "record_label": "referral",
    },
    {
        "input_file": INPUT_DIR / "shortages_breast_cancer_related.json",
        "source_key": "shortages",
        "source_family": "EMA",
        "document_title": "EMA Shortages",
        "document_type": "Shortages",
        "output_file": OUTPUT_DIR / "ema_shortages_factoids.json",
        "record_label": "shortage",
    },
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
    return LOG_DIR / f"factoids_ema_all_{timestamp}.log"


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
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError("Input JSON must be a list or a dict with an 'items' list")


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


def first_nonempty(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = norm_text(record.get(key))
        if value:
            return value
    return ""


def infer_document_year(items: list[dict[str, Any]]) -> int:
    years: list[int] = []

    for item in items:
        record = item.get("record", {})
        if not isinstance(record, dict):
            continue

        for key in (
            "last_updated_date",
            "european_commission_decision_date",
            "marketing_authorisation_date",
            "first_published_date",
            "opinion_adopted_date",
            "post_authorisation_opinion_date",
            "withdrawal_of_application_date",
            "dissemination_date",
            "procedure_start_date",
            "cmdh_position_date",
            "chmp_cvmp_opinion_date",
        ):
            value = norm_text(record.get(key))
            if not value:
                continue

            m = re.search(r"\b(\d{4})\b", value)
            if m:
                year = int(m.group(1))
                if 1900 <= year <= 2100:
                    years.append(year)

    return max(years) if years else datetime.now().year


# -------------------------------------------------------------------
# Record summarisation across source types
# -------------------------------------------------------------------

def extract_summary(item: dict[str, Any], source_key: str) -> dict[str, str]:
    summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
    record = item.get("record", {}) if isinstance(item.get("record"), dict) else {}

    if source_key == "medicines":
        entity_name = (
            first_nonempty(summary, "entity_name", "name")
            or first_nonempty(record, "name_of_medicine", "name")
        )
        active_substance = (
            first_nonempty(summary, "active_substance")
            or first_nonempty(record, "active_substance", "international_non_proprietary_name_common_name")
        )
        return {
            "entity_name": entity_name,
            "active_substance": active_substance,
            "therapeutic_indication": (
                first_nonempty(summary, "therapeutic_indication")
                or first_nonempty(record, "therapeutic_indication")
            ),
            "therapeutic_area": (
                first_nonempty(summary, "therapeutic_area")
                or first_nonempty(record, "therapeutic_area_mesh")
            ),
            "classification": (
                first_nonempty(summary, "classification")
                or first_nonempty(record, "pharmacotherapeutic_group_human", "atc_code_human")
            ),
            "status": first_nonempty(record, "medicine_status"),
            "opinion_status": first_nonempty(record, "opinion_status"),
            "holder": first_nonempty(record, "marketing_authorisation_developer_applicant_holder"),
            "ema_number": first_nonempty(record, "ema_product_number"),
            "procedure_number": "",
            "regulatory_outcome": "",
            "first_published_date": first_nonempty(record, "first_published_date"),
            "last_updated_date": first_nonempty(record, "last_updated_date"),
            "marketing_authorisation_date": first_nonempty(record, "marketing_authorisation_date"),
            "url": first_nonempty(record, "medicine_url"),
            "extra_context": "",
        }

    if source_key == "dhpcs":
        entity_name = (
            first_nonempty(summary, "entity_name")
            or first_nonempty(record, "name_of_medicine")
        )
        active_substance = (
            first_nonempty(summary, "active_substance")
            or first_nonempty(record, "active_substances")
        )
        return {
            "entity_name": entity_name,
            "active_substance": active_substance,
            "therapeutic_indication": "",
            "therapeutic_area": (
                first_nonempty(summary, "therapeutic_area")
                or first_nonempty(record, "therapeutic_area_mesh")
            ),
            "classification": (
                first_nonempty(summary, "classification")
                or first_nonempty(record, "atc_code_human", "dhpc_type")
            ),
            "status": (
                first_nonempty(summary, "status")
                or first_nonempty(record, "dhpc_type")
            ),
            "opinion_status": "",
            "holder": "",
            "ema_number": "",
            "procedure_number": first_nonempty(record, "procedure_number"),
            "regulatory_outcome": first_nonempty(record, "regulatory_outcome"),
            "first_published_date": first_nonempty(record, "first_published_date"),
            "last_updated_date": first_nonempty(record, "last_updated_date"),
            "marketing_authorisation_date": "",
            "url": first_nonempty(record, "dhpc_url"),
            "extra_context": first_nonempty(summary, "extra_context", "referral_name"),
        }

    if source_key == "post_authorisation":
        entity_name = (
            first_nonempty(summary, "entity_name")
            or first_nonempty(record, "name_of_medicine")
        )
        active_substance = (
            first_nonempty(summary, "active_substance")
            or first_nonempty(record, "active_substance", "international_non_proprietary_name_common_name")
        )
        return {
            "entity_name": entity_name,
            "active_substance": active_substance,
            "therapeutic_indication": "",
            "therapeutic_area": (
                first_nonempty(summary, "therapeutic_area")
                or first_nonempty(record, "therapeutic_area_mesh")
            ),
            "classification": (
                first_nonempty(summary, "classification")
                or first_nonempty(record, "atc_code_human")
            ),
            "status": first_nonempty(record, "post_authorisation_procedure_status"),
            "opinion_status": first_nonempty(record, "post_authorisation_opinion_status"),
            "holder": first_nonempty(record, "marketing_authorisation_developer_applicant_holder"),
            "ema_number": first_nonempty(record, "ema_product_number"),
            "procedure_number": "",
            "regulatory_outcome": "",
            "first_published_date": first_nonempty(record, "first_published_date"),
            "last_updated_date": first_nonempty(record, "last_updated_date"),
            "marketing_authorisation_date": first_nonempty(record, "marketing_authorisation_date"),
            "url": first_nonempty(record, "medicine_url"),
            "extra_context": "",
        }

    if source_key == "psusas":
        entity_name = (
            first_nonempty(summary, "entity_name")
            or first_nonempty(record, "active_substances_in_scope_of_procedure", "active_substance")
        )
        active_substance = (
            first_nonempty(summary, "active_substance")
            or first_nonempty(record, "active_substance")
        )
        return {
            "entity_name": entity_name,
            "active_substance": active_substance,
            "therapeutic_indication": "",
            "therapeutic_area": "",
            "classification": first_nonempty(summary, "classification", "category"),
            "status": (
                first_nonempty(summary, "status")
                or first_nonempty(record, "regulatory_outcome")
            ),
            "opinion_status": "",
            "holder": "",
            "ema_number": "",
            "procedure_number": first_nonempty(record, "procedure_number"),
            "regulatory_outcome": first_nonempty(record, "regulatory_outcome"),
            "first_published_date": first_nonempty(record, "first_published_date"),
            "last_updated_date": first_nonempty(record, "last_updated_date"),
            "marketing_authorisation_date": "",
            "url": first_nonempty(record, "psusa_url"),
            "extra_context": first_nonempty(record, "related_medicines"),
        }

    if source_key == "referrals":
        entity_name = (
            first_nonempty(summary, "entity_name")
            or first_nonempty(record, "referral_name")
        )
        active_substance = (
            first_nonempty(summary, "active_substance")
            or first_nonempty(record, "international_non_proprietary_name_inn_common_name")
        )
        return {
            "entity_name": entity_name,
            "active_substance": active_substance,
            "therapeutic_indication": "",
            "therapeutic_area": "",
            "classification": (
                first_nonempty(summary, "classification")
                or first_nonempty(record, "referral_type")
            ),
            "status": (
                first_nonempty(summary, "status")
                or first_nonempty(record, "current_status")
            ),
            "opinion_status": "",
            "holder": "",
            "ema_number": "",
            "procedure_number": first_nonempty(record, "reference_number"),
            "regulatory_outcome": "",
            "first_published_date": first_nonempty(record, "first_published_date"),
            "last_updated_date": first_nonempty(record, "last_updated_date"),
            "marketing_authorisation_date": "",
            "url": first_nonempty(record, "referral_url"),
            "extra_context": first_nonempty(
                record,
                "associated_names_centrally_authorised_medicines",
                "associated_names_non_centrally_authorised_medicines",
            ),
        }

    if source_key == "shortages":
        entity_name = (
            first_nonempty(summary, "entity_name")
            or first_nonempty(record, "medicine_affected")
        )
        active_substance = (
            first_nonempty(summary, "active_substance")
            or first_nonempty(record, "international_non_proprietary_name_inn_or_common_name")
        )
        return {
            "entity_name": entity_name,
            "active_substance": active_substance,
            "therapeutic_indication": "",
            "therapeutic_area": first_nonempty(record, "therapeutic_area_mesh"),
            "classification": first_nonempty(summary, "classification", "category"),
            "status": first_nonempty(record, "supply_shortage_status"),
            "opinion_status": "",
            "holder": "",
            "ema_number": "",
            "procedure_number": "",
            "regulatory_outcome": "",
            "first_published_date": first_nonempty(record, "first_published_date"),
            "last_updated_date": first_nonempty(record, "last_updated_date"),
            "marketing_authorisation_date": "",
            "url": first_nonempty(record, "shortage_url"),
            "extra_context": normalize_whitespace(
                " ".join(
                    x for x in [
                        first_nonempty(record, "pharmaceutical_forms_affected"),
                        first_nonempty(record, "strengths_affected"),
                        f"Alternatives available: {first_nonempty(record, 'availability_of_alternatives')}" if first_nonempty(record, "availability_of_alternatives") else "",
                    ] if x
                )
            ),
        }

    raise ValueError(f"Unsupported source_key: {source_key}")


# -------------------------------------------------------------------
# Prompting
# -------------------------------------------------------------------

def build_prompt(summary: dict[str, str], document_type: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a biomedical factoid generator working from EMA records. "
        "Your job is to create short, self-sufficient factual statements about breast-cancer-related EMA records. "
        "Each factoid must stand on its own without requiring the source record for context. "
        "Always use the medicine name, procedure name, or active substance explicitly in each factoid as appropriate. "
        "Do not invent facts. Only use information directly present in the provided fields. "
        "Focus on breast-cancer-relevant information. "
        "For medicines, prioritise indication, biomarker, disease setting, regulatory status, and treatment context. "
        "For DHPCs, shortages, referrals, PSUSAs, and post-authorisation records, prioritise the regulatory event, active substance or medicine involved, and its breast-cancer relevance. "
        "Return between 2 and 8 factoids. "
        f"Wrap each factoid exactly like this: {FACTOID_START}fact text{FACTOID_END} "
        "Return nothing except the factoids."
    )

    user_prompt = (
        f"Document type: {document_type}\n"
        f"Entity name: {summary['entity_name'] or 'N/A'}\n"
        f"Active substance: {summary['active_substance'] or 'N/A'}\n"
        f"Therapeutic indication: {summary['therapeutic_indication'] or 'N/A'}\n"
        f"Therapeutic area: {summary['therapeutic_area'] or 'N/A'}\n"
        f"Classification: {summary['classification'] or 'N/A'}\n"
        f"Status: {summary['status'] or 'N/A'}\n"
        f"Opinion status: {summary['opinion_status'] or 'N/A'}\n"
        f"Holder: {summary['holder'] or 'N/A'}\n"
        f"EMA number: {summary['ema_number'] or 'N/A'}\n"
        f"Procedure number: {summary['procedure_number'] or 'N/A'}\n"
        f"Regulatory outcome: {summary['regulatory_outcome'] or 'N/A'}\n"
        f"Marketing authorisation date: {summary['marketing_authorisation_date'] or 'N/A'}\n"
        f"First published date: {summary['first_published_date'] or 'N/A'}\n"
        f"Last updated date: {summary['last_updated_date'] or 'N/A'}\n"
        f"URL: {summary['url'] or 'N/A'}\n"
        f"Extra context: {summary['extra_context'] or 'N/A'}\n\n"
        "Generate self-sufficient factoids only about this record's breast-cancer relevance."
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

        key = fact.lower()
        if key not in seen:
            seen.add(key)
            factoids.append(fact)

    return factoids


def build_per_record_file_payload(
    config: dict[str, str],
    summary: dict[str, str],
    factoids: list[str],
    out_name: str,
    document_year: int,
) -> dict[str, Any]:
    return {
        "metadata": {
            "source_family": config["source_family"],
            "document_title": config["document_title"],
            "document_type": config["document_type"],
            "document_year": document_year,
            "file_name": out_name,
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
    summary: dict[str, str],
    document_type: str,
    model_name: str = MODEL_NAME,
) -> list[str]:
    last_error = None

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                messages=build_prompt(summary, document_type),
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
                f"{summary.get('entity_name', 'UNKNOWN')}: {exc}"
            )
            await asyncio.sleep(2)

    raise RuntimeError(f"LLM request failed after 3 attempts: {last_error}")


# -------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------

async def process_item(
    index: int,
    item: dict[str, Any],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
    config: dict[str, str],
    document_year: int,
) -> None:
    summary = extract_summary(item, config["source_key"])
    entity_name = summary["entity_name"] or f"record_{index}"

    print("\n==============================")
    print(f"SOURCE: {config['source_key']}")
    print(f"INDEX: {index}")
    print(f"ENTITY: {entity_name}")
    print(f"ACTIVE SUBSTANCE: {summary['active_substance']}")
    print(f"STATUS: {summary['status']}")
    print(f"THERAPEUTIC INDICATION: {summary['therapeutic_indication'][:1000]}")
    print("==============================\n")

    try:
        async with semaphore:
            factoids = await generate_factoids_for_record(
                client=client,
                summary=summary,
                document_type=config["document_type"],
            )
    except Exception as exc:
        print(f"[ERROR] Factoid generation failed for {entity_name}: {exc}")
        factoids = []

    result = {
        "record_index": index,
        "entity_name": summary["entity_name"],
        "active_substance": summary["active_substance"],
        "status": summary["status"],
        "procedure_number": summary["procedure_number"],
        "ema_number": summary["ema_number"],
        "url": summary["url"],
        "source_summary": summary,
        "factoids": [
            {"id": i + 1, "factoid_text": fact}
            for i, fact in enumerate(factoids)
        ],
    }

    async with results_lock:
        results.append(result)

    if WRITE_PER_RECORD_FILES and factoids:
        safe_name = slugify(summary["entity_name"] or summary["active_substance"] or f"record_{index}")
        out_name = f"{safe_name}.json"
        payload = build_per_record_file_payload(config, summary, factoids, out_name, document_year)

        per_record_dir = OUTPUT_DIR / config["source_key"] / "per_record"
        per_record_dir.mkdir(parents=True, exist_ok=True)
        (per_record_dir / out_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[DONE] {config['source_key']} | {entity_name} | factoids={len(factoids)}")


async def process_dataset(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: dict[str, str],
) -> tuple[int, int]:
    input_file = config["input_file"]
    output_file = config["output_file"]

    print("\n##################################################")
    print(f"Processing dataset: {config['source_key']}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("##################################################\n")

    if not input_file.exists():
        print(f"[WARN] Skipping missing file: {input_file}")
        return 0, 0

    try:
        raw = load_json(input_file)
        items = as_items(raw)
    except Exception as exc:
        print(f"[ERROR] Could not load {input_file}: {exc}")
        return 0, 0

    if MAX_ITEMS_PER_FILE is not None:
        items = items[:MAX_ITEMS_PER_FILE]

    if not items:
        print(f"[WARN] No items found in {input_file}")
        return 0, 0

    document_year = infer_document_year(items)
    results_lock = asyncio.Lock()
    results: list[dict[str, Any]] = []

    tasks = [
        process_item(
            index=i + 1,
            item=item,
            client=client,
            semaphore=semaphore,
            results=results,
            results_lock=results_lock,
            config=config,
            document_year=document_year,
        )
        for i, item in enumerate(items)
    ]

    await asyncio.gather(*tasks)

    results.sort(key=lambda x: (x.get("entity_name") or "").lower())

    flat_factoids: list[dict[str, Any]] = []
    global_id = 1

    for record in results:
        for fact in record["factoids"]:
            flat_factoids.append(
                {
                    "id": global_id,
                    "factoid_text": fact["factoid_text"],
                    "entity_name": record.get("entity_name", ""),
                    "active_substance": record.get("active_substance", ""),
                    "status": record.get("status", ""),
                    "procedure_number": record.get("procedure_number", ""),
                    "ema_number": record.get("ema_number", ""),
                    "url": record.get("url", ""),
                }
            )
            global_id += 1

    final_payload = {
        "metadata": {
            "source_family": config["source_family"],
            "document_title": config["document_title"],
            "document_type": config["document_type"],
            "document_year": document_year,
            "file_name": output_file.name,
        },
        "factoids": flat_factoids,
        "record_level_outputs": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DATASET DONE] {config['source_key']}")
    print(f"Records processed: {len(results)}")
    print(f"Factoids created: {len(flat_factoids)}")
    print(f"Wrote: {output_file}\n")

    return len(results), len(flat_factoids)


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
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")

    try:
        client = get_client()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    total_records = 0
    total_factoids = 0

    for config in DATASET_CONFIGS:
        records_count, factoids_count = await process_dataset(
            client=client,
            semaphore=semaphore,
            config=config,
        )
        total_records += records_count
        total_factoids += factoids_count

    total_time = time.time() - start_time

    print("\n======================================")
    print("ALL DATASETS DONE")
    print(f"Total records processed: {total_records}")
    print(f"Total factoids created: {total_factoids}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print("======================================")

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())