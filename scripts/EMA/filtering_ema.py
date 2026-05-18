#!/usr/bin/env python3
"""
Classify EMA JSON datasets as breast-cancer-related or not using a local GPT-OSS model.

Behavior:
- Reads multiple EMA JSON files from artifacts/EMA/downloaded
- Extracts key fields for each record depending on source file
- Sends the record information to a local OpenAI-compatible LLM
- Expects YES or NO
- Saves breast-cancer-related records to separate output files for each source JSON
- Saves non-breast-cancer-related records to separate output files for each source JSON
- Saves unclear/error cases to separate output files for each source JSON
- Saves console output to logs/filtering_ema_<datetime>.log

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
from typing import Any, Optional

from openai import AsyncOpenAI


INPUT_DIR = Path("artifacts/EMA/downloaded")
OUTPUT_DIR = Path("artifacts/EMA/filtered")
LOG_DIR = Path("logs")

MODEL_NAME = "GPT-OSS-120B"
CONCURRENT_REQUESTS = 20
MAX_COMPLETION_TOKENS = 200
MAX_ITEMS_PER_FILE = None  # set to int like 100 for testing

DATASET_FILES = [
    "medicines.json",
    "dhpcs.json",
    "post_authorisation.json",
    "psusas.json",
    "referrals.json",
    "shortages.json",  # optional; skipped if not present
]


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
    return LOG_DIR / f"filtering_ema_{timestamp}.log"


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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def as_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("data", "items", "results", "rows", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        for value in data.values():
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value

    raise ValueError("Could not find a list of records in JSON file")


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, (list, tuple, set)):
        parts = [norm_text(v) for v in value]
        return normalize_whitespace(" ".join(p for p in parts if p))
    if isinstance(value, dict):
        parts = [norm_text(v) for v in value.values()]
        return normalize_whitespace(" ".join(p for p in parts if p))
    return normalize_whitespace(str(value))


def first_nonempty(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = norm_text(record.get(key))
        if value:
            return value
    return ""


def dataset_key_from_filename(filename: str) -> str:
    return Path(filename).stem.lower()


def extract_record_summary(record: dict[str, Any], dataset_key: str) -> dict[str, str]:
    """
    Returns a normalized summary across EMA source files.
    """
    summary: dict[str, str] = {
        "dataset_key": dataset_key,
        "entity_name": "",
        "active_substance": "",
        "therapeutic_indication": "",
        "therapeutic_area": "",
        "status": "",
        "holder": "",
        "ema_number": "",
        "procedure_number": "",
        "regulatory_outcome": "",
        "classification": "",
        "url": "",
        "extra_context": "",
    }

    if dataset_key == "medicines":
        summary["entity_name"] = first_nonempty(record, "name", "name_of_medicine")
        summary["active_substance"] = first_nonempty(
            record,
            "active_substance",
            "international_non_proprietary_name_inn_or_common_name",
            "international_non_proprietary_name_common_name",
        )
        summary["therapeutic_indication"] = first_nonempty(record, "therapeutic_indication")
        summary["therapeutic_area"] = first_nonempty(record, "therapeutic_area_mesh")
        summary["status"] = first_nonempty(
            record,
            "authorisation_status",
            "status",
            "medicine_status",
            "opinion_status",
        )
        summary["holder"] = first_nonempty(
            record,
            "marketing_authorisation_holder",
            "marketing_authorisation_developer_applicant_holder",
        )
        summary["ema_number"] = first_nonempty(
            record,
            "ema_number",
            "ema_medicine_number",
            "ema_product_number",
        )
        summary["classification"] = first_nonempty(
            record,
            "pharmacotherapeutic_group_human",
            "atc_code_human",
        )
        summary["url"] = first_nonempty(record, "medicine_url")

    elif dataset_key == "dhpcs":
        summary["entity_name"] = first_nonempty(record, "name_of_medicine")
        summary["active_substance"] = first_nonempty(record, "active_substances")
        summary["therapeutic_area"] = first_nonempty(record, "therapeutic_area_mesh")
        summary["status"] = first_nonempty(record, "dhpc_type")
        summary["procedure_number"] = first_nonempty(record, "procedure_number")
        summary["regulatory_outcome"] = first_nonempty(record, "regulatory_outcome")
        summary["classification"] = first_nonempty(record, "category", "atc_code_human")
        summary["url"] = first_nonempty(record, "dhpc_url")
        summary["extra_context"] = first_nonempty(
            record,
            "referral_name",
            "other_related_medicines_nationally_authorised",
        )

    elif dataset_key == "post_authorisation":
        summary["entity_name"] = first_nonempty(record, "name_of_medicine")
        summary["active_substance"] = first_nonempty(
            record,
            "active_substance",
            "international_non_proprietary_name_common_name",
        )
        summary["therapeutic_area"] = first_nonempty(record, "therapeutic_area_mesh")
        summary["status"] = first_nonempty(
            record,
            "post_authorisation_procedure_status",
            "post_authorisation_opinion_status",
        )
        summary["holder"] = first_nonempty(record, "marketing_authorisation_developer_applicant_holder")
        summary["ema_number"] = first_nonempty(record, "ema_product_number")
        summary["classification"] = first_nonempty(record, "atc_code_human", "category")
        summary["url"] = first_nonempty(record, "medicine_url")
        summary["extra_context"] = normalize_whitespace(
            " ".join(
                x for x in [
                    first_nonempty(record, "orphan_medicine"),
                    first_nonempty(record, "conditional_approval"),
                    first_nonempty(record, "advanced_therapy"),
                    first_nonempty(record, "biosimilar"),
                ] if x
            )
        )

    elif dataset_key == "psusas":
        summary["entity_name"] = first_nonempty(
            record,
            "related_medicines",
            "active_substances_in_scope_of_procedure",
            "active_substance",
        )
        summary["active_substance"] = first_nonempty(
            record,
            "active_substance",
            "active_substances_in_scope_of_procedure",
        )
        summary["status"] = first_nonempty(record, "regulatory_outcome")
        summary["procedure_number"] = first_nonempty(record, "procedure_number")
        summary["classification"] = first_nonempty(record, "category")
        summary["url"] = first_nonempty(record, "psusa_url")

    elif dataset_key == "referrals":
        summary["entity_name"] = first_nonempty(
            record,
            "referral_name",
            "associated_names_centrally_authorised_medicines",
            "associated_names_non_centrally_authorised_medicines",
        )
        summary["active_substance"] = first_nonempty(
            record,
            "international_non_proprietary_name_inn_common_name",
        )
        summary["status"] = first_nonempty(record, "current_status")
        summary["procedure_number"] = first_nonempty(record, "reference_number")
        summary["regulatory_outcome"] = first_nonempty(record, "prac_recommendation")
        summary["classification"] = first_nonempty(
            record,
            "referral_type",
            "class",
            "authorisation_model",
        )
        summary["url"] = first_nonempty(record, "referral_url")
        summary["extra_context"] = normalize_whitespace(
            " ".join(
                x for x in [
                    first_nonempty(record, "safety_referral"),
                    first_nonempty(record, "associated_names_centrally_authorised_medicines"),
                    first_nonempty(record, "associated_names_non_centrally_authorised_medicines"),
                ] if x
            )
        )

    elif dataset_key == "shortages":
        summary["entity_name"] = first_nonempty(record, "name_of_medicine", "medicine_name", "name")
        summary["active_substance"] = first_nonempty(
            record,
            "active_substance",
            "active_substances",
            "international_non_proprietary_name_common_name",
        )
        summary["therapeutic_area"] = first_nonempty(record, "therapeutic_area_mesh")
        summary["status"] = first_nonempty(record, "current_status", "shortage_status", "status")
        summary["procedure_number"] = first_nonempty(record, "procedure_number")
        summary["classification"] = first_nonempty(record, "category", "atc_code_human")
        summary["url"] = first_nonempty(record, "shortage_url", "medicine_url")
        summary["extra_context"] = normalize_whitespace(
            " ".join(
                x for x in [
                    first_nonempty(record, "reason"),
                    first_nonempty(record, "other_related_medicines_nationally_authorised"),
                ] if x
            )
        )

    else:
        # generic fallback
        summary["entity_name"] = first_nonempty(
            record,
            "name",
            "name_of_medicine",
            "referral_name",
        )
        summary["active_substance"] = first_nonempty(
            record,
            "active_substance",
            "active_substances",
            "international_non_proprietary_name_common_name",
            "international_non_proprietary_name_inn_common_name",
        )
        summary["therapeutic_indication"] = first_nonempty(record, "therapeutic_indication")
        summary["therapeutic_area"] = first_nonempty(record, "therapeutic_area_mesh")
        summary["status"] = first_nonempty(record, "status", "current_status")
        summary["holder"] = first_nonempty(record, "marketing_authorisation_holder")
        summary["ema_number"] = first_nonempty(record, "ema_number", "ema_product_number")
        summary["procedure_number"] = first_nonempty(record, "procedure_number", "reference_number")
        summary["regulatory_outcome"] = first_nonempty(record, "regulatory_outcome")
        summary["classification"] = first_nonempty(record, "category", "class")
        summary["url"] = first_nonempty(record, "url", "medicine_url", "referral_url")

    return summary


def build_prompt(summary: dict[str, str]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a biomedical regulatory classifier. "
        "Decide whether an EMA record is related to breast cancer in any way. "
        "Answer YES if the record clearly concerns a medicine, active substance, safety procedure, "
        "post-authorisation activity, referral, or shortage related to breast cancer, metastatic breast cancer, "
        "HER2-positive breast cancer, HER2-negative breast cancer, triple-negative breast cancer, "
        "HR-positive breast cancer, ER-positive breast cancer, breast carcinoma, mammary carcinoma, "
        "or a drug clearly used in treatment, diagnosis, monitoring, prevention, or management of breast cancer. "
        "Answer YES also for breast-cancer drugs appearing in safety/regulatory datasets even when the record is not a medicine master record. "
        "Answer NO if the record is unrelated or only generally anticancer without clear breast-cancer relevance. "
        "Reply with exactly one word: YES or NO."
    )

    user_prompt = (
        f"EMA source file type: {summary['dataset_key'] or 'N/A'}\n"
        f"Entity name: {summary['entity_name'] or 'N/A'}\n"
        f"Active substance: {summary['active_substance'] or 'N/A'}\n"
        f"Therapeutic indication: {summary['therapeutic_indication'] or 'N/A'}\n"
        f"Therapeutic area: {summary['therapeutic_area'] or 'N/A'}\n"
        f"Status: {summary['status'] or 'N/A'}\n"
        f"Marketing authorisation holder: {summary['holder'] or 'N/A'}\n"
        f"EMA/product number: {summary['ema_number'] or 'N/A'}\n"
        f"Procedure/reference number: {summary['procedure_number'] or 'N/A'}\n"
        f"Regulatory outcome: {summary['regulatory_outcome'] or 'N/A'}\n"
        f"Classification: {summary['classification'] or 'N/A'}\n"
        f"Extra context: {summary['extra_context'] or 'N/A'}\n"
        f"URL: {summary['url'] or 'N/A'}\n\n"
        "Is this EMA record related to breast cancer in any way? "
        "Answer exactly YES or NO."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def classify_record(
    client: AsyncOpenAI,
    summary: dict[str, str],
    model_name: str = MODEL_NAME,
) -> Optional[bool]:
    last_error = None

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                messages=build_prompt(summary),
                model=model_name,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            content = response.choices[0].message.content or ""
            answer = normalize_whitespace(content).upper()
            print(f"[DEBUG RAW MODEL OUTPUT] {repr(content)}")

            if answer == "YES":
                return True
            if answer == "NO":
                return False

            match = re.search(r"\b(YES|NO)\b", answer)
            if match:
                return match.group(1) == "YES"

            return None

        except Exception as exc:
            last_error = exc
            print(f"[WARN] attempt {attempt + 1}/3 failed: {exc}")
            await asyncio.sleep(2)

    raise RuntimeError(f"LLM request failed after 3 attempts: {last_error}")


async def process_record(
    index: int,
    record: dict[str, Any],
    dataset_key: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    yes_results: list[dict[str, Any]],
    no_results: list[dict[str, Any]],
    unclear_results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
) -> None:
    summary = extract_record_summary(record, dataset_key)
    display_name = summary["entity_name"] or f"{dataset_key}_record_{index}"

    print("\n==============================")
    print(f"DATASET: {dataset_key}")
    print(f"INDEX: {index}")
    print(f"ENTITY: {display_name}")
    print(f"ACTIVE SUBSTANCE: {summary['active_substance']}")
    print(f"THERAPEUTIC INDICATION: {summary['therapeutic_indication']}")
    print(f"THERAPEUTIC AREA: {summary['therapeutic_area']}")
    print(f"PROCEDURE/EMA NUMBER: {summary['procedure_number'] or summary['ema_number']}")
    print("==============================\n")

    try:
        async with semaphore:
            is_relevant = await classify_record(client, summary)
    except Exception as exc:
        print(f"[ERROR] LLM request failed for {display_name}: {exc}")
        async with results_lock:
            unclear_results.append(
                {
                    "llm_label": "ERROR",
                    "llm_reason": str(exc),
                    "summary": summary,
                    "record": record,
                }
            )
        return

    item = {
        "llm_label": (
            "YES" if is_relevant is True else
            "NO" if is_relevant is False else
            "UNCLEAR"
        ),
        "summary": summary,
        "record": record,
    }

    async with results_lock:
        if is_relevant is True:
            yes_results.append(item)
            print(f"[YES] {display_name}")
        elif is_relevant is False:
            no_results.append(item)
            print(f"[NO] {display_name}")
        else:
            unclear_results.append(item)
            print(f"[UNCLEAR] Model did not return clean YES/NO for {display_name}")


def build_output_paths(dataset_key: str) -> tuple[Path, Path, Path]:
    yes_file = OUTPUT_DIR / f"{dataset_key}_breast_cancer_related.json"
    no_file = OUTPUT_DIR / f"{dataset_key}_not_breast_cancer_related.json"
    unclear_file = OUTPUT_DIR / f"{dataset_key}_unclear_breast_cancer_related.json"
    return yes_file, no_file, unclear_file


async def process_dataset(
    input_file: Path,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> None:
    dataset_key = dataset_key_from_filename(input_file.name)
    yes_output_file, no_output_file, unclear_output_file = build_output_paths(dataset_key)

    print("\n==================================================")
    print(f"START DATASET: {dataset_key}")
    print(f"INPUT FILE: {input_file}")
    print("==================================================\n")

    try:
        raw = json.loads(input_file.read_text(encoding="utf-8"))
        records = as_list(raw)
    except Exception as exc:
        print(f"[ERROR] Could not load {input_file}: {exc}")
        return

    if MAX_ITEMS_PER_FILE is not None:
        records = records[:MAX_ITEMS_PER_FILE]

    if not records:
        print(f"[ERROR] No records found in {input_file}")
        return

    results_lock = asyncio.Lock()
    yes_results: list[dict[str, Any]] = []
    no_results: list[dict[str, Any]] = []
    unclear_results: list[dict[str, Any]] = []

    print(f"Processing {len(records)} records from {input_file} ...")

    tasks = [
        process_record(
            index=i + 1,
            record=record,
            dataset_key=dataset_key,
            client=client,
            semaphore=semaphore,
            yes_results=yes_results,
            no_results=no_results,
            unclear_results=unclear_results,
            results_lock=results_lock,
        )
        for i, record in enumerate(records)
    ]

    await asyncio.gather(*tasks)

    yes_payload = {
        "metadata": {
            "source_file": str(input_file),
            "source_type": dataset_key,
            "model_name": MODEL_NAME,
            "count": len(yes_results),
        },
        "items": yes_results,
    }

    no_payload = {
        "metadata": {
            "source_file": str(input_file),
            "source_type": dataset_key,
            "model_name": MODEL_NAME,
            "count": len(no_results),
        },
        "items": no_results,
    }

    unclear_payload = {
        "metadata": {
            "source_file": str(input_file),
            "source_type": dataset_key,
            "model_name": MODEL_NAME,
            "count": len(unclear_results),
        },
        "items": unclear_results,
    }

    yes_output_file.write_text(
        json.dumps(yes_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    no_output_file.write_text(
        json.dumps(no_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unclear_output_file.write_text(
        json.dumps(unclear_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDONE DATASET: {dataset_key}")
    print(f"YES count: {len(yes_results)}")
    print(f"NO count: {len(no_results)}")
    print(f"UNCLEAR count: {len(unclear_results)}")
    print(f"Wrote: {yes_output_file}")
    print(f"Wrote: {no_output_file}")
    print(f"Wrote: {unclear_output_file}\n")


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

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return 1

    try:
        client = get_client()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    found_any = False
    for filename in DATASET_FILES:
        input_file = INPUT_DIR / filename
        if not input_file.exists():
            print(f"[SKIP] File not found: {input_file}")
            continue

        found_any = True
        await process_dataset(input_file, client, semaphore)

    if not found_any:
        print(f"[ERROR] None of the expected EMA JSON files were found in {INPUT_DIR}")
        return 1

    total_time = time.time() - start_time
    print("\nAll datasets done.")
    print(f"Total runtime: {total_time:.2f} seconds")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())