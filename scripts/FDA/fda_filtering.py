#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

CSV_PATH = Path("bc_drugs_reference.csv")
MERGED_DIR = Path("artifacts/fda/merged")
OUTPUT_DIR = Path("artifacts/fda/seed_from_csv")

WRITE_JSONL = True
WRITE_SUMMARY = True

# LLM fallback for unresolved matches
USE_LLM_FALLBACK = True
MODEL_NAME = os.getenv("MODEL_NAME", "GPT-OSS-120B")
MAX_COMPLETION_TOKENS = 300
LLM_CANDIDATE_LIMIT = 20


# -------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------

def normalize_text(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    value = normalize_text(value).replace(" ", "_")
    return value[:120] or "unknown"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for x in value:
                if isinstance(x, str) and x.strip():
                    return x.strip()
    return ""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def join_label_field(label: dict[str, Any], key: str) -> str:
    value = label.get(key)
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return "\n\n".join(parts)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


# -------------------------------------------------------------------
# Name handling
# -------------------------------------------------------------------

def gather_name_candidates(record: dict[str, Any]) -> set[str]:
    names: set[str] = set()

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

    for value in [
        metadata.get("generic_name"),
        metadata.get("brand_name"),
        metadata.get("manufacturer_name"),
    ]:
        if isinstance(value, str) and value.strip():
            names.add(normalize_text(value))

    for value in as_list(metadata.get("substance_names")):
        if isinstance(value, str) and value.strip():
            names.add(normalize_text(value))

    for key in ("generic_name", "brand_name", "substance_name"):
        for value in as_list(openfda.get(key)):
            if isinstance(value, str) and value.strip():
                names.add(normalize_text(value))

    for value in as_list(label.get("active_ingredient")):
        if isinstance(value, str) and value.strip():
            cleaned = re.sub(r"^active ingredients?[:\s]+", "", value, flags=re.I)
            cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
            cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|%)\b", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
            if cleaned:
                names.add(normalize_text(cleaned))

    return {x for x in names if x}


def csv_name_variants(name: str) -> list[str]:
    variants: set[str] = set()
    raw = (name or "").strip()
    if not raw:
        return []

    variants.add(normalize_text(raw))

    outer = re.sub(r"\(.*?\)", "", raw).strip()
    if outer:
        variants.add(normalize_text(outer))

    for inner in re.findall(r"\((.*?)\)", raw):
        inner = inner.strip()
        if inner:
            variants.add(normalize_text(inner))

    synonym_map = {
        "fluorouracil (5-fu)": ["fluorouracil", "5 fu", "5 fluorouracil"],
        "t-dm1 (ado-trastuzumab emtansine)": [
            "ado trastuzumab emtansine",
            "trastuzumab emtansine",
            "kadcyla",
        ],
        "t-dxd (trastuzumab deruxtecan)": [
            "trastuzumab deruxtecan",
            "fam trastuzumab deruxtecan nxki",
            "enhertu",
        ],
        "eribulin": ["eribulin", "eribulin mesylate", "halaven"],
        "pamidronate": ["pamidronate", "pamidronate disodium", "aredia"],
        "tamoxifen": ["tamoxifen", "tamoxifen citrate", "nolvadex", "soltamox"],
        "toremifene": ["toremifene", "toremifene citrate", "fareston"],
        "vinblastine": ["vinblastine", "vinblastine sulfate"],
        "epirubicin": ["epirubicin", "epirubicin hydrochloride", "ellence"],
    }

    key = normalize_text(raw)
    for k, vals in synonym_map.items():
        if key == normalize_text(k):
            for v in vals:
                variants.add(normalize_text(v))

    return sorted(v for v in variants if v)


# -------------------------------------------------------------------
# Record building
# -------------------------------------------------------------------

def build_record(csv_row: dict[str, str], merged_record: dict[str, Any], merged_file_name: str) -> dict[str, Any]:
    metadata = merged_record.get("metadata") if isinstance(merged_record.get("metadata"), dict) else {}
    label = merged_record.get("label") if isinstance(merged_record.get("label"), dict) else {}
    openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

    brand_names: list[str] = []
    seen = set()
    for source in [
        as_list(metadata.get("brand_name")),
        as_list(openfda.get("brand_name")),
    ]:
        for item in source:
            if isinstance(item, str) and item.strip():
                key = normalize_text(item)
                if key not in seen:
                    seen.add(key)
                    brand_names.append(item.strip())

    special_populations_parts = []
    for key in [
        "renal_impairment",
        "hepatic_impairment",
        "pediatric_use",
        "geriatric_use",
        "use_in_specific_populations",
        "special_populations",
    ]:
        text = join_label_field(label, key)
        if text:
            special_populations_parts.append(text)

    pregnancy_parts = []
    for key in [
        "pregnancy",
        "pregnancy_or_breast_feeding",
        "breastfeeding",
        "nursing_mothers",
        "teratogenic_effects",
    ]:
        text = join_label_field(label, key)
        if text:
            pregnancy_parts.append(text)

    label_date = first_nonempty(
        metadata.get("effective_time"),
        label.get("effective_time"),
    )

    generic_name = first_nonempty(
        csv_row.get("generic_name"),
        metadata.get("generic_name"),
        openfda.get("generic_name"),
        metadata.get("brand_name"),
        openfda.get("brand_name"),
    )

    return {
        "metadata": {
            "source_family": "FDA",
            "source_format": "JSON",
            "file_name": merged_file_name,
            "data_source": "FDA_label",
            "last_updated": str(date.today()),
            "reviewed": False,
            "source_label_set_id": first_nonempty(metadata.get("source_label_set_id"), label.get("set_id")),
            "source_label_id": first_nonempty(metadata.get("source_label_id"), label.get("id")),
            "spl_id": first_nonempty(metadata.get("spl_id"), openfda.get("spl_id")),
            "spl_set_id": first_nonempty(metadata.get("spl_set_id"), openfda.get("spl_set_id")),
            "application_number": first_nonempty(metadata.get("application_number"), openfda.get("application_number")),
            "manufacturer_name": first_nonempty(metadata.get("manufacturer_name"), openfda.get("manufacturer_name")),
            "metadata_link_status": merged_record.get("metadata_link_status"),
            "metadata_link_method": merged_record.get("metadata_link_method"),
            "metadata_match_candidates": merged_record.get("metadata_match_candidates"),
        },
        "belladonna_fields": {
            "generic_name": generic_name,
            "brand_names": brand_names,
            "drug_class": csv_row.get("drug_class", ""),
            "indications_and_usage": join_label_field(label, "indications_and_usage"),
            "dosage_and_administration": join_label_field(label, "dosage_and_administration"),
            "warnings": join_label_field(label, "warnings") or join_label_field(label, "warnings_and_cautions"),
            "black_box_warning": join_label_field(label, "boxed_warning"),
            "pregnancy_or_breastfeeding": "\n\n".join(pregnancy_parts).strip(),
            "adverse_reactions": join_label_field(label, "adverse_reactions"),
            "special_populations": "\n\n".join(special_populations_parts).strip(),
            "contraindications": join_label_field(label, "contraindications"),
            "drug_interactions": join_label_field(label, "drug_interactions"),
            "label_date": label_date,
            "fda_bc_indication": csv_row.get("fda_bc_indication", ""),
            "ema_bc_indication": csv_row.get("ema_bc_indication", ""),
            "fda_ema_status": csv_row.get("fda_ema_status", ""),
            "use_type": csv_row.get("use_type", ""),
            "bc_notes": csv_row.get("notes", ""),
        },
    }


# -------------------------------------------------------------------
# Indexing
# -------------------------------------------------------------------

def build_index(merged_dir: Path) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], dict[str, tuple[str, dict[str, Any]]], int]:
    name_index: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    file_index: dict[str, tuple[str, dict[str, Any]]] = {}
    total = 0

    for path in sorted(merged_dir.glob("*.json")):
        if path.name == "summary.json":
            continue

        try:
            record = read_json(path)
        except Exception as exc:
            print(f"[WARN] Could not parse {path.name}: {exc}", flush=True)
            continue

        if not isinstance(record, dict):
            continue

        total += 1
        file_index[path.name] = (path.name, record)

        names = gather_name_candidates(record)
        for name in names:
            name_index[name].append((path.name, record))

        if total % 10000 == 0:
            print(f"[INDEX] processed {total} merged FDA records", flush=True)

    return name_index, file_index, total


def choose_best_match(candidates: list[tuple[str, dict[str, Any]]], csv_generic_name: str) -> tuple[str, dict[str, Any]]:
    target_variants = set(csv_name_variants(csv_generic_name))

    def score(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, str]:
        file_name, record = item
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        label = record.get("label") if isinstance(label := record.get("label"), dict) else {}
        openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

        names = gather_name_candidates(record)
        exact = int(bool(target_variants & names))
        has_matched_metadata = int(record.get("metadata_link_status") == "matched")
        has_application = int(bool(first_nonempty(metadata.get("application_number"), openfda.get("application_number"))))
        label_date = first_nonempty(metadata.get("effective_time"), label.get("effective_time"))

        return (exact, has_matched_metadata, has_application, label_date)

    return sorted(candidates, key=score, reverse=True)[0]


# -------------------------------------------------------------------
# LLM fallback
# -------------------------------------------------------------------

def get_client() -> Optional[OpenAI]:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key or not base_url:
        return None

    return OpenAI(api_key=api_key, base_url=base_url)


def build_candidate_summary(file_name: str, record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

    return {
        "file_name": file_name,
        "generic_name": first_nonempty(metadata.get("generic_name"), openfda.get("generic_name")),
        "brand_name": first_nonempty(metadata.get("brand_name"), openfda.get("brand_name")),
        "substance_name": first_nonempty(metadata.get("substance_names"), openfda.get("substance_name")),
        "active_ingredient": first_nonempty(label.get("active_ingredient")),
        "application_number": first_nonempty(metadata.get("application_number"), openfda.get("application_number")),
        "manufacturer_name": first_nonempty(metadata.get("manufacturer_name"), openfda.get("manufacturer_name")),
        "indications_preview": join_label_field(label, "indications_and_usage")[:500],
    }


def llm_resolve_match(
    client: OpenAI,
    csv_row: dict[str, str],
    candidate_items: list[tuple[str, dict[str, Any]]],
) -> Optional[str]:
    if not candidate_items:
        return None

    candidates_payload = [
        build_candidate_summary(file_name, record)
        for file_name, record in candidate_items[:LLM_CANDIDATE_LIMIT]
    ]

    prompt = {
        "target_drug_from_csv": {
            "generic_name": csv_row.get("generic_name", ""),
            "drug_class": csv_row.get("drug_class", ""),
            "notes": csv_row.get("notes", ""),
        },
        "candidate_fda_records": candidates_payload,
        "task": (
            "Choose the single candidate that refers to the same drug as the CSV target, "
            "even if naming differs by shorthand, salt form, official FDA name, or brand name. "
            "If none match, return null."
        ),
        "output_schema": {
            "matched_file_name": "string or null",
            "reason": "short string"
        }
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="low",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You resolve whether drug names refer to the same underlying drug. "
                    "Use only the provided JSON. "
                    "Be careful with shorthand names, aliases, salt forms, and brand/generic equivalents. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, indent=2),
            },
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except Exception:
        return None

    matched_file_name = parsed.get("matched_file_name")
    if isinstance(matched_file_name, str) and matched_file_name.strip():
        return matched_file_name.strip()

    return None


def collect_llm_candidates(
    csv_generic_name: str,
    name_index: dict[str, list[tuple[str, dict[str, Any]]]],
) -> list[tuple[str, dict[str, Any]]]:
    variants = csv_name_variants(csv_generic_name)
    tokens = set()
    for v in variants:
        tokens.update(v.split())

    candidate_map: dict[str, tuple[str, dict[str, Any]]] = {}

    for key, items in name_index.items():
        if not key:
            continue

        # token-overlap candidate gathering for unresolved cases
        score = sum(1 for token in tokens if token in key)
        if score > 0:
            for file_name, record in items:
                candidate_map[file_name] = (file_name, record)

    ranked = []
    for file_name, record in candidate_map.values():
        names = gather_name_candidates(record)
        score = max(sum(1 for token in tokens if token in n) for n in names) if names else 0
        ranked.append((score, file_name, record))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [(file_name, record) for score, file_name, record in ranked[:LLM_CANDIDATE_LIMIT]]


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise RuntimeError(f"CSV not found: {CSV_PATH}")
    if not MERGED_DIR.exists():
        raise RuntimeError(f"Merged dir not found: {MERGED_DIR}")

    client = get_client() if USE_LLM_FALLBACK else None
    if USE_LLM_FALLBACK and client is None:
        print("[WARN] LLM fallback enabled but VIRTUAL_API_KEY/BASE_URL missing; fallback disabled.", flush=True)

    print(f"Reading CSV: {CSV_PATH}", flush=True)
    csv_rows = read_csv_rows(CSV_PATH)
    print(f"CSV rows: {len(csv_rows)}", flush=True)

    print(f"Building merged FDA index from: {MERGED_DIR}", flush=True)
    name_index, file_index, merged_total = build_index(MERGED_DIR)
    print(f"Merged FDA files indexed: {merged_total}", flush=True)
    print(f"Unique normalized names indexed: {len(name_index)}", flush=True)

    matched_rows = 0
    unmatched_rows = 0
    llm_matched_rows = 0
    output_records: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for i, row in enumerate(csv_rows, start=1):
        csv_generic_name = row.get("generic_name", "")
        if not csv_generic_name:
            unmatched_rows += 1
            unmatched.append({"csv_row": row, "reason": "missing_generic_name"})
            continue

        keys = csv_name_variants(csv_generic_name)
        candidates: list[tuple[str, dict[str, Any]]] = []
        seen_files = set()

        for key in keys:
            for item in name_index.get(key, []):
                file_name, record = item
                if file_name not in seen_files:
                    seen_files.add(file_name)
                    candidates.append(item)

        matched_file_name: Optional[str] = None
        matched_record: Optional[dict[str, Any]] = None
        matched_via = "direct"

        if candidates:
            matched_file_name, matched_record = choose_best_match(candidates, csv_generic_name)
        elif client is not None:
            llm_candidates = collect_llm_candidates(csv_generic_name, name_index)
            llm_choice = llm_resolve_match(client, row, llm_candidates)

            if llm_choice and llm_choice in file_index:
                matched_file_name, matched_record = file_index[llm_choice]
                matched_via = "llm_fallback"

        if matched_record is None or matched_file_name is None:
            unmatched_rows += 1
            unmatched.append(
                {
                    "csv_row": row,
                    "reason": "no_fda_match",
                    "lookup_keys": keys,
                }
            )
            print(f"[UNMATCHED] {csv_generic_name} | lookup_keys={keys}", flush=True)
            continue

        out_record = build_record(row, matched_record, matched_file_name)
        output_records.append(out_record)

        out_name = (
            f"{slugify(csv_generic_name)}__"
            f"{slugify(out_record['metadata'].get('source_label_set_id') or matched_file_name)}.json"
        )
        out_path = OUTPUT_DIR / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_record, f, ensure_ascii=False, indent=2)

        matched_rows += 1
        if matched_via == "llm_fallback":
            llm_matched_rows += 1

        print(
            f"[MATCHED] {i}/{len(csv_rows)} | {csv_generic_name} -> {matched_file_name} | via={matched_via}",
            flush=True,
        )

    if WRITE_JSONL:
        jsonl_path = OUTPUT_DIR / "belladonna_fda_seed.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    unmatched_path = OUTPUT_DIR / "unmatched_seed_rows.json"
    with unmatched_path.open("w", encoding="utf-8") as f:
        json.dump(unmatched, f, ensure_ascii=False, indent=2)

    if WRITE_SUMMARY:
        summary = {
            "csv_rows_total": len(csv_rows),
            "merged_fda_files_indexed": merged_total,
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
            "llm_matched_rows": llm_matched_rows,
            "output_dir": str(OUTPUT_DIR),
        }
        with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()