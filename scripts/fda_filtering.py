#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


CSV_PATH = Path("bc_drugs_reference.csv")
MERGED_DIR = Path("artifacts/fda/merged")
OUTPUT_DIR = Path("artifacts/fda/seed_from_csv")

WRITE_JSONL = True
WRITE_SUMMARY = True


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
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]


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


def build_record(csv_row: dict[str, str], merged_record: dict[str, Any], merged_file_name: str) -> dict[str, Any]:
    metadata = merged_record.get("metadata") if isinstance(merged_record.get("metadata"), dict) else {}
    label = merged_record.get("label") if isinstance(merged_record.get("label"), dict) else {}
    openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

    brand_names = []
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


def build_index(merged_dir: Path) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], int]:
    index: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    total = 0

    for path in sorted(merged_dir.glob("*.json")):
        if path.name == "summary.json":
            continue

        try:
            record = read_json(path)
        except Exception as exc:
            print(f"[WARN] Could not parse {path.name}: {exc}")
            continue

        if not isinstance(record, dict):
            continue

        total += 1
        names = gather_name_candidates(record)
        for name in names:
            index[name].append((path.name, record))

        if total % 10000 == 0:
            print(f"[INDEX] processed {total} merged FDA records", flush=True)

    return index, total


def choose_best_match(candidates: list[tuple[str, dict[str, Any]]], csv_generic_name: str) -> tuple[str, dict[str, Any]]:
    target = normalize_text(csv_generic_name)

    def score(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, str]:
        file_name, record = item
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        openfda = label.get("openfda") if isinstance(label.get("openfda"), dict) else {}

        names = gather_name_candidates(record)
        exact = int(target in names)

        has_matched_metadata = int(record.get("metadata_link_status") == "matched")
        has_application = int(bool(first_nonempty(metadata.get("application_number"), openfda.get("application_number"))))
        label_date = first_nonempty(metadata.get("effective_time"), label.get("effective_time"))

        return (exact, has_matched_metadata, has_application, label_date)

    return sorted(candidates, key=score, reverse=True)[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise RuntimeError(f"CSV not found: {CSV_PATH}")
    if not MERGED_DIR.exists():
        raise RuntimeError(f"Merged dir not found: {MERGED_DIR}")

    print(f"Reading CSV: {CSV_PATH}", flush=True)
    csv_rows = read_csv_rows(CSV_PATH)
    print(f"CSV rows: {len(csv_rows)}", flush=True)

    print(f"Building merged FDA index from: {MERGED_DIR}", flush=True)
    merged_index, merged_total = build_index(MERGED_DIR)
    print(f"Merged FDA files indexed: {merged_total}", flush=True)
    print(f"Unique normalized names indexed: {len(merged_index)}", flush=True)

    matched_rows = 0
    unmatched_rows = 0
    output_records: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for i, row in enumerate(csv_rows, start=1):
        csv_generic_name = row.get("generic_name", "")
        if not csv_generic_name:
            unmatched_rows += 1
            unmatched.append({"csv_row": row, "reason": "missing_generic_name"})
            continue

        key = normalize_text(csv_generic_name)
        candidates = merged_index.get(key, [])

        if not candidates:
            unmatched_rows += 1
            unmatched.append({"csv_row": row, "reason": "no_fda_match"})
            print(f"[UNMATCHED] {csv_generic_name}", flush=True)
            continue

        matched_file_name, matched_record = choose_best_match(candidates, csv_generic_name)
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
        print(f"[MATCHED] {i}/{len(csv_rows)} | {csv_generic_name} -> {matched_file_name}", flush=True)

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
            "output_dir": str(OUTPUT_DIR),
        }
        with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()