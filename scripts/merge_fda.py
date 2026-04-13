#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LABEL_DIR = Path("artifacts/fda/label/pages")
DRUGSFDA_DIR = Path("artifacts/fda/drugsfda/pages")
OUTPUT_DIR = Path("artifacts/fda/merged")

ALLOW_NAME_FALLBACK = False
KEEP_RAW_LABEL = False


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_text(value: str) -> str:
    value = value or ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    value = normalize_text(value).replace(" ", "_")
    return value[:120] or "record"


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_nonempty(*values: Any) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return value
        if isinstance(value, dict) and value:
            return value
    return None


def first_str(values: Any) -> Optional[str]:
    for v in as_list(values):
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_results_from_json(path: Path) -> Iterable[Dict[str, Any]]:
    obj = read_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        for item in obj["results"]:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(obj, dict):
        yield obj
        return
    raise ValueError(f"Unsupported JSON structure in {path}")


class DrugsFdaIndex:
    def __init__(self) -> None:
        self.by_spl_set_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_spl_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.total_records = 0

    def add(self, record: Dict[str, Any], source_file: str) -> None:
        self.total_records += 1
        openfda = record.get("openfda", {}) or {}
        products = as_list(record.get("products"))

        meta = {
            "source_file": source_file,
            "application_number": record.get("application_number"),
            "sponsor_name": record.get("sponsor_name"),
            "submissions": as_list(record.get("submissions")),
            "products": products,
            "openfda": openfda,
        }

        for value in as_list(openfda.get("spl_set_id")):
            if isinstance(value, str) and value.strip():
                self.by_spl_set_id[value.strip()].append(meta)

        for value in as_list(openfda.get("spl_id")):
            if isinstance(value, str) and value.strip():
                self.by_spl_id[value.strip()].append(meta)

        names: List[str] = []
        names.extend(x for x in as_list(openfda.get("generic_name")) if isinstance(x, str))
        names.extend(x for x in as_list(openfda.get("brand_name")) if isinstance(x, str))
        names.extend(x for x in as_list(openfda.get("substance_name")) if isinstance(x, str))

        for product in products:
            if not isinstance(product, dict):
                continue
            brand_name = product.get("brand_name")
            if isinstance(brand_name, str):
                names.append(brand_name)
            for ingredient in as_list(product.get("active_ingredients")):
                if isinstance(ingredient, dict) and isinstance(ingredient.get("name"), str):
                    names.append(ingredient["name"])

        for name in names:
            key = normalize_text(name)
            if key:
                self.by_name[key].append(meta)


def load_drugsfda_index(drugsfda_dir: Path) -> DrugsFdaIndex:
    index = DrugsFdaIndex()
    files = sorted(drugsfda_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files found in {drugsfda_dir.resolve()}")

    log(f"Indexing Drugs@FDA files from {drugsfda_dir.resolve()}")
    for i, path in enumerate(files, start=1):
        for record in iter_results_from_json(path):
            index.add(record, path.name)
        if i % 5 == 0 or i == len(files):
            log(f"[Drugs@FDA] indexed {i}/{len(files)} files | records={index.total_records}")

    return index


def unique_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        key = (
            record.get("application_number"),
            record.get("sponsor_name"),
            tuple(as_list(record.get("openfda", {}).get("spl_set_id"))),
            tuple(as_list(record.get("openfda", {}).get("spl_id"))),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def latest_submission_date(meta: Dict[str, Any]) -> str:
    best = ""
    for submission in as_list(meta.get("submissions")):
        if not isinstance(submission, dict):
            continue
        value = submission.get("submission_status_date") or ""
        if isinstance(value, str) and value > best:
            best = value
    return best


def choose_best_metadata(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sort_key(meta: Dict[str, Any]) -> Tuple[int, int, str, str]:
        products = as_list(meta.get("products"))
        has_prescription = int(
            any(
                isinstance(p, dict) and p.get("marketing_status") == "Prescription"
                for p in products
            )
        )
        has_products = int(bool(products))
        return (
            has_prescription,
            has_products,
            latest_submission_date(meta),
            meta.get("application_number") or "",
        )

    return sorted(candidates, key=sort_key, reverse=True)[0]


def extract_label_candidate_names(label_record: Dict[str, Any]) -> List[str]:
    names: List[str] = []

    openfda = label_record.get("openfda", {}) or {}
    names.extend(x for x in as_list(openfda.get("generic_name")) if isinstance(x, str))
    names.extend(x for x in as_list(openfda.get("brand_name")) if isinstance(x, str))
    names.extend(x for x in as_list(openfda.get("substance_name")) if isinstance(x, str))

    for item in as_list(label_record.get("active_ingredient")):
        if isinstance(item, str):
            item = re.sub(r"^active ingredients?[:\s]+", "", item, flags=re.I)
            item = re.sub(r"\([^)]*\)", " ", item)
            item = re.sub(r"\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|%)\b", " ", item, flags=re.I)
            item = re.sub(r"\s+", " ", item).strip(" ,;:-")
            if item:
                names.append(item)

    return [x for x in names if normalize_text(x)]


def metadata_matches_label_by_set_id(label_record: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    label_openfda = label_record.get("openfda", {}) or {}

    label_set_ids = set()
    for v in as_list(label_openfda.get("spl_set_id")):
        if isinstance(v, str) and v.strip():
            label_set_ids.add(v.strip())

    # fallback only if openfda lacks it
    if not label_set_ids:
        v = label_record.get("set_id")
        if isinstance(v, str) and v.strip():
            label_set_ids.add(v.strip())

    meta_set_ids = set()
    for v in as_list(meta.get("openfda", {}).get("spl_set_id")):
        if isinstance(v, str) and v.strip():
            meta_set_ids.add(v.strip())

    return bool(label_set_ids and meta_set_ids and (label_set_ids & meta_set_ids))


def metadata_matches_label_by_spl_id(label_record: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    label_openfda = label_record.get("openfda", {}) or {}

    label_ids = set()
    for v in as_list(label_openfda.get("spl_id")):
        if isinstance(v, str) and v.strip():
            label_ids.add(v.strip())

    # fallback only if openfda lacks it
    if not label_ids:
        v = label_record.get("id")
        if isinstance(v, str) and v.strip():
            label_ids.add(v.strip())

    meta_ids = set()
    for v in as_list(meta.get("openfda", {}).get("spl_id")):
        if isinstance(v, str) and v.strip():
            meta_ids.add(v.strip())

    return bool(label_ids and meta_ids and (label_ids & meta_ids))


def validate_and_choose(
    label_record: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    method: str,
) -> Tuple[Optional[Dict[str, Any]], str, int]:
    unique = unique_records(candidates)
    if not unique:
        return None, "unmatched", 0

    if method == "matched_by_spl_set_id":
        valid = [m for m in unique if metadata_matches_label_by_set_id(label_record, m)]
    elif method == "matched_by_spl_id":
        valid = [m for m in unique if metadata_matches_label_by_spl_id(label_record, m)]
    else:
        valid = unique

    if not valid:
        return None, f"{method}_validation_failed", len(unique)

    return choose_best_metadata(valid), method, len(valid)


def match_label_to_metadata(
    label_record: Dict[str, Any],
    index: DrugsFdaIndex,
    allow_name_fallback: bool,
) -> Tuple[Optional[Dict[str, Any]], str, int]:
    label_openfda = label_record.get("openfda", {}) or {}

    # 1. strongest: label openfda spl_set_id
    for set_id in as_list(label_openfda.get("spl_set_id")):
        if isinstance(set_id, str) and set_id.strip() and set_id.strip() in index.by_spl_set_id:
            return validate_and_choose(
                label_record,
                index.by_spl_set_id[set_id.strip()],
                "matched_by_spl_set_id",
            )

    # 2. fallback: label top-level set_id
    set_id = label_record.get("set_id")
    if isinstance(set_id, str) and set_id.strip() and set_id.strip() in index.by_spl_set_id:
        return validate_and_choose(
            label_record,
            index.by_spl_set_id[set_id.strip()],
            "matched_by_spl_set_id",
        )

    # 3. next strongest: label openfda spl_id
    for label_id in as_list(label_openfda.get("spl_id")):
        if isinstance(label_id, str) and label_id.strip() and label_id.strip() in index.by_spl_id:
            return validate_and_choose(
                label_record,
                index.by_spl_id[label_id.strip()],
                "matched_by_spl_id",
            )

    # 4. fallback: label top-level id
    label_id = label_record.get("id")
    if isinstance(label_id, str) and label_id.strip() and label_id.strip() in index.by_spl_id:
        return validate_and_choose(
            label_record,
            index.by_spl_id[label_id.strip()],
            "matched_by_spl_id",
        )

    # 5. optional name fallback
    if allow_name_fallback:
        all_candidates: List[Dict[str, Any]] = []
        for name in extract_label_candidate_names(label_record):
            key = normalize_text(name)
            all_candidates.extend(index.by_name.get(key, []))
        unique = unique_records(all_candidates)
        if len(unique) == 1:
            return unique[0], "matched_by_name_fallback", 1
        if len(unique) > 1:
            return choose_best_metadata(unique), "matched_by_name_fallback_ambiguous", len(unique)

    return None, "unmatched", 0


def summarize_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        summary.append(
            {
                "product_number": product.get("product_number"),
                "brand_name": product.get("brand_name"),
                "dosage_form": product.get("dosage_form"),
                "route": product.get("route"),
                "marketing_status": product.get("marketing_status"),
                "reference_drug": product.get("reference_drug"),
                "reference_standard": product.get("reference_standard"),
                "te_code": product.get("te_code"),
                "active_ingredients": product.get("active_ingredients"),
            }
        )
    return summary


def extract_label_sections(label_record: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "effective_time",
        "indications_and_usage",
        "dosage_and_administration",
        "warnings",
        "warnings_and_cautions",
        "boxed_warning",
        "contraindications",
        "drug_interactions",
        "adverse_reactions",
        "pregnancy",
        "pregnancy_or_breast_feeding",
        "teratogenic_effects",
        "nursing_mothers",
        "breastfeeding",
        "pediatric_use",
        "geriatric_use",
        "renal_impairment",
        "hepatic_impairment",
        "use_in_specific_populations",
        "special_populations",
        "description",
        "clinical_pharmacology",
        "how_supplied",
        "package_label_principal_display_panel",
        "spl_product_data_elements",
        "active_ingredient",
        "inactive_ingredient",
        "purpose",
        "openfda",
        "set_id",
        "id",
        "version",
    ]
    return {key: label_record.get(key) for key in keys if key in label_record}


def build_metadata_block(label_record: Dict[str, Any], matched_meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not matched_meta:
        return None

    openfda = matched_meta.get("openfda", {}) or {}
    products = as_list(matched_meta.get("products"))
    first_product = products[0] if products and isinstance(products[0], dict) else {}

    return {
        "source_family": "FDA",
        "source_format": "JSON",
        "source_label_set_id": label_record.get("set_id"),
        "source_label_id": label_record.get("id"),
        "effective_time": label_record.get("effective_time"),
        "application_number": matched_meta.get("application_number"),
        "sponsor_name": matched_meta.get("sponsor_name"),
        "brand_name": first_nonempty(first_product.get("brand_name"), first_str(openfda.get("brand_name"))),
        "generic_name": first_nonempty(first_str(openfda.get("generic_name")), first_product.get("brand_name")),
        "manufacturer_name": first_str(openfda.get("manufacturer_name")),
        "product_type": first_str(openfda.get("product_type")),
        "route": first_nonempty(first_str(openfda.get("route")), first_product.get("route")),
        "substance_names": as_list(openfda.get("substance_name")),
        "spl_id": first_str(openfda.get("spl_id")),
        "spl_set_id": first_str(openfda.get("spl_set_id")),
        "products": summarize_products(products),
        "submissions": matched_meta.get("submissions"),
        "metadata_source_file": matched_meta.get("source_file"),
    }


def output_filename(label_record: Dict[str, Any], metadata_block: Optional[Dict[str, Any]]) -> str:
    generic = None
    if metadata_block:
        generic = metadata_block.get("generic_name") or metadata_block.get("brand_name")

    if not generic:
        label_openfda = label_record.get("openfda", {}) or {}
        generic = (
            first_str(label_openfda.get("generic_name"))
            or first_str(label_openfda.get("brand_name"))
            or "unknown"
        )

    set_id = label_record.get("set_id") or label_record.get("id") or datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{slugify(str(generic))}__{set_id}.json"


def merge_sources(args: argparse.Namespace) -> Dict[str, Any]:
    label_dir = Path(args.label_dir)
    drugsfda_dir = Path(args.drugsfda_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise RuntimeError(f"No JSON files found in {label_dir.resolve()}")

    log(f"Label dir: {label_dir.resolve()}")
    log(f"Drugs@FDA dir: {drugsfda_dir.resolve()}")
    log(f"Output dir: {output_dir.resolve()}")
    log(f"Found {len(label_files)} label JSON files")

    index = load_drugsfda_index(drugsfda_dir)

    stats = Counter()
    files_processed = 0
    start = time.time()

    for i, label_path in enumerate(label_files, start=1):
        files_processed += 1
        try:
            records = list(iter_results_from_json(label_path))
        except Exception as exc:
            stats["label_files_failed"] += 1
            log(f"[WARN] Failed to parse {label_path.name}: {exc}")
            continue

        file_matched = 0
        file_unmatched = 0

        for label_record in records:
            stats["label_records_total"] += 1
            matched_meta, match_method, candidate_count = match_label_to_metadata(
                label_record=label_record,
                index=index,
                allow_name_fallback=args.allow_name_fallback,
            )

            if matched_meta:
                stats["label_records_matched"] += 1
                file_matched += 1
            else:
                stats["label_records_unmatched"] += 1
                file_unmatched += 1

            stats[match_method] += 1

            metadata_block = build_metadata_block(label_record, matched_meta)
            merged_record = {
                "metadata_link_status": "matched" if matched_meta else "unmatched",
                "metadata_link_method": match_method,
                "metadata_match_candidates": candidate_count,
                "metadata": metadata_block,
                "label": extract_label_sections(label_record),
            }

            if args.keep_raw_label:
                merged_record["raw_label_record"] = label_record

            out_path = output_dir / output_filename(label_record, metadata_block)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(merged_record, f, ensure_ascii=False, indent=2)

        if i % 5 == 0 or i == len(label_files):
            elapsed = time.time() - start
            log(
                f"[Labels] {i}/{len(label_files)} files | "
                f"total_records={stats['label_records_total']} | "
                f"matched={stats['label_records_matched']} | "
                f"unmatched={stats['label_records_unmatched']} | "
                f"elapsed={elapsed:.1f}s"
            )

    summary = {
        "drugsfda_records_indexed": index.total_records,
        "label_files_processed": files_processed,
        "label_records_total": stats["label_records_total"],
        "label_records_matched": stats["label_records_matched"],
        "label_records_unmatched": stats["label_records_unmatched"],
        "match_breakdown": dict(stats),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step 1: merge FDA label-page records with Drugs@FDA metadata."
    )
    p.add_argument(
        "--label-dir",
        default=str(LABEL_DIR),
        help=f"Directory containing label JSON files (default: {LABEL_DIR})",
    )
    p.add_argument(
        "--drugsfda-dir",
        default=str(DRUGSFDA_DIR),
        help=f"Directory containing Drugs@FDA JSON files (default: {DRUGSFDA_DIR})",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Directory for merged output JSON files (default: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--allow-name-fallback",
        action="store_true",
        default=ALLOW_NAME_FALLBACK,
        help="Allow conservative name-based fallback matching when set_id/spl_id matches are missing",
    )
    p.add_argument(
        "--keep-raw-label",
        action="store_true",
        default=KEEP_RAW_LABEL,
        help="Include the complete raw label record in the merged output",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    summary = merge_sources(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()