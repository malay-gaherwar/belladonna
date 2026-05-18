#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LICENSE_SUMMARY_JSON = Path("artifacts/Elsevier/license_summary.json")
FACTOIDS_DIR = Path("artifacts/Elsevier/factoids")

# If True, rewrite even if metadata["license_label"] already exists.
# If False, reruns will skip files already done.
FORCE_REWRITE = False

# If True, files with DOI not found in license_summary get metadata["license_label"] = "NO_OPEN_ACCESS"
# If False, unmatched files are left unchanged.
FILL_NO_OPEN_ACCESS_WHEN_MISSING = False


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_doi(value: Any) -> str:
    if value is None:
        return ""

    s = str(value).strip().lower()

    if not s:
        return ""

    # Normalize common DOI URL forms.
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    ]

    for prefix in prefixes:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break

    return s


def build_license_map(license_summary: dict[str, Any]) -> dict[str, str]:
    per_file = license_summary.get("per_file", [])
    license_map: dict[str, str] = {}

    for row in per_file:
        if not isinstance(row, dict):
            continue

        doi = normalize_doi(row.get("doi", ""))
        if not doi:
            continue

        license_label = str(row.get("license_label", "")).strip()
        if not license_label:
            continue

        license_map[doi] = license_label

    return license_map


def update_factoid_file(
    factoid_path: Path,
    license_map: dict[str, str],
) -> tuple[str, str]:
    """
    Returns:
      ("updated", doi)
      ("skipped_already_done", doi)
      ("skipped_no_doi", "")
      ("skipped_no_match", doi)
      ("unchanged", doi)
      ("error", reason)
    """
    try:
        data = load_json(factoid_path)
    except Exception as e:
        return ("error", f"read_error: {e}")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return ("error", "missing_or_invalid_metadata")

    doi = normalize_doi(metadata.get("doi"))
    if not doi:
        return ("skipped_no_doi", "")

    # Resume behavior: skip files already processed.
    if not FORCE_REWRITE and metadata.get("license_label"):
        return ("skipped_already_done", doi)

    match = license_map.get(doi)
    if not match:
        if not FILL_NO_OPEN_ACCESS_WHEN_MISSING:
            return ("skipped_no_match", doi)

        new_label = "NO_OPEN_ACCESS"
    else:
        new_label = match

    old_label = metadata.get("license_label")

    if old_label == new_label:
        return ("unchanged", doi)

    metadata["license_label"] = new_label

    try:
        save_json(factoid_path, data)
    except Exception as e:
        return ("error", f"write_error: {e}")

    return ("updated", doi)


def main() -> None:
    if not LICENSE_SUMMARY_JSON.exists():
        raise FileNotFoundError(f"License summary not found: {LICENSE_SUMMARY_JSON}")

    if not FACTOIDS_DIR.exists():
        raise FileNotFoundError(f"Factoids directory not found: {FACTOIDS_DIR}")

    license_summary = load_json(LICENSE_SUMMARY_JSON)
    license_map = build_license_map(license_summary)

    factoid_files = sorted(FACTOIDS_DIR.glob("*.json"))
    if not factoid_files:
        raise FileNotFoundError(f"No JSON files found in: {FACTOIDS_DIR}")

    updated = 0
    unchanged = 0
    skipped_already_done = 0
    skipped_no_doi = 0
    skipped_no_match = 0
    errors = 0

    for i, factoid_file in enumerate(factoid_files, start=1):
        status, info = update_factoid_file(factoid_file, license_map)

        if status == "updated":
            updated += 1
        elif status == "unchanged":
            unchanged += 1
        elif status == "skipped_already_done":
            skipped_already_done += 1
        elif status == "skipped_no_doi":
            skipped_no_doi += 1
        elif status == "skipped_no_match":
            skipped_no_match += 1
        elif status == "error":
            errors += 1
            print(f"[ERROR] {factoid_file.name}: {info}")

        if i % 1000 == 0 or i == len(factoid_files):
            print(
                f"Processed {i}/{len(factoid_files)} | "
                f"updated={updated} already_done={skipped_already_done} "
                f"unchanged={unchanged} no_doi={skipped_no_doi} "
                f"no_match={skipped_no_match} errors={errors}"
            )

    print("\nDone.")
    print(f"License records loaded: {len(license_map)}")
    print(f"Factoid files scanned: {len(factoid_files)}")
    print(f"Updated: {updated}")
    print(f"Skipped (already done): {skipped_already_done}")
    print(f"Unchanged: {unchanged}")
    print(f"Skipped (no DOI): {skipped_no_doi}")
    print(f"Skipped (no match in license summary): {skipped_no_match}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()