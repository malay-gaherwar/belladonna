#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


FACTOIDS_DIR = Path("artifacts/Elsevier/factoids")
REPORT_PATH = Path("artifacts/Elsevier/factoid_license_check.json")
PROGRESS_EVERY = 1000


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if not FACTOIDS_DIR.exists():
        raise FileNotFoundError(f"Factoids directory not found: {FACTOIDS_DIR}")

    factoid_files = sorted(FACTOIDS_DIR.glob("*.json"))
    if not factoid_files:
        raise FileNotFoundError(f"No JSON files found in: {FACTOIDS_DIR}")

    counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    problem_files: list[dict[str, str]] = []

    for i, factoid_file in enumerate(factoid_files, start=1):
        try:
            data = load_json(factoid_file)
        except Exception as e:
            counts["errors"] += 1
            problem_files.append(
                {
                    "file": factoid_file.name,
                    "status": "error",
                    "detail": f"read_error: {e}",
                }
            )
            continue

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            counts["missing_metadata"] += 1
            problem_files.append(
                {
                    "file": factoid_file.name,
                    "status": "missing_metadata",
                    "detail": "metadata is missing or not an object",
                }
            )
            continue

        if "license_label" not in metadata:
            counts["missing_license_label"] += 1
            problem_files.append(
                {
                    "file": factoid_file.name,
                    "status": "missing_license_label",
                    "detail": "metadata.license_label is absent",
                }
            )
            continue

        license_label = str(metadata.get("license_label") or "").strip()
        if not license_label:
            counts["empty_license_label"] += 1
            problem_files.append(
                {
                    "file": factoid_file.name,
                    "status": "empty_license_label",
                    "detail": "metadata.license_label is empty",
                }
            )
            continue

        counts["has_license_label"] += 1
        license_counts[license_label] += 1

        if i % PROGRESS_EVERY == 0 or i == len(factoid_files):
            print(
                f"Processed {i}/{len(factoid_files)} | "
                f"has={counts['has_license_label']} "
                f"missing={counts['missing_license_label']} "
                f"empty={counts['empty_license_label']} "
                f"bad_metadata={counts['missing_metadata']} "
                f"errors={counts['errors']}"
            )

    report = {
        "factoids_dir": str(FACTOIDS_DIR),
        "files_scanned": len(factoid_files),
        "counts": dict(counts),
        "license_counts": dict(license_counts.most_common()),
        "problem_files": problem_files,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\nDone.")
    print(f"Factoid files scanned: {len(factoid_files)}")
    print(f"Has license_label: {counts['has_license_label']}")
    print(f"Missing license_label: {counts['missing_license_label']}")
    print(f"Empty license_label: {counts['empty_license_label']}")
    print(f"Missing/invalid metadata: {counts['missing_metadata']}")
    print(f"Errors: {counts['errors']}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
