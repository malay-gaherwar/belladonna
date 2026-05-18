#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SUMMARY_PATH = Path("artifacts/EPMC/license_summary.json")

SUPPORTED_VERSIONS = {"1.0", "2.0", "2.5", "3.0", "4.0"}
FAMILY_PREFIXES = (
    "CC_BY_NC_ND",
    "CC_BY_NC_SA",
    "CC_BY_NC",
    "CC_BY_ND",
    "CC_BY_SA",
    "CC_BY",
)


def detect_version(text: str) -> str | None:
    t = text.lower()

    patterns = [
        r"creativecommons\.org/licenses/by-nc-nd/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nc-sa/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nc/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nd/(\d\.\d)",
        r"creativecommons\.org/licenses/by-sa/(\d\.\d)",
        r"creativecommons\.org/licenses/by/(\d\.\d)",
        r"\bcc by-nc-nd\s+(\d\.\d)\b",
        r"\bcc by-nc-sa\s+(\d\.\d)\b",
        r"\bcc by-nc\s+(\d\.\d)\b",
        r"\bcc by-nd\s+(\d\.\d)\b",
        r"\bcc by-sa\s+(\d\.\d)\b",
        r"\bcc by\s+(\d\.\d)\b",
        r"creative commons attribution-noncommercial-noderivs\s+(\d\.\d)",
        r"creative commons attribution-noncommercial-sharealike\s+(\d\.\d)",
        r"creative commons attribution-noncommercial\s+(\d\.\d)",
        r"creative commons attribution-no derivatives\s+(\d\.\d)",
        r"creative commons attribution-noderivs\s+(\d\.\d)",
        r"creative commons attribution-sharealike\s+(\d\.\d)",
        r"creative commons attribution\s+(\d\.\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            version = match.group(1)
            if version in SUPPORTED_VERSIONS:
                return version

    return None


def refine_license_label(record: dict) -> tuple[str, bool]:
    original_label = str(record.get("license_label", ""))
    if not original_label.endswith("_UNSPECIFIED"):
        return original_label, False

    raw_text = str(record.get("raw_permissions_text", ""))
    version = detect_version(raw_text)
    if not version:
        return original_label, False

    family_label = original_label.removesuffix("_UNSPECIFIED")
    if family_label not in FAMILY_PREFIXES:
        return original_label, False

    return f"{family_label}_{version.replace('.', '_')}", True


def build_license_counts(per_file: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in per_file:
        label = str(record.get("license_label", "")).strip()
        if label:
            counts[label] += 1
    return dict(counts.most_common())


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_file = data.get("per_file", [])
    if not isinstance(per_file, list):
        raise RuntimeError("Invalid summary format: 'per_file' must be a list.")

    updated_records: list[dict] = []
    changed = 0
    changed_counts: Counter[str] = Counter()
    unchanged_unspecified = 0

    for record in per_file:
        if not isinstance(record, dict):
            updated_records.append(record)
            continue

        updated = dict(record)
        new_label, did_change = refine_license_label(updated)
        if did_change:
            updated["license_label"] = new_label
            updated["license_label_source"] = "refined_from_unspecified"
            changed += 1
            changed_counts[new_label] += 1
        elif str(updated.get("license_label", "")).endswith("_UNSPECIFIED"):
            unchanged_unspecified += 1

        updated_records.append(updated)

    output = dict(data)
    output["per_file"] = updated_records
    output["license_counts"] = build_license_counts(updated_records)
    output["unspecified_refinement"] = {
        "source_summary": str(SUMMARY_PATH),
        "output_summary": str(SUMMARY_PATH),
        "records_reclassified": changed,
        "records_still_unspecified": unchanged_unspecified,
        "reclassified_counts": dict(changed_counts.most_common()),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Updated summary in place: {SUMMARY_PATH}")
    print(f"Reclassified unspecified records: {changed}")
    print(f"Still unspecified: {unchanged_unspecified}")
    if changed_counts:
        print("Top refined labels:")
        for label, count in changed_counts.most_common(10):
            print(f"{label} = {count}")


if __name__ == "__main__":
    main()
