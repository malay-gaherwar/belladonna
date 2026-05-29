from pathlib import Path
import json

ARTIFACTS_DIR = Path("artifacts")

LICENSE_INFO_BY_FOLDER = {
    "ASCO": {
        "copyright": "ASCO copyright",
        "commercial_use": "no",
        "personal_use": "yes",
    },
    "ESMO": {
        "copyright": "ESMO copyright",
        "commercial_use": "no",
        "personal_use": "yes",
    },
    "EMA": {
        "copyright": "EMA / EU public sector information",
        "commercial_use": "yes",
        "personal_use": "yes",
    },
    "FDA": {
        "copyright": "U.S. Government work, public domain",
        "commercial_use": "yes",
        "personal_use": "yes",
    },
    "CTG": {
        "copyright": "U.S. Government work, public domain (ClinicalTrials.gov)",
        "commercial_use": "yes",
        "personal_use": "yes",
    },
}

# If True, overwrite metadata["license_info"] even when it already exists.
FORCE_REWRITE = False


def add_license_info(json_path: Path, license_info: dict) -> str:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return "skipped_not_object"

    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return "skipped_bad_metadata"

    existing = metadata.get("license_info")
    if existing == license_info and not FORCE_REWRITE:
        return "unchanged"
    if existing is not None and not FORCE_REWRITE:
        return "skipped_already_present"

    metadata["license_info"] = dict(license_info)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return "updated"


def main() -> None:
    print(f"[START] Artifacts dir: {ARTIFACTS_DIR}")
    print(f"[START] Force rewrite: {FORCE_REWRITE}")
    print()

    totals = {
        "updated": 0,
        "unchanged": 0,
        "skipped_already_present": 0,
        "skipped_not_object": 0,
        "skipped_bad_metadata": 0,
        "errors": 0,
    }

    for folder_name, license_info in LICENSE_INFO_BY_FOLDER.items():
        folder = ARTIFACTS_DIR / folder_name
        factoids_dir = folder / "factoids"

        print(f"[FOLDER] {folder_name}  ->  license_info = {license_info}")

        if not factoids_dir.is_dir():
            print(f"  [MISSING] {factoids_dir}")
            print()
            continue

        json_files = sorted(factoids_dir.glob("*.json"))
        print(f"  [INFO] JSON files: {len(json_files)}")

        per_folder = {k: 0 for k in totals}

        for i, json_path in enumerate(json_files, start=1):
            try:
                status = add_license_info(json_path, license_info)
            except Exception as e:
                per_folder["errors"] += 1
                totals["errors"] += 1
                print(f"  [ERROR] {json_path.name}: {e}")
                continue

            per_folder[status] = per_folder.get(status, 0) + 1
            totals[status] = totals.get(status, 0) + 1

            if i % 1000 == 0:
                print(
                    f"  Processed {i}/{len(json_files)} | "
                    f"updated={per_folder['updated']} "
                    f"already_present={per_folder['skipped_already_present']} "
                    f"unchanged={per_folder['unchanged']} "
                    f"errors={per_folder['errors']}"
                )

        print(f"  [DONE {folder_name}] {per_folder}")
        print()

    print("[SUMMARY]")
    for key, value in totals.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
