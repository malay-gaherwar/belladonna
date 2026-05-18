from pathlib import Path
import json

ARTIFACTS_DIR = Path("artifacts")

TARGET_FOLDERS = ["AGO", "ASCO", "CTG", "EMA", "ESMO", "FDA"]

LICENSE_BY_FOLDER = {
    "AGO": {"commercial": "not_allowed", "research": "allowed"},
    "ASCO": {"commercial": "not_allowed", "research": "allowed"},
    "ESMO": {"commercial": "not_allowed", "research": "allowed"},
    "FDA": {"commercial": "allowed", "research": "allowed"},
    "EMA": {"commercial": "allowed", "research": "allowed"},
    "CTG": {"commercial": "allowed", "research": "allowed"},
}

# Safety switch: only AGO is actually modified for now.
WRITE_ENABLED_FOLDERS = {"AGO"}


def add_license_label(json_path: Path, license_label: dict) -> bool:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print(f"[SKIP] Not a JSON object: {json_path}")
        return False

    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        print(f"[SKIP] metadata is not an object: {json_path}")
        return False

    before = metadata.get("license_label")

    metadata["license_label"] = {
        "commercial": license_label["commercial"],
        "research": license_label["research"],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return before != metadata["license_label"]


def main():
    stats = {
        "folders_seen": 0,
        "folders_missing": 0,
        "factoids_dirs_missing": 0,
        "files_seen": 0,
        "files_modified": 0,
        "files_already_correct": 0,
        "files_debug_only": 0,
        "files_failed": 0,
    }

    print(f"[START] Artifacts dir: {ARTIFACTS_DIR}")
    print(f"[START] Write-enabled folders: {sorted(WRITE_ENABLED_FOLDERS)}")
    print()

    for folder_name in TARGET_FOLDERS:
        folder = ARTIFACTS_DIR / folder_name
        factoids_dir = folder / "factoid"

        print(f"[FOLDER] {folder_name}")

        if not folder.is_dir():
            stats["folders_missing"] += 1
            print(f"  [MISSING] Folder does not exist: {folder}")
            continue

        stats["folders_seen"] += 1

        if not factoids_dir.is_dir():
            stats["factoids_dirs_missing"] += 1
            print(f"  [MISSING] factoids dir does not exist: {factoids_dir}")
            continue

        license_label = LICENSE_BY_FOLDER[folder_name]
        json_files = sorted(factoids_dir.rglob("*.json"))

        print(f"  [INFO] factoids dir: {factoids_dir}")
        print(f"  [INFO] JSON files found: {len(json_files)}")
        print(f"  [INFO] license_label would be: {license_label}")

        if folder_name not in WRITE_ENABLED_FOLDERS:
            stats["files_debug_only"] += len(json_files)
            for path in json_files[:5]:
                print(f"  [DEBUG ONLY] Would update: {path}")
            if len(json_files) > 5:
                print(f"  [DEBUG ONLY] ... plus {len(json_files) - 5} more files")
            print()
            continue

        for json_path in json_files:
            stats["files_seen"] += 1
            try:
                changed = add_license_label(json_path, license_label)
                if changed:
                    stats["files_modified"] += 1
                    print(f"  [UPDATED] {json_path}")
                else:
                    stats["files_already_correct"] += 1
                    print(f"  [UNCHANGED] Already correct: {json_path}")
            except Exception as e:
                stats["files_failed"] += 1
                print(f"  [FAILED] {json_path}: {e}")

        print()

    print("[SUMMARY]")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print()
    print("[DONE] Only AGO was modified. Other folders were debug-only.")


if __name__ == "__main__":
    main()