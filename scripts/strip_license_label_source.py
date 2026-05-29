"""Remove the `license_label_source` field from every factoid file's metadata.

Scans all source factoid directories on the server, in parallel.
Files without the field are read once and left untouched.
Atomic writes (tmp + rename) for files that need a change.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/bulk-saturn/malaygaherwar/belladonna")

SOURCE_DIRS = [
    ROOT / "artifacts/EPMC/factoids",
    ROOT / "artifacts/Elsevier/factoids",
    ROOT / "artifacts/CTG/factoids",
    ROOT / "artifacts/AGO/factoids",
    ROOT / "artifacts/ASCO/factoids",
    ROOT / "artifacts/ESMO/factoids",
    ROOT / "artifacts/EMA/factoids",
    ROOT / "artifacts/FDA/factoids",
]

FIELD = "license_label_source"
WORKERS = 32


def process_one(path: Path) -> tuple[Path, str]:
    """Return (path, action) where action in {'unchanged','stripped','error:...'}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return path, f"error:read:{e.__class__.__name__}"

    meta = data.get("metadata")
    if not isinstance(meta, dict) or FIELD not in meta:
        return path, "unchanged"

    meta.pop(FIELD, None)

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return path, f"error:write:{e.__class__.__name__}"

    return path, "stripped"


def process_dir(d: Path) -> dict[str, int]:
    if not d.exists():
        print(f"[{d}] skipped — directory missing")
        return {"scanned": 0, "stripped": 0, "unchanged": 0, "errors": 0}

    files = sorted(d.glob("*_factoids.json"))
    n = len(files)
    print(f"[{d}] {n} files")

    counts = {"scanned": 0, "stripped": 0, "unchanged": 0, "errors": 0}
    t0 = time.time()
    last_print = t0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed(ex.submit(process_one, p) for p in files):
            _, action = fut.result()
            counts["scanned"] += 1
            if action == "stripped":
                counts["stripped"] += 1
            elif action == "unchanged":
                counts["unchanged"] += 1
            else:
                counts["errors"] += 1
                print(f"  {action}")

            now = time.time()
            if counts["scanned"] % 2000 == 0 or now - last_print > 30:
                rate = counts["scanned"] / max(1e-9, now - t0)
                print(f"  [{d.parent.name}] {counts['scanned']}/{n} "
                      f"stripped={counts['stripped']} rate={rate:.0f}/s")
                last_print = now

    elapsed = time.time() - t0
    rate = counts["scanned"] / max(1e-9, elapsed)
    print(f"[{d}] done in {elapsed:.1f}s ({rate:.0f}/s) -> {counts}")
    return counts


def main():
    print(f"WORKERS = {WORKERS}")
    print(f"FIELD   = {FIELD!r}")
    print()

    total = {"scanned": 0, "stripped": 0, "unchanged": 0, "errors": 0}
    for d in SOURCE_DIRS:
        c = process_dir(d)
        for k in total:
            total[k] += c.get(k, 0)
        print()

    print("=" * 70)
    print(f"GRAND TOTAL -> {total}")


if __name__ == "__main__":
    main()
