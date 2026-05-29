"""Strip `license_label_source` from EPMC factoid metadata.

Reads a pre-built file list (one path per line) — avoids slow NFS glob+sort.
Uses a ThreadPoolExecutor for parallel I/O.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FIELD = "license_label_source"
WORKERS = 64


def process_one(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"error:read:{e.__class__.__name__}:{path.name}"

    meta = data.get("metadata")
    if not isinstance(meta, dict) or FIELD not in meta:
        return "unchanged"

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
        return f"error:write:{e.__class__.__name__}:{path.name}"

    return "stripped"


def main():
    if len(sys.argv) != 2:
        print("usage: strip_license_label_source_v2.py <file_list.txt>", file=sys.stderr)
        return 2

    list_path = Path(sys.argv[1])
    with open(list_path, "r") as f:
        files = [Path(line.strip()) for line in f if line.strip()]

    n = len(files)
    print(f"file list:  {list_path}")
    print(f"files:      {n}")
    print(f"workers:    {WORKERS}")
    print(f"field:      {FIELD!r}")
    print()

    counts = {"scanned": 0, "stripped": 0, "unchanged": 0, "errors": 0}
    t0 = time.time()
    last_print = t0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed(ex.submit(process_one, p) for p in files):
            res = fut.result()
            counts["scanned"] += 1
            if res == "stripped":
                counts["stripped"] += 1
            elif res == "unchanged":
                counts["unchanged"] += 1
            else:
                counts["errors"] += 1
                print(f"  {res}")

            now = time.time()
            if counts["scanned"] % 5000 == 0 or now - last_print > 60:
                rate = counts["scanned"] / max(1e-9, now - t0)
                eta = (n - counts["scanned"]) / max(1e-9, rate)
                print(f"  [{counts['scanned']}/{n}] "
                      f"stripped={counts['stripped']} "
                      f"errors={counts['errors']} "
                      f"rate={rate:.0f}/s eta={eta/60:.1f}min",
                      flush=True)
                last_print = now

    elapsed = time.time() - t0
    rate = counts["scanned"] / max(1e-9, elapsed)
    print()
    print(f"done in {elapsed:.1f}s ({rate:.0f}/s) -> {counts}")


if __name__ == "__main__":
    sys.exit(main())
