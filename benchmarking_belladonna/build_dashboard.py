#!/usr/bin/env python3
"""Rebuild the dashboards from existing benchmark result files, without
re-running the RAG.

  - Rebuilds compare/data.js from every rag__*__expert200.json (the
    side-by-side comparison of backing LLMs).
  - Rebuilds dashboard/data.js (single-run detail view) from the most
    recently produced result file.

    python3 build_dashboard.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
from run_rag import build_summary, build_compare, RESULTS_DIR, DATASET_ID  # noqa: E402


def main():
    files = sorted(RESULTS_DIR.glob(f"rag__*__{DATASET_ID}.json"))
    if not files:
        print(f"No RAG result files in {RESULTS_DIR}. Run bench/run_rag.py first.")
        raise SystemExit(1)

    # Comparison across all backing-LLM runs.
    build_compare()

    # Single-run detail view = the most recently modified result file.
    latest = max(files, key=lambda f: f.stat().st_mtime)
    build_summary(latest)
    print(f"Detail view points at: {latest.name}")
    print("Open compare/index.html (leaderboard) or dashboard/index.html (detail).")


if __name__ == "__main__":
    main()
