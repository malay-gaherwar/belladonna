#!/usr/bin/env python3
"""
Filter already-downloaded ClinicalTrials.gov per-study JSON files.

INPUT:
  artifacts/CTG/studies/*.json   (one study per file)

OUTPUT:
  artifacts/CTG/filtered_studies/*.json   (copied as-is; no JSON rewriting)

LOG:
  logs/ctg_filter_<timestamp>.log

Behavior:
- Filters for INTERVENTIONAL studies that include at least one DRUG (and optionally BIOLOGICAL)
- Excludes studies that contain RADIATION intervention types (configurable)
- Copies matching study JSON files into filtered_studies/ without modifying contents
- Prints + logs progress: total, processed, matched, skipped/errors
- Measures total runtime

Requires:
  pip install (none)  # standard library only
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


# -------------------------
# CONFIG
# -------------------------
CTG_DIR = "artifacts/CTG"
IN_DIR = os.path.join(CTG_DIR, "studies")
OUT_DIR = os.path.join(CTG_DIR, "filtered_studies")
LOG_DIR = "logs"

# "Drug trial" definition
REQUIRE_STUDY_TYPE = "INTERVENTIONAL"
INCLUDE_INTERVENTION_TYPES: Set[str] = {"DRUG"}  # add "BIOLOGICAL" if you want
EXCLUDE_IF_CONTAINS_TYPES: Set[str] = {"RADIATION"}  # exclude if any of these appear

# Progress printing
PRINT_EVERY = 100


# -------------------------
# HELPERS
# -------------------------
def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"ctg_filter_{utc_stamp()}.log")

    logger = logging.getLogger("ctg_filter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def extract_intervention_types(study: Dict[str, Any]) -> Set[str]:
    """
    protocolSection.armsInterventionsModule.interventions[*].type
    """
    interventions = safe_get(study, ["protocolSection", "armsInterventionsModule", "interventions"], default=[])
    types: Set[str] = set()
    if isinstance(interventions, list):
        for it in interventions:
            if isinstance(it, dict):
                t = it.get("type")
                if isinstance(t, str) and t.strip():
                    types.add(t.strip().upper())
    return types


def is_match(study: Dict[str, Any]) -> bool:
    study_type = safe_get(study, ["protocolSection", "designModule", "studyType"], default="")
    if str(study_type).upper() != REQUIRE_STUDY_TYPE:
        return False

    itypes = extract_intervention_types(study)
    if not itypes:
        return False

    # Must include at least one allowed type
    if not (itypes & INCLUDE_INTERVENTION_TYPES):
        return False

    # Exclude if any forbidden types appear
    if itypes & EXCLUDE_IF_CONTAINS_TYPES:
        return False

    return True


def main() -> int:
    logger = setup_logger()
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.json")))
    total = len(paths)
    if total == 0:
        logger.error(f"No input files found in: {IN_DIR}")
        return 1

    t0 = time.time()
    processed = 0
    matched = 0
    errors = 0
    copied = 0

    logger.info(f"INPUT_DIR:  {IN_DIR}")
    logger.info(f"OUTPUT_DIR: {OUT_DIR}")
    logger.info(f"Total study files: {total}")
    logger.info(f"Filter: studyType={REQUIRE_STUDY_TYPE}, include={sorted(INCLUDE_INTERVENTION_TYPES)}, "
                f"exclude_if_contains={sorted(EXCLUDE_IF_CONTAINS_TYPES)}")

    for p in paths:
        processed += 1

        try:
            with open(p, "r", encoding="utf-8") as f:
                study = json.load(f)
            if not isinstance(study, dict):
                raise ValueError("Top-level JSON is not an object")
        except Exception as e:
            errors += 1
            logger.warning(f"Failed to read/parse: {p} | {e}")
            continue

        if is_match(study):
            matched += 1

            # Copy file as-is (no rewriting)
            base = os.path.basename(p)
            dst = os.path.join(OUT_DIR, base)
            try:
                shutil.copy2(p, dst)
                copied += 1
            except Exception as e:
                errors += 1
                logger.warning(f"Failed to copy: {p} -> {dst} | {e}")

        if processed % PRINT_EVERY == 0 or processed == total:
            elapsed = time.time() - t0
            logger.info(
                f"Progress: {processed}/{total} processed | "
                f"matched={matched} | copied={copied} | errors={errors} | "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    logger.info("========== SUMMARY ==========")
    logger.info(f"Total input files:   {total}")
    logger.info(f"Processed:           {processed}")
    logger.info(f"Matched (in filter): {matched}")
    logger.info(f"Copied (out):        {copied}")
    logger.info(f"Errors:              {errors}")
    logger.info(f"Total time:          {elapsed:.2f} seconds")
    if elapsed > 0:
        logger.info(f"Throughput:          {processed/elapsed:.2f} files/sec")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())