#!/usr/bin/env python3
"""
ClinicalTrials.gov v2 API - download ALL studies for an exact phrase query,
with exponential retry + resume support.

Key properties:
- No JSON parsing for saving (we save raw response bytes EXACTLY as returned).
- We DO parse minimally to read nextPageToken and list of NCT IDs so we can paginate
  and fetch per-study endpoints. (Parsing doesn't alter the saved files.)
- START_AT lets you skip the first N studies deterministically (by download order).
- Resume supported via a checkpoint file that stores downloaded_count and nextPageToken.

Outputs (in artifacts/CTG):
- pages/search_page_<index>_<timestamp>.json         (raw page responses)
- studies/<global_index>_<NCTID>.json                (raw per-study responses)
- checkpoint.json                                    (resume state)
- run_meta_<timestamp>.json                          (config + run metadata)

Requires: pip install requests
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

CONFIG = {
    "SEARCH_URL": "https://clinicaltrials.gov/api/v2/studies",
    "STUDY_URL_BASE": "https://clinicaltrials.gov/api/v2/studies",  # + "/{nctId}"
    "OUTDIR": "artifacts/CTG",
    "QUERY_PHRASE": "breast cancer",
    "PAGE_SIZE": 1000,  # try large; API may clamp it. Still fine.
    "TIMEOUT_SECS": 60,

    # Resume controls:
    "START_AT": 0,      # skip first N studies (you can change later)
    "RESUME": True,     # if checkpoint exists, resume from it

    # Retry controls:
    "MAX_RETRIES": 12,          # per request
    "BACKOFF_BASE_SECS": 1.0,   # exponential base
    "BACKOFF_MAX_SECS": 120.0,  # cap
    "JITTER_FRAC": 0.25,        # +/- 25% jitter
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs(base: str) -> Dict[str, str]:
    pages_dir = os.path.join(base, "pages")
    studies_dir = os.path.join(base, "studies")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(studies_dir, exist_ok=True)
    return {"pages": pages_dir, "studies": studies_dir}


@dataclass
class Checkpoint:
    downloaded_count: int
    next_page_token: Optional[str]
    page_index: int

    @staticmethod
    def path(outdir: str) -> str:
        return os.path.join(outdir, "checkpoint.json")

    @staticmethod
    def load(outdir: str) -> Optional["Checkpoint"]:
        p = Checkpoint.path(outdir)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Checkpoint(
            downloaded_count=int(d.get("downloaded_count", 0)),
            next_page_token=d.get("next_page_token"),
            page_index=int(d.get("page_index", 0)),
        )

    def save(self, outdir: str) -> None:
        p = Checkpoint.path(outdir)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "downloaded_count": self.downloaded_count,
                    "next_page_token": self.next_page_token,
                    "page_index": self.page_index,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, p)


def backoff_sleep(attempt: int) -> None:
    base = CONFIG["BACKOFF_BASE_SECS"]
    cap = CONFIG["BACKOFF_MAX_SECS"]
    # exponential: base * 2^(attempt-1)
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    # jitter: +/- jitter_frac
    jitter = CONFIG["JITTER_FRAC"]
    lo = delay * (1.0 - jitter)
    hi = delay * (1.0 + jitter)
    time.sleep(random.uniform(lo, hi))


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> requests.Response:
    last_exc: Optional[Exception] = None

    for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
        try:
            resp = session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=CONFIG["TIMEOUT_SECS"],
            )

            # Retry on common transient statuses
            if resp.status_code in (429, 500, 502, 503, 504):
                print(f"[WARN] HTTP {resp.status_code} for {resp.url} (attempt {attempt}/{CONFIG['MAX_RETRIES']})")
                if attempt < CONFIG["MAX_RETRIES"]:
                    backoff_sleep(attempt)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            return resp

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            print(f"[WARN] Request failed (attempt {attempt}/{CONFIG['MAX_RETRIES']}): {e}")
            if attempt < CONFIG["MAX_RETRIES"]:
                backoff_sleep(attempt)
                continue
            raise

    # should never reach
    if last_exc:
        raise last_exc
    raise RuntimeError("request_with_retry failed unexpectedly")


def save_raw_bytes(path: str, content: bytes) -> None:
    with open(path, "wb") as f:
        f.write(content)


def build_search_params(next_page_token: Optional[str]) -> dict:
    query_term = f"\"{CONFIG['QUERY_PHRASE']}\""  # exact phrase
    params = {
        "format": "json",
        "query.cond": query_term,
        "pageSize": CONFIG["PAGE_SIZE"],
        "countTotal": "true",
    }
    if next_page_token:
        params["pageToken"] = next_page_token
    return params


def extract_studies_and_token(page_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    studies = page_json.get("studies") or []
    token = page_json.get("nextPageToken")
    return studies, token


def get_nct_id(study_obj: Dict[str, Any]) -> Optional[str]:
    return (
        (study_obj.get("protocolSection") or {})
        .get("identificationModule", {})
        .get("nctId")
    )


def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def main() -> int:
    outdir = CONFIG["OUTDIR"]
    dirs = ensure_dirs(outdir)
    ts = utc_stamp()

    # Save run metadata
    meta_path = os.path.join(outdir, f"run_meta_{ts}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_started_utc": datetime.now(timezone.utc).isoformat(),
                "config": CONFIG,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] OUTDIR: {outdir}")
    print(f"[INFO] Query: \"{CONFIG['QUERY_PHRASE']}\"")
    print(f"[INFO] START_AT: {CONFIG['START_AT']}")
    print(f"[INFO] Meta: {meta_path}")

    # Resume
    cp = None
    if CONFIG["RESUME"]:
        cp = Checkpoint.load(outdir)
        if cp:
            print(f"[INFO] Resuming from checkpoint: downloaded_count={cp.downloaded_count}, "
                  f"page_index={cp.page_index}, next_page_token={'YES' if cp.next_page_token else 'NO'}")

    downloaded_count = cp.downloaded_count if cp else 0
    page_index = cp.page_index if cp else 0
    next_page_token = cp.next_page_token if cp else None

    # START_AT: skip first N studies (by download order)
    start_at = int(CONFIG["START_AT"])
    if downloaded_count < start_at:
        print(f"[INFO] START_AT requires skipping {start_at - downloaded_count} studies (continuing pagination).")

    session = requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": "ctgov-raw-downloader/2.0",
    }

    total_seen = downloaded_count  # counts studies encountered (including skipped)
    total_saved = downloaded_count # counts studies saved as individual files

    while True:
        params = build_search_params(next_page_token)
        full_url = f"{CONFIG['SEARCH_URL']}?{urlencode(params)}"
        print(f"\n[INFO] Fetch page {page_index} URL: {full_url}")

        resp = request_with_retry(session, "GET", CONFIG["SEARCH_URL"], params=params, headers=headers)

        # Save raw page bytes exactly as returned
        page_path = os.path.join(dirs["pages"], f"search_page_{page_index:06d}_{ts}.json")
        save_raw_bytes(page_path, resp.content)
        print(f"[OK] Saved raw page: {page_path} ({len(resp.content)} bytes)")

        # Parse only to find NCT IDs and next token
        page_json = resp.json()
        studies, next_token = extract_studies_and_token(page_json)

        print(f"[INFO] Page studies: {len(studies)} | nextPageToken: {'YES' if next_token else 'NO'}")

        # Download each study record individually (raw)
        for s in studies:
            total_seen += 1

            nct = get_nct_id(s)
            if not nct:
                print("[WARN] No nctId in study object; skipping per-study download.")
                continue

            if total_seen <= start_at:
                # skip until we cross START_AT
                if total_seen % 100 == 0:
                    print(f"[INFO] Skipping... seen={total_seen} (START_AT={start_at})")
                continue

            # per-study download
            study_url = f"{CONFIG['STUDY_URL_BASE']}/{nct}"
            study_params = {"format": "json"}
            study_resp = request_with_retry(session, "GET", study_url, params=study_params, headers=headers)

            # Save raw bytes exactly as returned
            idx = total_saved + 1
            fname = f"{idx:07d}_{sanitize_filename(nct)}.json"
            study_path = os.path.join(dirs["studies"], fname)
            save_raw_bytes(study_path, study_resp.content)
            total_saved += 1

            if total_saved % 50 == 0:
                print(f"[INFO] Saved studies: {total_saved} (latest: {nct})")

            # Update checkpoint frequently (so resume is safe)
            Checkpoint(
                downloaded_count=total_saved,
                next_page_token=next_page_token,  # checkpoint token BEFORE advancing page; ok for resume
                page_index=page_index,
            ).save(outdir)

        # Advance pagination
        next_page_token = next_token
        page_index += 1

        # Save checkpoint at end of page with new token
        Checkpoint(
            downloaded_count=total_saved,
            next_page_token=next_page_token,
            page_index=page_index,
        ).save(outdir)

        if not next_page_token:
            print("\n[DONE] No nextPageToken. Reached end of results.")
            print(f"[DONE] Total saved study files: {total_saved}")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
