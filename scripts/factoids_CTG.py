#!/usr/bin/env python3
"""
ClinicalTrials.gov -> Belladonna Factoids (LLM)

Reads up to MAX_FILES study JSONs from:
  artifacts/CTG/filtered_studies/

For each study, calls your local OpenAI-compatible LLM to extract standalone factoids
(tagged with <<<FACTOID>>> ... <<<END_FACTOID>>>), then writes ONE output JSON per study:

{
  "metadata": {
    "source_family": "Clinical Trials",
    "document_title": "...",
    "document_type": "Clinical Trials",
    "document_year": 2020,
    "file_name": "NCTxxxx.json"
  },
  "factoids": [
    {"id": 1, "factoid_text": "..."},
    ...
  ]
}

Outputs to:
  artifacts/CTG/factoids/

Logs to:
  logs/ctg_factoids_<timestamp>.log

Requires:
  pip install openai
Env:
  VIRTUAL_API_KEY
  BASE_URL
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = Path("artifacts/CTG/filtered_studies")
OUTPUT_DIR = Path("artifacts/CTG/factoids")
LOG_DIR = Path("logs")

MAX_FILES =6708

MODEL_NAME = "GPT-OSS-120B"
MAX_COMPLETION_TOKENS = 4096
SLEEP = 0.2

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

# If a study has huge text, keep prompts bounded.
MAX_FIELD_CHARS = 12000


# -------------------------
# UTIL
# -------------------------
def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ctg_factoids_{utc_stamp()}.log"

    logger = logging.getLogger("ctg_factoids")
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


def load_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY is not set")
    if not base_url:
        raise RuntimeError("BASE_URL is not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def pick_study_files() -> List[Path]:
    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No json files found in {INPUT_DIR}")
    return files[:MAX_FILES]


def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def extract_blocks(text: str) -> List[str]:
    pattern = re.compile(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        re.DOTALL,
    )
    return [b.strip() for b in pattern.findall(text or "")]


def clamp(s: Optional[str], max_chars: int = MAX_FIELD_CHARS) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n[TRUNCATED]"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# -------------------------
# CTG FIELD EXTRACTION
# -------------------------
def build_metadata_from_study(study: Dict[str, Any], input_file: Path) -> Dict[str, Any]:
    ident = safe_get(study, ["protocolSection", "identificationModule"], default={}) or {}
    status = safe_get(study, ["protocolSection", "statusModule"], default={}) or {}

    title = ident.get("briefTitle") or ident.get("officialTitle") or ident.get("nctId") or input_file.stem

    # Prefer the year from "studyFirstPostDateStruct.date" else last update else start date
    date_candidates = [
        safe_get(status, ["studyFirstPostDateStruct", "date"], default=None),
        safe_get(status, ["lastUpdatePostDateStruct", "date"], default=None),
        safe_get(status, ["startDateStruct", "date"], default=None),
        safe_get(status, ["completionDateStruct", "date"], default=None),
    ]
    year: Optional[int] = None
    for d in date_candidates:
        if isinstance(d, str):
            m = re.match(r"^(\d{4})", d.strip())
            if m:
                year = int(m.group(1))
                break

    return {
        "source_family": "Clinical Trials",
        "document_title": str(title),
        "document_type": "Clinical Trials",
        "document_year": year,
        "file_name": input_file.name,
    }


def build_study_context_text(study: Dict[str, Any]) -> str:
    """
    Build a compact, high-signal text payload for the LLM.
    Keep it self-contained: include NCT, title, conditions, key design, interventions, outcomes, eligibility, status.
    """
    proto = study.get("protocolSection") or {}
    ident = proto.get("identificationModule") or {}
    status = proto.get("statusModule") or {}
    design = proto.get("designModule") or {}
    desc = proto.get("descriptionModule") or {}
    conds = proto.get("conditionsModule") or {}
    arms = proto.get("armsInterventionsModule") or {}
    outs = proto.get("outcomesModule") or {}
    elig = proto.get("eligibilityModule") or {}
    refs = proto.get("referencesModule") or {}
    docs = study.get("documentSection") or {}

    nct = ident.get("nctId") or ""
    brief_title = ident.get("briefTitle") or ""
    official_title = ident.get("officialTitle") or ""
    org = (ident.get("organization") or {}).get("fullName") or ""

    overall_status = status.get("overallStatus") or ""
    why_stopped = status.get("whyStopped") or ""

    conditions = conds.get("conditions") or []
    keywords = conds.get("keywords") or []

    study_type = design.get("studyType") or ""
    phases = design.get("phases") or []
    enrollment = (design.get("enrollmentInfo") or {})
    design_info = design.get("designInfo") or {}

    brief_summary = desc.get("briefSummary") or ""
    detailed_description = desc.get("detailedDescription") or ""

    interventions = arms.get("interventions") or []
    arm_groups = arms.get("armGroups") or []

    primary_outcomes = outs.get("primaryOutcomes") or []
    secondary_outcomes = outs.get("secondaryOutcomes") or []

    eligibility_criteria = elig.get("eligibilityCriteria") or ""
    sex = elig.get("sex") or ""
    min_age = elig.get("minimumAge") or ""
    max_age = elig.get("maximumAge") or ""

    has_results = study.get("hasResults", None)

    # Keep references/doc metadata short; they can help the LLM avoid hallucinating
    references = refs.get("references") or []
    large_docs = safe_get(docs, ["largeDocumentModule", "largeDocs"], default=[]) or []

    payload = {
        "nctId": nct,
        "briefTitle": brief_title,
        "officialTitle": official_title,
        "organization": org,
        "status": {"overallStatus": overall_status, "whyStopped": why_stopped},
        "conditions": conditions,
        "keywords": keywords,
        "design": {
            "studyType": study_type,
            "phases": phases,
            "enrollmentInfo": enrollment,
            "designInfo": design_info,
        },
        "description": {
            "briefSummary": clamp(brief_summary),
            "detailedDescription": clamp(detailed_description),
        },
        "armsInterventions": {
            "armGroups": arm_groups,
            "interventions": interventions,
        },
        "outcomes": {
            "primaryOutcomes": primary_outcomes,
            "secondaryOutcomes": secondary_outcomes,
        },
        "eligibility": {
            "sex": sex,
            "minimumAge": min_age,
            "maximumAge": max_age,
            "eligibilityCriteria": clamp(eligibility_criteria),
        },
        "hasResults": has_results,
        "documents": [
            {
                "typeAbbrev": d.get("typeAbbrev"),
                "label": d.get("label"),
                "date": d.get("date"),
                "filename": d.get("filename"),
                "size": d.get("size"),
            }
            for d in large_docs[:20]
            if isinstance(d, dict)
        ],
        "references": [
            {
                "type": r.get("type"),
                "pmid": r.get("pmid"),
                "citation": r.get("citation"),
            }
            for r in references[:20]
            if isinstance(r, dict)
        ],
    }

    # Send as JSON string to keep structure crisp and reduce hallucination.
    return json.dumps(payload, ensure_ascii=False, indent=2)


# -------------------------
# LLM PROMPTS
# -------------------------
def build_prompts(metadata: Dict[str, Any], study_context: str) -> Tuple[str, str]:
    system = (
        "You extract atomic factoids from a ClinicalTrials.gov study record.\n"
        "Output ONLY factoids wrapped in the exact tags.\n"
        "Each factoid must be a single standalone statement that is understandable without context.\n"
        "Do NOT invent details. Use ONLY what is present in the provided study record.\n"
        "If a claim is not explicitly supported, do not include it.\n"
        "Prefer factual statements about: disease/condition, intervention drug(s), study design, population, endpoints, eligibility, status, key dates, and whether results exist.\n"
        "If the study is withdrawn/terminated/hasResults=false, include a clear factoid stating results are not posted / enrollment is zero if present.\n"
        "Avoid duplicates.\n"
        f"Use tags exactly:\n{FACTOID_START}\n...factoid...\n{FACTOID_END}\n"
    )

    # Ensure self-sufficiency: include NCT and disease/drug terms in each statement when applicable.
    user = (
        "You are given a single ClinicalTrials.gov study record in JSON form.\n\n"
        "Task:\n"
        "1) Read the WHOLE record.\n"
        "2) Extract self-sufficient factoids.\n\n"
        "Rules:\n"
        "- One factoid = one sentence.\n"
        "- Each factoid should mention the condition/disease and (if relevant) the drug/intervention name.\n"
        "- When stating design facts, include the study's NCT ID.\n"
        "- Preserve numbers exactly (dates, enrollment counts, time frames).\n"
        "- If there are NO drug interventions, you may still output design/status/eligibility factoids.\n"
        "- Do not output JSON. Do not number. Do not add commentary.\n"
        "- Output ONLY tagged factoids.\n\n"
        f"Metadata (for your awareness only):\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"Study record:\n{study_context}"
    )

    return system, user


def extract_factoids_from_study(client: OpenAI, metadata: Dict[str, Any], study_context: str) -> Tuple[List[str], Dict[str, int], str]:
    system, user = build_prompts(metadata, study_context)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    content = resp.choices[0].message.content or ""
    blocks = [normalize(b) for b in extract_blocks(content)]
    blocks = [b for b in blocks if b]

    usage = getattr(resp, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    return blocks, usage_dict, content


# -------------------------
# MAIN
# -------------------------
def main() -> int:
    logger = setup_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = load_client()
    files = pick_study_files()

    logger.info(f"INPUT_DIR: {INPUT_DIR}")
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"MAX_FILES: {MAX_FILES}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")

    t0 = time.time()
    processed = 0
    written = 0
    total_in_factoids = 0
    tok_in = tok_out = tok_total = 0

    for f in files:
        processed += 1
        logger.info(f"[{processed}/{len(files)}] Processing: {f.name}")

        try:
            study = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(study, dict):
                raise ValueError("Top-level JSON is not an object")
        except Exception as e:
            logger.error(f"Failed to read/parse {f.name}: {e}")
            continue

        metadata = build_metadata_from_study(study, f)
        study_context = build_study_context_text(study)

        try:
            factoid_texts, usage, raw_llm_output = extract_factoids_from_study(client, metadata, study_context)
        except Exception as e:
            logger.error(f"LLM call failed for {f.name}: {e}")
            continue

        tok_in += usage.get("input_tokens", 0)
        tok_out += usage.get("output_tokens", 0)
        tok_total += usage.get("total_tokens", 0)

        # Build output object
        out_obj = {
            "metadata": {
                "source_family": metadata.get("source_family"),
                "document_title": metadata.get("document_title") or "",
                "document_type": metadata.get("document_type"),
                "document_year": metadata.get("document_year"),
                "file_name": metadata.get("file_name"),
            },
            "factoids": [{"id": i + 1, "factoid_text": t} for i, t in enumerate(factoid_texts)],
        }

        out_path = OUTPUT_DIR / f"{f.stem}_factoids.json"
        with out_path.open("w", encoding="utf-8") as out_f:
            json.dump(out_obj, out_f, ensure_ascii=False, indent=2)

        written += 1
        total_in_factoids += len(factoid_texts)

        logger.info(
            f"Wrote: {out_path.name} | factoids={len(factoid_texts)} | "
            f"tokens(in/out/total)={usage['input_tokens']}/{usage['output_tokens']}/{usage['total_tokens']}"
        )

        time.sleep(SLEEP)

    elapsed = time.time() - t0
    logger.info("========== SUMMARY ==========")
    logger.info(f"Files selected:       {len(files)}")
    logger.info(f"Files processed:      {processed}")
    logger.info(f"Files written:        {written}")
    logger.info(f"Total factoids:       {total_in_factoids}")
    logger.info(f"Total time (sec):     {elapsed:.2f}")
    if elapsed > 0:
        logger.info(f"Throughput (files/s): {written/elapsed:.2f}")
    logger.info(f"Tokens input:         {tok_in}")
    logger.info(f"Tokens output:        {tok_out}")
    logger.info(f"Tokens total:         {tok_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())