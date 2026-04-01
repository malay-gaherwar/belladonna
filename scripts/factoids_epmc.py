#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# 🔥 CHANGE THIS
INPUT_DIR = Path("artifacts/epmc_fulltext/processed")
OUTPUT_DIR = Path("artifacts/epmc_fulltext/factoids")

MODEL_NAME = os.getenv("MODEL_NAME", "GPT-OSS-120B")
MAX_COMPLETION_TOKENS = 4096
SLEEP = 0.2

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

# ============================================================
# CLIENT
# ============================================================

def load_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY is not set")
    if not base_url:
        raise RuntimeError("BASE_URL is not set")

    return OpenAI(api_key=api_key, base_url=base_url)

# ============================================================
# FILE PICKER
# ============================================================

def pick_json_files() -> List[Path]:
    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DIR}")
    return files

# ============================================================
# UTIL
# ============================================================

def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())

def extract_blocks(text: str) -> List[str]:
    pattern = re.compile(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        re.DOTALL,
    )
    return [b.strip() for b in pattern.findall(text)]

# ============================================================
# PROMPT
# ============================================================

def build_prompts(document_title: str, full_text: str) -> Tuple[str, str]:
    system = (
        "You extract atomic, self-sufficient breast cancer factoids.\n"
        "Each factoid must be fully understandable without context.\n"
        "No explanations. No JSON. Only tagged factoids.\n"
    )

    user = (
        f"You are given a full scientific article.\n\n"
        f"Document title: {document_title}\n\n"
        f"Task:\n"
        f"- Extract high-quality, clinically meaningful factoids.\n"
        f"- Focus on: treatment, biomarkers, outcomes, toxicity, epidemiology, mechanisms.\n"
        f"- Avoid generic statements.\n"
        f"- Preserve numbers exactly.\n"
        f"- Each factoid must be standalone.\n\n"
        f"Format:\n"
        f"{FACTOID_START}\n"
        f"<factoid>\n"
        f"{FACTOID_END}\n\n"
        f"Text:\n{full_text[:20000]}"  # 🔥 limit to avoid token overflow
    )

    return system, user

# ============================================================
# FACTOID EXTRACTION
# ============================================================

def extract_factoids(
    client: OpenAI,
    document_title: str,
    full_text: str,
) -> Tuple[List[str], Dict[str, int]]:

    system, user = build_prompts(document_title, full_text)

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

    return blocks, usage_dict

# ============================================================
# PROCESS FILE
# ============================================================

def process_one_file(client: OpenAI, input_file: Path) -> None:
    data = json.loads(input_file.read_text(encoding="utf-8"))

    metadata = data["metadata"]
    full_text = data["full_text"]

    document_title = metadata.get("TITLE", input_file.stem)

    print(f"\nProcessing: {input_file.name}")

    factoids: List[str] = []
    seen = set()

    blocks, usage = extract_factoids(client, document_title, full_text)

    for b in blocks:
        if b not in seen:
            seen.add(b)
            factoids.append(b)

    output = {
        "metadata": metadata,
        "factoids": [
            {"id": i + 1, "factoid_text": f}
            for i, f in enumerate(factoids)
        ],
    }

    output_json = OUTPUT_DIR / f"{input_file.stem}_factoids.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_json}")
    print(f"Factoids: {len(factoids)}")
    print(f"Tokens used: {usage['total_tokens']}")

    time.sleep(SLEEP)

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    client = load_client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = pick_json_files()
    print(f"Found {len(input_files)} JSON files.")

    for input_file in input_files:
        try:
            process_one_file(client, input_file)
        except Exception as e:
            print(f"Failed on {input_file.name}: {e}")

if __name__ == "__main__":
    main()