#!/usr/bin/env python3
"""
Filter Elsevier XML papers for breast cancer relevance using a local LLM.

Behavior:
- Reads XML files from artifacts/elsevier/xml
- Extracts title and abstract from Elsevier XML
- Sends abstract to a local OpenAI-compatible LLM endpoint
- Expects YES or NO
- If YES: prints the paper title and moves the file to artifacts/elsevier/filtered_xml
- Prints file name and abstract
- Saves console output to artifacts/elsevier/output.txt
- Processes up to 100 LLM requests simultaneously

Requirements:
    pip install openai beautifulsoup4 lxml

Environment variables expected:
    VIRTUAL_API_KEY
    BASE_URL
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from bs4 import BeautifulSoup
from openai import AsyncOpenAI


INPUT_DIR = Path("artifacts/elsevier_old/xml")
OUTPUT_DIR = Path("artifacts/elsevier_old/filtered_xml")
MAX_FILES = 345286
MODEL_NAME = "sota"
CONCURRENT_REQUESTS = 100


class Tee:
    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError(
            "Missing environment variable VIRTUAL_API_KEY. "
            "Make sure it is exported in your shell."
        )
    if not base_url:
        raise RuntimeError(
            "Missing environment variable BASE_URL. "
            "Make sure it is exported in your shell."
        )

    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_title_and_abstract(xml_text: str) -> Tuple[str, str]:
    """
    Extract title and abstract from Elsevier-like XML.

    Strategy:
    1. Prefer structured full-text title/abstract fields
    2. Fall back to core metadata fields
    """

    soup = BeautifulSoup(xml_text, "xml")

    # ---------- TITLE ----------
    title = ""

    title_candidates = [
        "ce:title",
        "dc:title",
        "title-text",
        "article-title",
        "title",
    ]

    for tag_name in title_candidates:
        tag = soup.find(tag_name)
        if tag and tag.get_text(strip=True):
            title = normalize_whitespace(tag.get_text(" ", strip=True))
            break

    # ---------- ABSTRACT ----------
    abstract = ""

    # 1. Prefer structured Elsevier abstract
    ce_abstract = soup.find("ce:abstract")
    if ce_abstract:
        paras = ce_abstract.find_all(["ce:simple-para", "ce:para", "para", "p"])
        if paras:
            abstract = normalize_whitespace(
                " ".join(p.get_text(" ", strip=True) for p in paras)
            )
        else:
            abstract = normalize_whitespace(ce_abstract.get_text(" ", strip=True))

    # 2. Fallback to metadata abstract
    if not abstract:
        dc_desc = soup.find("dc:description")
        if dc_desc and dc_desc.get_text(strip=True):
            abstract = normalize_whitespace(dc_desc.get_text(" ", strip=True))

    # 3. Generic fallback
    if not abstract:
        print("no abstract found")
        abstract = ""

    return title, abstract


def build_prompt(title: str, abstract: str) -> list[dict[str, str]]:
    """
    Strict prompt so the model returns only YES or NO.
    """
    system_prompt = (
        "You are a biomedical literature relevance classifier. "
        "Decide whether a paper is relevant to breast cancer in any way. "
        "Relevant includes direct study of breast cancer, breast tumors, "
        "breast carcinoma, mammary tumors in humans or model organisms, "
        "breast cancer biomarkers, breast cancer treatment, diagnosis, prognosis, "
        "breast cancer cell lines, breast oncology, screening, metastasis, "
        "or papers clearly applicable specifically to breast cancer. "
        "If the paper is unrelated, answer NO. "
        "Reply with exactly one word: YES or NO."
    )

    user_prompt = (
        f"Title: {title or 'N/A'}\n\n"
        f"Abstract: {abstract or 'N/A'}\n\n"
        "Is this paper relevant to breast cancer in any way? "
        "Answer exactly YES or NO."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def classify_abstract(
    client: AsyncOpenAI,
    title: str,
    abstract: str,
    model_name: str = MODEL_NAME,
) -> Optional[bool]:
    """
    Returns:
        True  -> relevant
        False -> not relevant
        None  -> could not parse answer
    """
    if not abstract.strip():
        return False

    response = await client.chat.completions.create(
        messages=build_prompt(title, abstract),
        model=model_name,
        max_completion_tokens=8000,
    )

    content = response.choices[0].message.content or ""
    answer = normalize_whitespace(content).upper()
    print(f"[DEBUG RAW MODEL OUTPUT] {repr(content)}")

    if answer == "YES":
        return True
    if answer == "NO":
        return False

    match = re.search(r"\b(YES|NO)\b", answer)
    if match:
        return match.group(1) == "YES"

    return None


async def process_file(
    xml_path: Path,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> None:
    try:
        xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[ERROR] Could not read {xml_path.name}: {exc}")
        return

    title, abstract = extract_title_and_abstract(xml_text)

    if not title:
        title = xml_path.stem

    print("\n==============================")
    print(f"FILE: {xml_path.name}")
    print(f"TITLE: {title}")
    print("ABSTRACT:")
    print(abstract)
    print("==============================\n")

    if not abstract:
        print(f"[SKIP] No abstract found: {xml_path.name} | title={title}")
        return

    try:
        async with semaphore:
            is_relevant = await classify_abstract(client, title, abstract)
    except Exception as exc:
        print(f"[ERROR] LLM request failed for {xml_path.name}: {exc}")
        return

    if is_relevant is True:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT_DIR / xml_path.name

        try:
            shutil.move(str(xml_path), str(destination))
            print(f"[YES] {title}")
        except Exception as exc:
            print(f"[ERROR] Could not move {xml_path.name}: {exc}")

    elif is_relevant is False:
        print(f"[NO] {title}")

    else:
        print(f"[UNCLEAR] Model did not return clean YES/NO for {xml_path.name}")


async def async_main() -> int:
    start_time = time.time()

    log_path = Path("artifacts/elsevier/output.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_path)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return 1

    xml_files = sorted(INPUT_DIR.glob("*.xml"))[:MAX_FILES]

    if not xml_files:
        print(f"[ERROR] No XML files found in {INPUT_DIR}")
        return 1

    try:
        client = get_client()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    print(f"Processing {len(xml_files)} XML files from {INPUT_DIR} ...")
    print(f"Max concurrent LLM calls: {CONCURRENT_REQUESTS}")

    tasks = [
        process_file(xml_file, client, semaphore)
        for xml_file in xml_files
    ]

    await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    print("\nDone.")
    print(f"Total runtime: {total_time:.2f} seconds")

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())