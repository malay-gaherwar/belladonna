#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from bs4 import BeautifulSoup
from openai import OpenAI


INPUT_DIR = Path("artifacts/epmc_fulltext/xml")
OUTPUT_DIR = Path("artifacts/epmc_fulltext/filtered_xml")
MAX_FILES = 20
MODEL_NAME = "GPT-OSS-120B"
MAX_RETRIES_ON_UNCLEAR = 3


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


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("VIRTUAL_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_title_abstract_fulltext(xml_text: str) -> Tuple[str, str, str]:
    soup = BeautifulSoup(xml_text, "xml")

    title = ""
    tag = soup.find("article-title")
    if tag:
        title = normalize_whitespace(tag.get_text(" ", strip=True))

    abstract = ""
    tag = soup.find("abstract")
    if tag:
        paras = tag.find_all("p")
        if paras:
            abstract = normalize_whitespace(" ".join(p.get_text(" ", strip=True) for p in paras))
        else:
            abstract = normalize_whitespace(tag.get_text(" ", strip=True))

    full_text = ""
    tag = soup.find("body")
    if tag:
        paras = tag.find_all("p")
        if paras:
            full_text = normalize_whitespace(" ".join(p.get_text(" ", strip=True) for p in paras))
        else:
            full_text = normalize_whitespace(tag.get_text(" ", strip=True))

    return title, abstract, full_text


def build_prompt(title: str, text: str, text_type: str):
    return [
        {
            "role": "system",
            "content": "You are a biomedical relevance classifier. Answer only YES or NO.",
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n\n{text_type}: {text}\n\n"
                "Is this about breast cancer? YES or NO."
            ),
        },
    ]


def parse_yes_no(content: str) -> Optional[bool]:
    content = normalize_whitespace(content).upper()
    if content == "YES":
        return True
    if content == "NO":
        return False

    match = re.search(r"\b(YES|NO)\b", content)
    if match:
        return match.group(1) == "YES"
    return None


def update_token_stats(response, token_stats):
    usage = response.usage

    token_stats["input"] += usage.prompt_tokens
    token_stats["output"] += usage.completion_tokens



def classify_text(client, title, text, text_type, token_stats):
    for attempt in range(1, MAX_RETRIES_ON_UNCLEAR + 2):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_prompt(title, text, text_type),
            max_completion_tokens=8000,
            extra_body={"reasoning_effort": "low"},
        )

        update_token_stats(response, token_stats)

        content = response.choices[0].message.content or ""
        print(f"[DEBUG][Attempt {attempt}] {repr(content)}")

        parsed = parse_yes_no(content)
        if parsed is not None:
            return parsed

        print("[WARN] Retrying...")

    return None


def print_progress(stats, total):
    print(
        f"[PROGRESS] YES: {stats['yes']} | "
        f"NO: {stats['no']} | "
        f"CHECKED: {stats['checked']} | TOTAL: {total}"
    )


def process_file(xml_path, client, stats, unclear_files, total, token_stats):
    xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")

    title, abstract, full_text = extract_title_abstract_fulltext(xml_text)
    if not title:
        title = xml_path.stem

    text = abstract if abstract.strip() else full_text
    text_type = "Abstract" if abstract.strip() else "Full text"

    print("\n==============================")
    print(f"FILE: {xml_path.name}")
    print(f"TITLE: {title}")
    print(f"{text_type.upper()}:")
    print(text)
    print("==============================\n")

    if not text.strip():
        unclear_files.append(xml_path.name)
        stats["checked"] += 1
        print_progress(stats, total)
        return

    result = classify_text(client, title, text, text_type, token_stats)

    if result is True:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(xml_path), str(OUTPUT_DIR / xml_path.name))
        print(f"[YES] {title}")
        stats["yes"] += 1

    elif result is False:
        print(f"[NO] {title}")
        stats["no"] += 1

    else:
        print(f"[UNCLEAR] {xml_path.name}")
        unclear_files.append(xml_path.name)

    stats["checked"] += 1
    print_progress(stats, total)


def main():
    start_time = time.time()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"logs/epmc_filter_{timestamp}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_path)

    # ---- PRINT FULL SCRIPT AT START ----
    print("===== FULL SCRIPT (START) =====\n")
    with open(__file__, "r", encoding="utf-8") as f:
        print(f.read())
    print("\n===== END SCRIPT =====\n")

    print(f"MODEL: {MODEL_NAME}")
    print("REASONING_EFFORT: low\n")

    total_files = len(list(INPUT_DIR.glob("*.xml")))
    xml_files = sorted(INPUT_DIR.glob("*.xml"))[:MAX_FILES]

    client = get_client()

    stats = {"yes": 0, "no": 0, "checked": 0}
    unclear_files = []
    token_stats = {
        "input": 0,
        "output": 0,
    }

    for xml_file in xml_files:
        process_file(
            xml_file,
            client,
            stats,
            unclear_files,
            total_files,
            token_stats,
        )

    print("\nDone.")
    print(f"Runtime: {time.time() - start_time:.2f}s")

    print("\nToken usage:")
    print(f"Input tokens: {token_stats['input']}")
    print(f"Output tokens: {token_stats['output']}")
    print(
        f"Total tokens: "
        f"{token_stats['input'] + token_stats['output']}"
    )

    if unclear_files:
        print("\nUNCLEAR FILES:")
        for f in unclear_files:
            print(f"- {f}")


if __name__ == "__main__":
    main()