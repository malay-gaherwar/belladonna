#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAI

INPUT_DIR = Path("AGO_REF_PDF/output_md_llm/")
OUTPUT_DIR = Path("AGO_REF_PDF/factoid/")

MODEL_NAME = "GPT-OSS-120B"
MAX_COMPLETION_TOKENS = 4096
SLEEP = 0.2

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

PAGE_PATTERN = re.compile(
    r"<!-- PAGE (\d+) START -->(.*?)<!-- PAGE \1 END -->",
    re.DOTALL,
)

H1_PATTERN = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def load_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY is not set")
    if not base_url:
        raise RuntimeError("BASE_URL is not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def pick_md_file() -> Path:
    files = sorted(INPUT_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No md files found in {INPUT_DIR}")
    return files[0]


def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def parse_pages(md_text: str) -> List[Dict[str, Any]]:
    matches = PAGE_PATTERN.findall(md_text)
    if not matches:
        return [{"page_num": 1, "heading": None, "content": md_text.strip()}]

    pages: List[Dict[str, Any]] = []

    for page_num_str, raw_content in matches:
        page_num = int(page_num_str)
        content = raw_content.strip()

        heading_match = H1_PATTERN.search(content)
        heading = heading_match.group(1).strip() if heading_match else None

        pages.append(
            {
                "page_num": page_num,
                "heading": heading,
                "content": content,
            }
        )

    return pages


def extract_blocks(text: str) -> List[str]:
    pattern = re.compile(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        re.DOTALL,
    )
    return [b.strip() for b in pattern.findall(text)]


def build_metadata(input_file: Path) -> Dict[str, Any]:
    stem = input_file.stem

    year = None
    m = re.search(r"_(\d{4})[A-Z]?_", stem)
    if m:
        year = int(m.group(1))

    title = stem
    title = re.sub(r"^AGO_\d{4}[A-Z]?_\d+_", "", title)
    title = re.sub(r"_REF$", "", title)
    title = title.replace("_", " ").strip()

    return {
        "source_family": "AGO Guidelines Breast",
        "document_title": title if title else input_file.stem.replace("_", " "),
        "document_year": year,
        "file_name": input_file.name,
    }


def build_prompts(page_num: int, heading: str | None, page_content: str) -> Tuple[str, str]:
    system = (
        "You extract atomic breast cancer factoids from one markdown page.\n"
        "You must evaluate the whole page as a single unit before deciding what factoids to create.\n"
        "No explanation. No grouping. No metadata. No JSON.\n"
        "Each factoid must be one clear, standalone statement.\n"
        "If the page is mainly references, bibliography, or continuation references for the previous page, output nothing.\n"
    )

    heading_text = heading if heading else "[no heading found]"

    user = (
        f"You are given one markdown page from an AGO breast cancer guideline.\n\n"
        f"Page number: {page_num}\n"
        f"Page heading: {heading_text}\n\n"
        f"Task:\n"
        f"1. Read the WHOLE page first.\n"
        f"2. Decide whether this page contains actual content worth turning into factoids.\n"
        f"3. If the page is mostly references, continuation references, citation lists, or bibliography for the previous page, output nothing.\n"
        f"4. If the page contains recommendations, findings, risk factors, protective factors, evidence summaries, quantitative results, or table-based claims, extract the relevant factoids.\n\n"
        f"Rules:\n"
        f"- Use the whole page context before creating any factoid.\n"
        f"- Do not create factoids from references alone.\n"
        f"- Do not create factoids from author lists, journal names, page numbers, or boilerplate.\n"
        f"- Preserve numbers exactly.\n"
        f"- Merge table rows into clean statements when appropriate.\n"
        f"- One factoid = one statement.\n"
        f"- Avoid duplicates.\n"
        f"- Output only tagged factoids in this exact format:\n\n"
        f"{FACTOID_START}\n"
        f"<factoid text>\n"
        f"{FACTOID_END}\n\n"
        f"- No bullet list.\n"
        f"- No numbering.\n"
        f"- No commentary.\n"
        f"- If no factoids should be created from this page, output nothing.\n\n"
        f"Markdown page:\n\n"
        f"{page_content}"
    )

    return system, user


def extract_factoids_from_page(
    client: OpenAI,
    page_num: int,
    heading: str | None,
    page_content: str,
) -> Tuple[List[str], Dict[str, int]]:
    system, user = build_prompts(page_num, heading, page_content)

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


def main():
    client = load_client()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = pick_md_file()
    md_text = input_file.read_text(encoding="utf-8")

    pages = parse_pages(md_text)
    metadata = build_metadata(input_file)

    print(f"Processing: {input_file.name}")
    print(f"Pages detected: {len(pages)}")

    factoids: List[str] = []
    seen = set()

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    log_lines = []
    log_lines.append(f"Processing file: {input_file.name}")
    log_lines.append(f"Pages detected: {len(pages)}")
    log_lines.append("")

    for idx, page in enumerate(pages, 1):
        page_num = page["page_num"]
        heading = page["heading"]
        content = page["content"]

        print(f"Page {idx}/{len(pages)} (page {page_num})")

        blocks, usage = extract_factoids_from_page(
            client=client,
            page_num=page_num,
            heading=heading,
            page_content=content,
        )

        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_tokens += usage["total_tokens"]

        before_count = len(factoids)
        for b in blocks:
            if b not in seen:
                seen.add(b)
                factoids.append(b)
        added_count = len(factoids) - before_count

        log_lines.append(f"Page {page_num}")
        log_lines.append(f"Heading: {heading if heading else '[no heading]'}")
        log_lines.append(f"Input tokens: {usage['input_tokens']}")
        log_lines.append(f"Output tokens: {usage['output_tokens']}")
        log_lines.append(f"Total tokens: {usage['total_tokens']}")
        log_lines.append(f"Factoids added: {added_count}")
        log_lines.append("")

        time.sleep(SLEEP)

    output = {
        "document_metadata": metadata,
        "factoids": [
            {"id": i + 1, "factoid_text": f}
            for i, f in enumerate(factoids)
        ],
    }

    output_json = OUTPUT_DIR / f"{input_file.stem}_factoids.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log_lines.append("SUMMARY")
    log_lines.append(f"Input tokens total: {total_input_tokens}")
    log_lines.append(f"Output tokens total: {total_output_tokens}")
    log_lines.append(f"Total tokens overall: {total_tokens}")
    log_lines.append(f"Final factoid count: {len(factoids)}")

    output_log = OUTPUT_DIR / f"{input_file.stem}_factoids.log"
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"Saved JSON: {output_json}")
    print(f"Saved log: {output_log}")
    print(f"Factoid count: {len(factoids)}")
    print(f"Input tokens total: {total_input_tokens}")
    print(f"Output tokens total: {total_output_tokens}")
    print(f"Total tokens overall: {total_tokens}")


if __name__ == "__main__":
    main()