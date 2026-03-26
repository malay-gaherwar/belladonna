#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

INPUT_DIR = Path("ASCO/output_md_llm_asco")
OUTPUT_DIR = Path("ASCO/factoids")

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
SOURCE_PDF_PATTERN = re.compile(r"^Source PDF:\s*`([^`]+)`", re.MULTILINE)


def load_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY is not set")
    if not base_url:
        raise RuntimeError("BASE_URL is not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def pick_md_files() -> List[Path]:
    files = sorted(INPUT_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No md files found in {INPUT_DIR}")
    return files


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


def infer_document_year(file_stem: str, source_pdf_name: str | None) -> int | None:
    candidates = [file_stem]
    if source_pdf_name:
        candidates.append(Path(source_pdf_name).stem)

    for candidate in candidates:
        m = re.search(r"(19|20)\d{2}", candidate)
        if m:
            return int(m.group(0))
    return None


def prettify_title_from_stem(file_stem: str) -> str:
    title = file_stem.replace("_", " ").replace("-", " ").strip()
    title = re.sub(r"\s+", " ", title)
    return title


def extract_source_pdf_name(md_text: str) -> str | None:
    m = SOURCE_PDF_PATTERN.search(md_text)
    return m.group(1).strip() if m else None


def extract_document_title(md_text: str, input_file: Path) -> str:
    top_h1_match = H1_PATTERN.search(md_text)
    if top_h1_match:
        title = normalize(top_h1_match.group(1))
        if title:
            return title

    source_pdf_name = extract_source_pdf_name(md_text)
    if source_pdf_name:
        return prettify_title_from_stem(Path(source_pdf_name).stem)

    return prettify_title_from_stem(input_file.stem)


def build_metadata(input_file: Path, md_text: str) -> Dict[str, Any]:
    source_pdf_name = extract_source_pdf_name(md_text)
    document_title = extract_document_title(md_text, input_file)
    document_year = infer_document_year(input_file.stem, source_pdf_name)

    return {
        "source_family": "ASCO Breast Cancer Guidelines",
        "document_title": document_title,
        "document_type": "Guideline",
        "document_year": document_year,
        "file_name": input_file.name,
        "source_pdf_name": source_pdf_name,
    }


def build_prompts(page_num: int, heading: str | None, page_content: str) -> Tuple[str, str]:
    system = (
        "You extract atomic breast cancer factoids from one markdown page.\n"
        "You must evaluate the whole page as a single unit before deciding what factoids to create.\n"
        "No explanation. No grouping. No metadata. No JSON.\n"
        "Each factoid must be one clear, standalone statement.\n"
        "If the page is mainly references, bibliography, author lists, disclosures, affiliations, appendix membership tables, "
        "or continuation references for the previous page, output nothing.\n"
    )

    heading_text = heading if heading else "[no heading found]"

    user = (
        f"You are given one markdown page from an ASCO breast cancer guideline.\n\n"
        f"Page number: {page_num}\n"
        f"Page heading: {heading_text}\n\n"
        f"Task:\n"
        f"1. Read the WHOLE page first.\n"
        f"2. Decide whether this page contains actual clinical or epidemiologic content worth turning into factoids.\n"
        f"3. If the page is mostly references, citation lists, bibliography, author lists, affiliations, disclosures, appendix membership tables, "
        f"or boilerplate, output nothing.\n"
        f"4. If the page contains recommendations, findings, treatment options, resource-stratified guidance, risk factors, "
        f"evidence summaries, quantitative results, epidemiologic burden, or table-based clinical claims, extract the relevant factoids.\n\n"
        f"Rules:\n"
        f"- Use the whole page context before creating any factoid.\n"
        f"- Do not create factoids from references alone.\n"
        f"- Do not create factoids from author lists, journal names, page numbers, copyright text, or boilerplate.\n"
        f"- Preserve numbers exactly.\n"
        f"- Preserve qualifiers such as Basic, Limited, Enhanced, first-line, second-line, third-line, HR-positive, HER2-positive, triple-negative, PD-L1-positive.\n"
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


def process_one_file(client: OpenAI, input_file: Path) -> None:
    md_text = input_file.read_text(encoding="utf-8")
    pages = parse_pages(md_text)
    metadata = build_metadata(input_file, md_text)

    print(f"\nProcessing: {input_file.name}")
    print(f"Pages detected: {len(pages)}")

    factoids: List[str] = []
    seen = set()

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    log_lines = []
    log_lines.append(f"Processing file: {input_file.name}")
    log_lines.append(f"Document title: {metadata['document_title']}")
    log_lines.append(f"Document year: {metadata['document_year']}")
    log_lines.append(f"Source PDF: {metadata['source_pdf_name']}")
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
        "metadata": metadata,
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


def main() -> None:
    client = load_client()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = pick_md_files()
    print(f"Found {len(input_files)} markdown files.")

    for input_file in input_files:
        try:
            process_one_file(client, input_file)
        except Exception as e:
            print(f"Failed on {input_file.name}: {e}")


if __name__ == "__main__":
    main()