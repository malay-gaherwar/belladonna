#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

from openai import AsyncOpenAI


INPUT_DIR = Path("artifacts/elsevier/non_duplicates")
OUTPUT_DIR = Path("artifacts/elsevier/filtered_xml")
REJECTED_DIR = Path("artifacts/elsevier/rejected_xml")
FILTERING_ERROR_DIR = Path("artifacts/elsevier/filtering_error")
PROCESSED_LOG = Path("logs/elsevier_processed_files.txt")

MAX_FILES = 338266
MODEL_NAME = "GPT-OSS-120B"
MAX_RETRIES_ON_UNCLEAR = 3
MAX_COMPLETION_TOKENS = 1000
CONCURRENCY = 50

NS = {
    "default": "http://www.elsevier.com/xml/svapi/article/dtd",
    "dc": "http://purl.org/dc/elements/1.1/",
}


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
    return AsyncOpenAI(
        api_key=os.getenv("VIRTUAL_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_text_recursive(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return normalize_whitespace("".join(elem.itertext()))


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_first_by_local_name(root: ET.Element, name: str) -> Optional[ET.Element]:
    for elem in root.iter():
        if local_name(elem.tag) == name:
            return elem
    return None


def extract_title_and_abstract(xml_text: str) -> Tuple[str, str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return "", ""

    title = ""
    abstract = ""

    coredata = root.find("default:coredata", NS)
    if coredata is not None:
        title = get_text_recursive(coredata.find("dc:title", NS))
        abstract = get_text_recursive(coredata.find("dc:description", NS))

    if not title:
        title_elem = find_first_by_local_name(root, "title")
        title = get_text_recursive(title_elem)

    if not abstract:
        desc_elem = find_first_by_local_name(root, "description")
        abstract = get_text_recursive(desc_elem)

    if not abstract:
        abstract_elem = find_first_by_local_name(root, "abstract")
        if abstract_elem is not None:
            para_texts = []
            for elem in abstract_elem.iter():
                if local_name(elem.tag) in {"simple-para", "para", "p"}:
                    text = get_text_recursive(elem)
                    if text:
                        para_texts.append(text)
            if para_texts:
                abstract = normalize_whitespace(" ".join(para_texts))
            else:
                abstract = get_text_recursive(abstract_elem)

    return title, abstract


def build_prompt(title: str, abstract: str):
    return [
        {
            "role": "system",
            "content": "You are a biomedical relevance classifier. Answer only YES or NO.",
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n\n"
                f"Abstract: {abstract}\n\n"
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


def move_to_folder(src: Path, dst_folder: Path) -> None:
    dst_folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_folder / src.name))


def load_processed_files() -> set[str]:
    if not PROCESSED_LOG.exists():
        return set()

    with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


async def mark_processed(file_name: str, lock: asyncio.Lock) -> None:
    async with lock:
        PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
            f.write(file_name + "\n")


async def update_token_stats(response, token_stats, lock: asyncio.Lock):
    usage = response.usage
    if usage is None:
        return

    async with lock:
        token_stats["input"] += getattr(usage, "prompt_tokens", 0) or 0
        token_stats["output"] += getattr(usage, "completion_tokens", 0) or 0


async def classify_text(client, title, abstract, token_stats, token_lock, file_name: str):
    for attempt in range(1, MAX_RETRIES_ON_UNCLEAR + 2):
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_prompt(title, abstract),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            extra_body={"reasoning_effort": "low"},
        )

        await update_token_stats(response, token_stats, token_lock)

        content = response.choices[0].message.content or ""
        print(f"[DEBUG][{file_name}][Attempt {attempt}] {repr(content)}")

        parsed = parse_yes_no(content)
        if parsed is not None:
            return parsed

        print(f"[WARN] {file_name} | retrying...")

    return None


async def print_progress(stats, total, stats_lock: asyncio.Lock):
    async with stats_lock:
        print(
            f"[PROGRESS] FILTERED: {stats['yes']} | "
            f"REJECTED: {stats['no']} | "
            f"UNCLEAR: {stats['unclear']} | "
            f"FILTERING_ERROR: {stats['filtering_error']} | "
            f"CHECKED: {stats['checked']} | "
            f"TOTAL: {total}"
        )


async def process_file(
    xml_path: Path,
    client,
    stats,
    unclear_files,
    filtering_error_files,
    total,
    token_stats,
    stats_lock: asyncio.Lock,
    token_lock: asyncio.Lock,
    list_lock: asyncio.Lock,
    processed_log_lock: asyncio.Lock,
):
    try:
        xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[FILTERING ERROR] {xml_path.name} | read failed: {exc}")
        try:
            move_to_folder(xml_path, FILTERING_ERROR_DIR)
            print(f"[MOVED] {xml_path.name} -> {FILTERING_ERROR_DIR}")
        except Exception as move_exc:
            print(f"[ERROR] Could not move {xml_path.name} to filtering_error: {move_exc}")

        async with list_lock:
            filtering_error_files.append(xml_path.name)
        async with stats_lock:
            stats["filtering_error"] += 1
            stats["checked"] += 1
        await mark_processed(xml_path.name, processed_log_lock)
        await print_progress(stats, total, stats_lock)
        return

    title, abstract = extract_title_and_abstract(xml_text)

    if not title:
        title = xml_path.stem

    text_for_classification = abstract.strip()
    used_title_fallback = False

    if not text_for_classification and title.strip():
        text_for_classification = title.strip()
        used_title_fallback = True

    if not text_for_classification:
        print(f"[FILTERING ERROR] {xml_path.name} | no abstract and no usable title")
        try:
            move_to_folder(xml_path, FILTERING_ERROR_DIR)
            print(f"[MOVED] {xml_path.name} -> {FILTERING_ERROR_DIR}")
        except Exception as move_exc:
            print(f"[ERROR] Could not move {xml_path.name} to filtering_error: {move_exc}")

        async with list_lock:
            filtering_error_files.append(xml_path.name)
        async with stats_lock:
            stats["filtering_error"] += 1
            stats["checked"] += 1
        await mark_processed(xml_path.name, processed_log_lock)
        await print_progress(stats, total, stats_lock)
        return

    print(f"[START] {xml_path.name} | {title}")
    if used_title_fallback:
        print(f"[TITLE FALLBACK] {xml_path.name}")

    try:
        result = await classify_text(
            client,
            title,
            text_for_classification,
            token_stats,
            token_lock,
            xml_path.name,
        )
    except Exception as exc:
        print(f"[FILTERING ERROR] {xml_path.name} | LLM failed: {exc}")
        try:
            move_to_folder(xml_path, FILTERING_ERROR_DIR)
            print(f"[MOVED] {xml_path.name} -> {FILTERING_ERROR_DIR}")
        except Exception as move_exc:
            print(f"[ERROR] Could not move {xml_path.name} to filtering_error: {move_exc}")

        async with list_lock:
            filtering_error_files.append(xml_path.name)
        async with stats_lock:
            stats["filtering_error"] += 1
            stats["checked"] += 1
        await mark_processed(xml_path.name, processed_log_lock)
        await print_progress(stats, total, stats_lock)
        return

    if result is True:
        try:
            move_to_folder(xml_path, OUTPUT_DIR)
            print(f"[FILTERED] {xml_path.name} -> {OUTPUT_DIR} | {title}")
            async with stats_lock:
                stats["yes"] += 1
        except Exception as exc:
            print(f"[FILTERING ERROR] {xml_path.name} | move to filtered failed: {exc}")
            try:
                move_to_folder(xml_path, FILTERING_ERROR_DIR)
                print(f"[MOVED] {xml_path.name} -> {FILTERING_ERROR_DIR}")
            except Exception as move_exc:
                print(f"[ERROR] Could not move {xml_path.name} to filtering_error: {move_exc}")

            async with list_lock:
                filtering_error_files.append(xml_path.name)
            async with stats_lock:
                stats["filtering_error"] += 1

    elif result is False:
        try:
            move_to_folder(xml_path, REJECTED_DIR)
            print(f"[REJECTED] {xml_path.name} -> {REJECTED_DIR} | {title}")
            async with stats_lock:
                stats["no"] += 1
        except Exception as exc:
            print(f"[FILTERING ERROR] {xml_path.name} | move to rejected failed: {exc}")
            try:
                move_to_folder(xml_path, FILTERING_ERROR_DIR)
                print(f"[MOVED] {xml_path.name} -> {FILTERING_ERROR_DIR}")
            except Exception as move_exc:
                print(f"[ERROR] Could not move {xml_path.name} to filtering_error: {move_exc}")

            async with list_lock:
                filtering_error_files.append(xml_path.name)
            async with stats_lock:
                stats["filtering_error"] += 1

    else:
        print(f"[FILTERING ERROR] {xml_path.name} | could not parse YES/NO after retries")
        try:
            move_to_folder(xml_path, FILTERING_ERROR_DIR)
            print(f"[MOVED] {xml_path.name} -> {FILTERING_ERROR_DIR}")
        except Exception as move_exc:
            print(f"[ERROR] Could not move {xml_path.name} to filtering_error: {move_exc}")

        async with list_lock:
            unclear_files.append(xml_path.name)
            filtering_error_files.append(xml_path.name)
        async with stats_lock:
            stats["unclear"] += 1
            stats["filtering_error"] += 1

    async with stats_lock:
        stats["checked"] += 1

    await mark_processed(xml_path.name, processed_log_lock)
    await print_progress(stats, total, stats_lock)


async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    client,
    stats,
    unclear_files,
    filtering_error_files,
    total,
    token_stats,
    stats_lock: asyncio.Lock,
    token_lock: asyncio.Lock,
    list_lock: asyncio.Lock,
    processed_log_lock: asyncio.Lock,
):
    while True:
        xml_path = await queue.get()
        if xml_path is None:
            queue.task_done()
            return

        try:
            await process_file(
                xml_path,
                client,
                stats,
                unclear_files,
                filtering_error_files,
                total,
                token_stats,
                stats_lock,
                token_lock,
                list_lock,
                processed_log_lock,
            )
        finally:
            queue.task_done()


async def main():
    start_time = time.time()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"logs/elsevier_filter_{timestamp}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_path)

    print("===== FULL SCRIPT (START) =====\n")
    with open(__file__, "r", encoding="utf-8") as f:
        print(f.read())
    print("\n===== END SCRIPT =====\n")

    print(f"MODEL: {MODEL_NAME}")
    print("REASONING_EFFORT: low")
    print(f"MAX_COMPLETION_TOKENS: {MAX_COMPLETION_TOKENS}")
    print(f"CONCURRENCY: {CONCURRENCY}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    FILTERING_ERROR_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return

    already_processed = load_processed_files()
    all_input_files = sorted(INPUT_DIR.glob("*.xml"))
    total_files = len(all_input_files)

    xml_files = [
        f for f in all_input_files
        if f.name not in already_processed
    ][:MAX_FILES]

    print(f"Already processed files in log: {len(already_processed)}")
    print(f"Files remaining for this run: {len(xml_files)}")
    print(f"Total files currently in input folder: {total_files}\n")

    if not xml_files:
        print("[INFO] No files left to process.")
        return

    client = get_client()

    stats = {
        "yes": 0,
        "no": 0,
        "unclear": 0,
        "filtering_error": 0,
        "checked": 0,
    }
    unclear_files = []
    filtering_error_files = []
    token_stats = {
        "input": 0,
        "output": 0,
    }

    stats_lock = asyncio.Lock()
    token_lock = asyncio.Lock()
    list_lock = asyncio.Lock()
    processed_log_lock = asyncio.Lock()

    queue: asyncio.Queue = asyncio.Queue()

    for xml_file in xml_files:
        await queue.put(xml_file)

    workers = []
    for i in range(CONCURRENCY):
        workers.append(
            asyncio.create_task(
                worker(
                    i,
                    queue,
                    client,
                    stats,
                    unclear_files,
                    filtering_error_files,
                    total_files,
                    token_stats,
                    stats_lock,
                    token_lock,
                    list_lock,
                    processed_log_lock,
                )
            )
        )

    await queue.join()

    for _ in workers:
        await queue.put(None)

    await asyncio.gather(*workers)

    print("\nDone.")
    print(f"Runtime: {time.time() - start_time:.2f}s")

    print("\nFinal counts:")
    print(f"Filtered: {stats['yes']}")
    print(f"Rejected: {stats['no']}")
    print(f"Unclear: {stats['unclear']}")
    print(f"Filtering error: {stats['filtering_error']}")
    print(f"Checked: {stats['checked']}")
    print(f"Total input at start: {total_files}")

    print("\nToken usage:")
    print(f"Input tokens: {token_stats['input']}")
    print(f"Output tokens: {token_stats['output']}")
    print(f"Total tokens: {token_stats['input'] + token_stats['output']}")

    if unclear_files:
        print("\nUNCLEAR FILES:")
        for f in unclear_files:
            print(f"- {f}")

    if filtering_error_files:
        print("\nFILES MOVED TO FILTERING_ERROR:")
        for f in filtering_error_files:
            print(f"- {f}")


if __name__ == "__main__":
    asyncio.run(main())