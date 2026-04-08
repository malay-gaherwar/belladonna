#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("artifacts/epmc_fulltext/processed")
OUTPUT_DIR = Path("artifacts/epmc_fulltext/factoids")
FAILED_DIR = Path("artifacts/epmc_fulltext/factoids_failed")
LOG_DIR = Path("logs")

MODEL_NAME = os.getenv("MODEL_NAME", "GPT-OSS-120B")
MAX_COMPLETION_TOKENS = 4096

CONCURRENCY = 20
MAX_RETRIES = 4
REQUEST_TIMEOUT_SECONDS = 300
PROGRESS_EVERY = 100
ENABLE_THINKING = False

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"

# Full-text handling:
# We do NOT truncate the document globally.
# We chunk only when needed so the whole paper can still be processed.
MODEL_MAX_INPUT_CHARS = 110000
CHUNK_SIZE_CHARS = 80000
CHUNK_OVERLAP_CHARS = 5000

# Optional test limit
MAX_FILES: Optional[int] = None


# ============================================================
# LOGGING
# ============================================================

class Tee:
    def __init__(self, filepath: Path) -> None:
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


def init_logging() -> Tee:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"factoids_epmc_{time.strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 100)
    print("EPMC FACTOID EXTRACTION LOG")
    print("=" * 100)
    print(f"Log file: {log_path}")
    print(f"Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("=" * 100)
    print("SCRIPT SOURCE")
    print("=" * 100)
    try:
        script_path = Path(__file__).resolve()
        print(script_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not print script source: {e}")

    print()
    print("=" * 100)
    print("RUN OUTPUT")
    print("=" * 100)
    return tee


# ============================================================
# CLIENT
# ============================================================

def load_client() -> AsyncOpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY is not set")
    if not base_url:
        raise RuntimeError("BASE_URL is not set")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


# ============================================================
# UTIL
# ============================================================

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())

def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)

def output_path_for(input_file: Path) -> Path:
    return OUTPUT_DIR / f"{input_file.stem}_factoids.json"

def fail_path_for(input_file: Path) -> Path:
    return FAILED_DIR / f"{input_file.stem}_failed.json"

def short_error_message(exc: Exception) -> str:
    msg = str(exc).strip()
    if not msg:
        msg = exc.__class__.__name__
    return f"{exc.__class__.__name__}: {msg}"

def extract_blocks(text: str) -> List[str]:
    pattern = re.compile(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        re.DOTALL,
    )
    return [b.strip() for b in pattern.findall(text)]

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = normalize(item).casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(normalize(item))
    return out

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text or ""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks


# ============================================================
# PROMPTS
# ============================================================

def build_single_pass_prompts(document_title: str, full_text: str) -> Tuple[str, str]:
    system = (
        "You extract atomic, self-contained clinical factoids from breast-cancer-related full text.\n"
        "Each factoid must stand alone without external context.\n"
        "Use only information explicitly stated in the provided text.\n"
        "Do not invent, infer, or generalize beyond the text.\n"
        "Avoid duplicates.\n"
        "Output ONLY tagged factoids.\n"
    )

    user = (
        f"Document title: {document_title}\n\n"
        f"Task:\n"
        f"Extract atomic, self-contained factoids from the full text below.\n"
        f"Each factoid must explicitly name the disease, drug, biomarker, patient population, species, "
        f"or study context so the factoid is understandable on its own.\n"
        f"If there are no useful factoids, output nothing.\n\n"
        f"Required format:\n"
        f"{FACTOID_START}\n"
        f"<factoid>\n"
        f"{FACTOID_END}\n\n"
        f"Full text:\n{full_text}"
    )

    return system, user


def build_chunk_prompts(
    document_title: str,
    chunk_text: str,
    chunk_index: int,
    chunk_total: int,
) -> Tuple[str, str]:
    system = (
        "You extract atomic, self-contained clinical factoids from breast-cancer-related text chunks.\n"
        "Each factoid must stand alone without external context.\n"
        "Use only information explicitly stated in the provided chunk.\n"
        "Do not invent, infer, or generalize beyond the text.\n"
        "Avoid duplicates within this chunk.\n"
        "Output ONLY tagged factoids.\n"
    )

    user = (
        f"Document title: {document_title}\n"
        f"Chunk: {chunk_index} of {chunk_total}\n\n"
        f"Task:\n"
        f"Extract atomic, self-contained factoids from this chunk.\n"
        f"Each factoid must explicitly name the disease, drug, biomarker, patient population, species, "
        f"or study context so the factoid is understandable on its own.\n"
        f"Do not refer to 'this study' or 'the paper' without naming the context.\n"
        f"If there are no useful factoids, output nothing.\n\n"
        f"Required format:\n"
        f"{FACTOID_START}\n"
        f"<factoid>\n"
        f"{FACTOID_END}\n\n"
        f"Chunk text:\n{chunk_text}"
    )

    return system, user


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_context_length_error(exc: Exception) -> bool:
    msg = str(exc)
    lower = msg.lower()
    return (
        "maximum context length" in lower
        or "input length" in lower and "exceeds" in lower
        or "context length" in lower
    )

def is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    name = exc.__class__.__name__.lower()

    if is_context_length_error(exc):
        return False

    retry_markers = [
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "serviceunavailable",
        "service unavailable",
        "503",
        "502",
        "500",
        "connection",
        "api connection",
        "temporarily unavailable",
        "overloaded",
        "server error",
        "internal error",
    ]

    if any(marker in msg for marker in retry_markers):
        return True

    if "timeouterror" in name:
        return True

    return False


# ============================================================
# MODEL CALL
# ============================================================

async def call_model(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
) -> str:
    async def _do_request() -> str:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
        )
        return resp.choices[0].message.content or ""

    return await asyncio.wait_for(_do_request(), timeout=REQUEST_TIMEOUT_SECONDS)


# ============================================================
# FACTOID EXTRACTION
# ============================================================

async def extract_factoids_single_pass(
    client: AsyncOpenAI,
    document_title: str,
    full_text: str,
) -> List[str]:
    system, user = build_single_pass_prompts(document_title, full_text)
    content = await call_model(client, system, user)
    return dedupe_preserve_order(extract_blocks(content))


async def extract_factoids_chunked(
    client: AsyncOpenAI,
    document_title: str,
    full_text: str,
) -> List[str]:
    chunks = split_text_into_chunks(
        full_text,
        chunk_size=CHUNK_SIZE_CHARS,
        overlap=CHUNK_OVERLAP_CHARS,
    )

    all_factoids: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        system, user = build_chunk_prompts(
            document_title=document_title,
            chunk_text=chunk,
            chunk_index=i,
            chunk_total=len(chunks),
        )
        content = await call_model(client, system, user)
        chunk_factoids = extract_blocks(content)
        all_factoids.extend(chunk_factoids)

    return dedupe_preserve_order(all_factoids)


async def extract_factoids(
    client: AsyncOpenAI,
    document_title: str,
    full_text: str,
) -> List[str]:
    # Try single pass if text is probably safe enough.
    # If it is too large or still triggers a context error, fall back to chunking.
    if len(full_text) <= MODEL_MAX_INPUT_CHARS:
        try:
            return await extract_factoids_single_pass(client, document_title, full_text)
        except Exception as e:
            if is_context_length_error(e):
                print(f"[{now_ts()}] [CHUNK-FALLBACK] {document_title} -> context-length hit in single pass, switching to chunking")
                return await extract_factoids_chunked(client, document_title, full_text)
            raise

    print(f"[{now_ts()}] [CHUNK] {document_title} -> full_text too large for single pass ({len(full_text)} chars), chunking")
    return await extract_factoids_chunked(client, document_title, full_text)


# ============================================================
# STATS
# ============================================================

class Stats:
    def __init__(self, total: int) -> None:
        self.total = total
        self.discovered = total
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
        self.skipped = 0
        self.empty_factoids = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()

    async def record_skip(self, filename: str) -> None:
        async with self.lock:
            self.skipped += 1
            self.processed += 1
            print(f"[{now_ts()}] [{self.processed}/{self.total}] [SKIP] {filename} -> already processed")
            self._maybe_print()

    async def record_success(self, filename: str, factoid_count: int, worker_id: int) -> None:
        async with self.lock:
            self.succeeded += 1
            self.processed += 1
            if factoid_count == 0:
                self.empty_factoids += 1
            print(
                f"[{now_ts()}] [{self.processed}/{self.total}] [OK] "
                f"worker={worker_id} {filename} -> {factoid_count} factoids"
            )
            self._maybe_print()

    async def record_failure(self, filename: str, err: str, worker_id: int) -> None:
        async with self.lock:
            self.failed += 1
            self.processed += 1
            print(
                f"[{now_ts()}] [{self.processed}/{self.total}] [ERROR] "
                f"worker={worker_id} {filename} -> {err}"
            )
            self._maybe_print()

    def _maybe_print(self) -> None:
        if self.processed % PROGRESS_EVERY == 0 or self.processed == self.total:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0.0
            remaining = self.total - self.processed
            eta_sec = remaining / rate if rate > 0 else 0.0
            print(
                f"[{now_ts()}] PROGRESS "
                f"processed={self.processed}/{self.total} | "
                f"ok={self.succeeded} | failed={self.failed} | skipped={self.skipped} | "
                f"empty={self.empty_factoids} | rate={rate:.2f} files/sec | eta={eta_sec/60:.1f} min"
            )


# ============================================================
# PROCESS FILE
# ============================================================

async def process_one_file(
    client: AsyncOpenAI,
    input_file: Path,
) -> Tuple[bool, str, int]:
    data = safe_read_json(input_file)

    metadata = data.get("metadata")
    full_text = data.get("full_text")

    if not isinstance(metadata, dict):
        raise ValueError("Missing or invalid 'metadata'")
    if not isinstance(full_text, str):
        raise ValueError("Missing or invalid 'full_text'")

    document_title = metadata.get("TITLE") or input_file.stem

    factoids = await extract_factoids(client, document_title, full_text)

    output = {
        "metadata": metadata,
        "factoids": [
            {"id": i + 1, "factoid_text": factoid}
            for i, factoid in enumerate(factoids)
        ],
    }

    safe_write_json(output_path_for(input_file), output)
    return True, input_file.name, len(factoids)


async def process_with_retries(
    client: AsyncOpenAI,
    input_file: Path,
) -> Tuple[bool, str, int, Optional[str]]:
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ok, fname, count = await process_one_file(client, input_file)
            return ok, fname, count, None

        except Exception as e:
            last_error = short_error_message(e)

            # Never retry hard context-length failures
            if is_context_length_error(e):
                fail_payload = {
                    "input_file": str(input_file),
                    "error": last_error,
                    "timestamp": now_ts(),
                    "retryable": False,
                    "reason": "context_length_exceeded_after_chunking_or_single_pass",
                }
                safe_write_json(fail_path_for(input_file), fail_payload)
                return False, input_file.name, 0, last_error

            # Retry only temporary failures
            if is_retryable_error(e) and attempt < MAX_RETRIES:
                backoff = min(2 ** (attempt - 1), 30)
                print(
                    f"[{now_ts()}] [RETRY] {input_file.name} | "
                    f"attempt={attempt}/{MAX_RETRIES} | error={last_error} | sleeping={backoff}s"
                )
                await asyncio.sleep(backoff)
                continue

            fail_payload = {
                "input_file": str(input_file),
                "error": last_error,
                "timestamp": now_ts(),
                "retryable": False if not is_retryable_error(e) else True,
            }
            safe_write_json(fail_path_for(input_file), fail_payload)
            return False, input_file.name, 0, last_error

    return False, input_file.name, 0, last_error


# ============================================================
# WORKER
# ============================================================

async def worker(
    worker_id: int,
    queue: asyncio.Queue[Path],
    client: AsyncOpenAI,
    stats: Stats,
) -> None:
    while True:
        try:
            input_file = await queue.get()
        except asyncio.CancelledError:
            return

        try:
            if output_path_for(input_file).exists():
                await stats.record_skip(input_file.name)
                continue

            ok, fname, count, err = await process_with_retries(client, input_file)

            if ok:
                await stats.record_success(fname, count, worker_id)
            else:
                await stats.record_failure(fname, err or "Unknown error", worker_id)

        finally:
            queue.task_done()


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    ensure_dirs()
    client = load_client()

    files = sorted(INPUT_DIR.glob("*.json"))
    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    if not files:
        print(f"No input files found in {INPUT_DIR}")
        return

    print(f"Input dir:          {INPUT_DIR}")
    print(f"Output dir:         {OUTPUT_DIR}")
    print(f"Failed dir:         {FAILED_DIR}")
    print(f"Log dir:            {LOG_DIR}")
    print(f"Model:              {MODEL_NAME}")
    print(f"Concurrency:        {CONCURRENCY}")
    print(f"Max retries:        {MAX_RETRIES}")
    print(f"Request timeout:    {REQUEST_TIMEOUT_SECONDS}s")
    print(f"Max files:          {MAX_FILES if MAX_FILES is not None else 'ALL'}")
    print(f"Discovered files:   {len(files)}")
    print(f"Single-pass limit:  {MODEL_MAX_INPUT_CHARS} chars")
    print(f"Chunk size:         {CHUNK_SIZE_CHARS} chars")
    print(f"Chunk overlap:      {CHUNK_OVERLAP_CHARS} chars")
    print()

    stats = Stats(total=len(files))
    queue: asyncio.Queue[Path] = asyncio.Queue()

    for f in files:
        queue.put_nowait(f)

    workers = [
        asyncio.create_task(worker(i + 1, queue, client, stats))
        for i in range(CONCURRENCY)
    ]

    try:
        await queue.join()
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


if __name__ == "__main__":
    tee: Optional[Tee] = None
    try:
        tee = init_logging()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    finally:
        if tee is not None:
            tee.flush()
            tee.close()