#!/usr/bin/env python3

from __future__ import annotations

import base64
import datetime
import inspect
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

INPUT_DIR = Path("artifacts/AGO/downloaded")
OUTPUT_DIR = Path("artifacts/AGO/processed")
MAX_FILES = 25

MODEL_NAME = "Qwen3.5-397B-A17B-FP8"
RENDER_DPI = 220
MAX_COMPLETION_TOKENS = 4096
SLEEP_BETWEEN_REQUESTS = 0.2

# How many pages of a single PDF to transcribe concurrently. Kept small on
# purpose so we don't hammer the API.
MAX_WORKERS = 4


def create_log_file() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"vlm_md_extraction_{timestamp}.log"


def write_log_header(log_path: Path) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== SCRIPT SNAPSHOT ===\n\n")
        try:
            f.write(inspect.getsource(sys.modules[__name__]))
        except Exception:
            f.write("[Could not capture script source]\n")
        f.write("\n\n=== RUN LOG ===\n\n")


def append_log(log_path: Path, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message)


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def render_pdf_page_to_png_bytes(page: fitz.Page, dpi: int = 220) -> bytes:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def transcribe_page_with_llm(
    client: OpenAI,
    image_bytes: bytes,
    pdf_name: str,
    page_num: int,
) -> tuple[str, dict[str, int]]:
    data_url = image_bytes_to_data_url(image_bytes)

    system_prompt = (
        "You are a precise document transcription engine. "
        "Your task is to transcribe all visible text from a document page image into Markdown. "
        "Do not summarize. Do not explain. Do not infer missing text. Do not add commentary. "
        "Return only the transcribed Markdown."
    )

    user_prompt = (
        f"Transcribe this PDF page exactly and as completely as possible into Markdown.\n\n"
        f"Rules:\n"
        f"1. Extract all readable text on the page.\n"
        f"2. Preserve natural reading order.\n"
        f"3. Preserve headings using Markdown headings where appropriate.\n"
        f"4. Preserve bullet points and numbered lists using Markdown list syntax.\n"
        f"5. Preserve short line breaks where useful.\n"
        f"6. If the page contains a simple table, reproduce it as a Markdown table.\n"
        f"7. If the table is complex or alignment would be lost in a Markdown table, reproduce it inside a fenced code block as a fixed-width plain-text table with aligned columns.\n"
        f"8. Do not list column headers separately from the rows.\n"
        f"9. Do not convert a table into bullets.\n"
        f"10. Keep row-to-column relationships intact.\n"
        f"11. Use spaces to align columns inside code-block tables.\n"
        f"12. If a row wraps, indent continuation lines under the first column when using a fixed-width table.\n"
        f"13. If some text is unclear, mark it as [unclear] rather than guessing.\n"
        f"14. Do not summarize or omit repeated-looking content.\n"
        f"15. Do not mention image quality or give meta-comments.\n"
        f"16. Return only Markdown content for this page.\n\n"
        f"Document: {pdf_name}\n"
        f"Page: {page_num}\n"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    content = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }

    return content, usage_dict


def extract_pdf_with_vision_llm(
    pdf_path: Path,
    client: OpenAI,
    log_path: Path,
    dpi: int = 220,
    max_workers: int = MAX_WORKERS,
) -> tuple[str, dict[str, int]]:
    # Render all pages first (CPU-bound, fast) so the worker pool only
    # does network-bound VLM calls.
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        rendered: list[bytes] = []
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            rendered.append(render_pdf_page_to_png_bytes(page, dpi=dpi))
    finally:
        doc.close()

    print(f"  {total_pages} pages, parallel workers={max_workers}", flush=True)

    results: dict[int, tuple[str, dict[str, int]]] = {}

    def transcribe_one(page_index: int) -> tuple[int, str, dict[str, int]]:
        page_num = page_index + 1
        page_md, usage = transcribe_page_with_llm(
            client=client,
            image_bytes=rendered[page_index],
            pdf_name=pdf_path.name,
            page_num=page_num,
        )
        return page_index, page_md, usage

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(transcribe_one, i) for i in range(total_pages)]
        for fut in as_completed(futures):
            page_index, page_md, usage = fut.result()
            results[page_index] = (page_md, usage)
            completed += 1
            page_num = page_index + 1
            print(
                f"  Page {page_num} done ({completed}/{total_pages})",
                flush=True,
            )
            append_log(
                log_path,
                f"  Page {page_num}/{total_pages}\n"
                f"    Input tokens: {usage['input_tokens']}\n"
                f"    Output tokens: {usage['output_tokens']}\n"
                f"    Total tokens: {usage['total_tokens']}\n\n",
            )

    # Re-assemble in page order regardless of completion order.
    total_usage = {"input": 0, "output": 0, "total": 0}
    page_blocks: list[str] = []
    for page_index in range(total_pages):
        page_md, usage = results[page_index]
        total_usage["input"] += usage["input_tokens"]
        total_usage["output"] += usage["output_tokens"]
        total_usage["total"] += usage["total_tokens"]
        page_num = page_index + 1
        block = [
            f"<!-- PAGE {page_num} START -->",
            "",
            f"## Page {page_num}",
            "",
            page_md,
            "",
            f"<!-- PAGE {page_num} END -->",
        ]
        page_blocks.append("\n".join(block))

    # Light delay before moving on to the next PDF.
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    document_header = [
        f"# {pdf_path.stem}",
        "",
        f"Source PDF: `{pdf_path.name}`",
        "",
    ]

    full_markdown = "\n".join(document_header) + "\n\n" + "\n\n".join(page_blocks)
    return full_markdown, total_usage


def main() -> int:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        print("Error: VIRTUAL_API_KEY is not set.")
        return 1

    if not base_url:
        print("Error: BASE_URL is not set.")
        return 1

    if not INPUT_DIR.exists():
        print(f"Error: input directory does not exist: {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = create_log_file()
    write_log_header(log_path)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))[:MAX_FILES]
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        append_log(log_path, f"No PDF files found in {INPUT_DIR}\n")
        return 1

    # Resume logic:
    #   - Skip PDFs whose markdown already exists in OUTPUT_DIR.
    #   - The most-recently-written markdown is treated as potentially
    #     incomplete (the script may have been killed mid-write), so we
    #     delete it and re-process its PDF from scratch.
    existing_mds = sorted(OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime)
    if existing_mds:
        last_md = existing_mds[-1]
        print(
            f"Resume: removing last md to redo from scratch (kill-safety): "
            f"{last_md.name}"
        )
        append_log(
            log_path,
            f"Resume: removing last md {last_md.name} (will reprocess)\n",
        )
        last_md.unlink()
        existing_mds = existing_mds[:-1]

    completed_stems = {p.stem for p in existing_mds}
    skipped = [p for p in pdf_files if p.stem in completed_stems]
    pdf_files = [p for p in pdf_files if p.stem not in completed_stems]

    print(f"Resume: {len(skipped)} already done, {len(pdf_files)} to process")
    append_log(
        log_path,
        f"Resume: {len(skipped)} skipped, {len(pdf_files)} to process\n",
    )
    if not pdf_files:
        print("Nothing left to process.")
        append_log(log_path, "Nothing left to process.\n")
        return 0

    client = OpenAI(api_key=api_key, base_url=base_url)
    global_usage = {"input": 0, "output": 0, "total": 0}

    append_log(log_path, f"Input directory: {INPUT_DIR}\n")
    append_log(log_path, f"Output directory: {OUTPUT_DIR}\n")
    append_log(log_path, f"Model: {MODEL_NAME}\n")
    append_log(log_path, f"Render DPI: {RENDER_DPI}\n")
    append_log(log_path, f"Max files: {MAX_FILES}\n")
    append_log(log_path, f"Parallel workers per PDF: {MAX_WORKERS}\n\n")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        append_log(log_path, f"Processing: {pdf_file.name}\n")

        try:
            full_markdown, usage = extract_pdf_with_vision_llm(
                pdf_file,
                client,
                log_path,
                dpi=RENDER_DPI,
            )
        except Exception as e:
            print(f"Failed on {pdf_file.name}: {e}")
            append_log(log_path, f"Failed on {pdf_file.name}: {e}\n\n")
            continue

        output_file = OUTPUT_DIR / f"{pdf_file.stem}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        print(f"Saved: {output_file}")
        append_log(
            log_path,
            f"Saved: {output_file}\n"
            f"File input tokens: {usage['input']}\n"
            f"File output tokens: {usage['output']}\n"
            f"File total tokens: {usage['total']}\n\n",
        )

        global_usage["input"] += usage["input"]
        global_usage["output"] += usage["output"]
        global_usage["total"] += usage["total"]

    append_log(
        log_path,
        "=== SUMMARY ===\n"
        f"Total input tokens: {global_usage['input']}\n"
        f"Total output tokens: {global_usage['output']}\n"
        f"Total tokens: {global_usage['total']}\n",
    )

    print(f"Log saved: {log_path}")
    print(f"Total input tokens: {global_usage['input']}")
    print(f"Total output tokens: {global_usage['output']}")
    print(f"Total tokens: {global_usage['total']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())