#!/usr/bin/env python3

from __future__ import annotations

import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

INPUT_DIR = Path("AGO_REF_PDF/E_REF_PDF")
OUTPUT_DIR = Path("AGO_REF_PDF/output_txt_llm")
MAX_FILES = 1

MODEL_NAME = "Qwen3.5-122B-A10B-FP8"
RENDER_DPI = 220
MAX_COMPLETION_TOKENS = 4096
SLEEP_BETWEEN_REQUESTS = 0.2


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def render_pdf_page_to_png_bytes(page: fitz.Page, dpi: int = 220) -> bytes:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def transcribe_page_with_llm(client: OpenAI, image_bytes: bytes, pdf_name: str, page_num: int) -> str:
    data_url = image_bytes_to_data_url(image_bytes)

    system_prompt = (
        "You are a precise document transcription engine. "
        "Your task is to transcribe all visible text from a document page image. "
        "Do not summarize. Do not explain. Do not infer missing text. Do not add commentary. "
        "Return only the transcribed text."
    )

    user_prompt = (
        f"Transcribe this PDF page exactly and as completely as possible.\n\n"
        f"Rules:\n"
        f"1. Extract all readable text on the page.\n"
        f"2. Preserve natural reading order.\n"
        f"3. Preserve headings, bullet points, numbered references, and short line breaks where useful.\n"
        f"4. If the page contains a table, reproduce it as a fixed-width plain-text table with aligned columns.\n"
        f"5.Do not list column headers separately from the rows.\n"
        f"6.Do not convert a table into bullets.\n"
        f"7.Keep row-to-column relationships intact.\n"
        f"8. Use spaces to align columns.\n"
        f"9. If a row wraps, indent continuation lines under the first column.\n"
        f"10. If some text is unclear, mark it as [unclear] rather than guessing.\n"
        f"11. Do not summarize or omit repeated-looking content.\n"
        f"12. Do not mention image quality or give meta-comments.\n\n"
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

    return (response.choices[0].message.content or "").strip()


def extract_pdf_with_vision_llm(pdf_path: Path, client: OpenAI, dpi: int = 220) -> str:
    doc = fitz.open(pdf_path)
    page_texts = []

    try:
        total_pages = len(doc)

        for page_index in range(total_pages):
            page_num = page_index + 1
            print(f"  Page {page_num}/{total_pages}")

            page = doc.load_page(page_index)
            image_bytes = render_pdf_page_to_png_bytes(page, dpi=dpi)

            page_text = transcribe_page_with_llm(
                client=client,
                image_bytes=image_bytes,
                pdf_name=pdf_path.name,
                page_num=page_num,
            )

            block = [
                f"===== PAGE {page_num} START =====",
                page_text,
                f"===== PAGE {page_num} END =====",
            ]
            page_texts.append("\n".join(block))

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    finally:
        doc.close()

    return "\n\n".join(page_texts)


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

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))[:MAX_FILES]
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        try:
            full_text = extract_pdf_with_vision_llm(pdf_file, client, dpi=RENDER_DPI)
        except Exception as e:
            print(f"Failed on {pdf_file.name}: {e}")
            continue

        output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Saved: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())