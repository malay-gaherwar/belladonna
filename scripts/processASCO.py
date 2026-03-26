#!/usr/bin/env python3

from __future__ import annotations

import base64
import datetime
import inspect
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

INPUT_DIR = Path("ASCO")
OUTPUT_DIR = Path("ASCO/output_md_llm_asco")
MAX_FILES = 50

MODEL_NAME = "Qwen3.5-397B-A17B-FP8"
RENDER_DPI = 260
MAX_COMPLETION_TOKENS = 4096
SLEEP_BETWEEN_REQUESTS = 0.2

MARGIN_PATTERNS = [
    r"^Downloaded from .*",
    r"^Copyright .*",
    r"^Journal of Clinical Oncology.*",
    r"^J Clin Oncol.*",
    r"^Volume \d+.*",
    r"^Issue \d+.*",
    r"^DOI[: ].*",
    r"^© .*",
    r"^This information is current as of .*",
    r"^\d{4,5}\s+© .*",
]

TEXT_HEADING_PATTERNS = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "recommendations",
    "guideline questions",
    "guideline implementation",
    "guideline disclaimer",
    "health disparities",
    "multiple chronic conditions",
    "cost implications",
    "additional resources",
    "related asco guidelines",
    "affiliations",
    "editor's note",
    "author contributions",
    "corresponding author",
    "equal contribution",
    "acknowledgment",
    "references",
    "appendix",
    "the bottom line",
]


def create_log_file() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"vlm_md_extraction_asco_{timestamp}.log"


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


def normalize_unicode_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u00ad": "",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2013": "–",
        "\u2014": "—",
        "\u2212": "-",
        "\xa0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def render_pdf_page_variants(page: fitz.Page, dpi: int = RENDER_DPI) -> dict[str, bytes]:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    full_pix = page.get_pixmap(matrix=matrix, alpha=False)
    full_png = full_pix.tobytes("png")

    rect = page.rect
    crop = fitz.Rect(
        rect.x0 + rect.width * 0.06,
        rect.y0 + rect.height * 0.04,
        rect.x1 - rect.width * 0.04,
        rect.y1 - rect.height * 0.05,
    )
    crop_pix = page.get_pixmap(matrix=matrix, clip=crop, alpha=False)
    crop_png = crop_pix.tobytes("png")

    return {"full": full_png, "main": crop_png}


def build_page_prompt(
    pdf_name: str,
    page_num: int,
    previous_table_context: str | None = None,
) -> str:
    prompt = (
        f"Transcribe this ASCO / medical-journal PDF page into Markdown as faithfully as possible.\n\n"
        f"Document: {pdf_name}\n"
        f"Page: {page_num}\n\n"
        f"Rules:\n"
        f"1. Extract all readable text in natural reading order.\n"
        f"2. Do not summarize.\n"
        f"3. Do not explain.\n"
        f"4. Do not infer missing words.\n"
        f"5. If text is unclear, write [unclear].\n"
        f"6. Preserve medical terminology, biomarkers, drug names, statistics, symbols, and confidence intervals exactly when readable.\n"
        f"7. Preserve headings using normal Markdown headings where appropriate.\n"
        f"8. Preserve normal body text as paragraphs.\n"
        f"9. Ignore figures, flowcharts, diagrams, and non-table graphical algorithms. Do not transcribe figure box contents. If a figure caption/title is plainly visible, you may keep only the caption/title line.\n"
        f"10. If the page contains a table, keep it as a table-like transcription exactly as in AGO style.\n"
        f"11. For simple tables, use a normal Markdown table.\n"
        f"12. For complex or wide tables, use a fenced code block with aligned plain-text columns.\n"
        f"13. Do NOT use LaTeX or any \\begin{{tabular}} format.\n"
        f"14. Do NOT convert table content into bullets or prose.\n"
        f"15. Keep row-to-column relationships intact.\n"
        f"16. Preserve labels such as '(continued)' or 'continued on following page' when visible.\n"
        f"17. If header/footer or vertical margin text appears, move it to labeled blockquote sections at the end.\n"
        f"18. Return only Markdown for this page.\n"
    )

    if previous_table_context:
        prompt += (
            f"\nPrevious table context is provided below only to help maintain column continuity if this page continues a table.\n"
            f"Do not repeat rows that are not visible on this page.\n\n"
            f"{previous_table_context[:2500]}\n"
        )

    return prompt


def split_margin_lines(md: str) -> str:
    body_lines: list[str] = []
    margin_lines: list[str] = []

    for line in md.splitlines():
        stripped = line.strip()
        if stripped and any(re.match(p, stripped, flags=re.IGNORECASE) for p in MARGIN_PATTERNS):
            margin_lines.append(stripped)
        else:
            body_lines.append(line)

    out = "\n".join(body_lines).rstrip()

    if margin_lines:
        if out:
            out += "\n\n"
        out += "> Page margin/header/footer text:\n"
        out += "\n".join(f"> {line}" for line in margin_lines)

    return out.strip()


def clean_llm_markdown(md: str) -> str:
    md = normalize_unicode_text(md)

    md = re.sub(r"^\s*```markdown\s*\n", "", md, flags=re.IGNORECASE)
    md = re.sub(r"\n```\s*$", "", md)

    md = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        "[unclear table formatting removed]",
        md,
        flags=re.DOTALL,
    )

    md = re.sub(
        r"```mermaid.*?```",
        "[figure content omitted]",
        md,
        flags=re.DOTALL,
    )

    md = re.sub(r"^\*\*Footer text\*\*:?", "> Page margin/header/footer text:", md, flags=re.MULTILINE)
    md = re.sub(r"^\*\*Page margin/header/footer text\*\*:?", "> Page margin/header/footer text:", md, flags=re.MULTILINE)
    md = re.sub(r"^\*\*Vertical margin text\*\*:?", "> Vertical margin text:", md, flags=re.MULTILINE)

    md = md.replace("[blank]", "[unclear]")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    return split_margin_lines(md)


def repair_broken_prose_lines(lines: list[str]) -> list[str]:
    repaired: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            repaired.append("")
            i += 1
            continue

        current = line.rstrip()
        while i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()

            if not nxt:
                break

            current_stripped = current.strip()
            next_stripped = nxt.strip()

            if (
                current_stripped
                and next_stripped
                and current_stripped.endswith("-")
                and re.match(r"^[a-z]", next_stripped)
                and not re.search(r"\b[A-Z]{2,}-$", current_stripped)
            ):
                current = current[:-1] + next_stripped
                i += 1
                continue

            if (
                current_stripped
                and next_stripped
                and not current_stripped.endswith((".", ":", ";", "?", "!", "|"))
                and not current_stripped.startswith("#")
                and not current_stripped.startswith(">")
                and not re.match(r"^\s*[-*]\s+", next_stripped)
                and not re.match(r"^\s*\d+[.)]\s+", next_stripped)
                and not re.match(r"^[A-Z0-9][A-Z0-9 \-/,&()]{3,}$", current_stripped)
                and not re.match(r"^[A-Z0-9][A-Z0-9 \-/,&()]{3,}$", next_stripped)
                and len(current_stripped) > 35
                and len(next_stripped) > 20
            ):
                current = current + " " + next_stripped
                i += 1
                continue

            break

        repaired.append(current)
        i += 1

    return repaired


def convert_native_text_to_md(text: str, page_num: int) -> str:
    text = normalize_unicode_text(text)
    lines = [line.rstrip() for line in text.splitlines()]
    lines = repair_broken_prose_lines(lines)

    cleaned: list[str] = []
    previous_blank = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue

        previous_blank = False

        low = stripped.lower()
        if len(stripped) < 140 and any(low.startswith(p) for p in TEXT_HEADING_PATTERNS):
            cleaned.append(f"### {stripped}")
        elif len(stripped) < 120 and stripped.isupper() and len(stripped.split()) <= 10:
            cleaned.append(f"### {stripped.title()}")
        else:
            cleaned.append(stripped)

    md = "\n".join(cleaned).strip()
    md = split_margin_lines(md)

    if not md:
        md = f"[No extractable native text on page {page_num}]"
    return md


def native_text_quality_score(native_text: str) -> dict[str, float]:
    text = native_text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    total_chars = max(len(text), 1)

    ligatures = text.count("ﬁ") + text.count("ﬂ")
    soft_artifacts = text.count("\u00ad")
    hyphen_ends = sum(1 for line in lines if line.endswith("-"))
    short_lines = sum(1 for line in lines if len(line) < 35)
    weird_ratio = (ligatures + soft_artifacts) / total_chars

    return {
        "char_count": float(len(text)),
        "line_count": float(len(lines)),
        "ligatures": float(ligatures),
        "soft_artifacts": float(soft_artifacts),
        "hyphen_ends": float(hyphen_ends),
        "short_line_ratio": (short_lines / len(lines)) if lines else 1.0,
        "weird_ratio": weird_ratio,
    }


def looks_like_continued_table(native_text: str) -> bool:
    t = native_text.lower()
    if re.search(r"^\s*table\s+[a-z0-9]+\b", native_text, flags=re.IGNORECASE | re.MULTILINE):
        return True
    if "(continued on following page)" in t or "trial outcomes" in t:
        return True
    if re.search(r"^\s*table\s", t, flags=re.MULTILINE):
        return True
    return False


def should_use_native_text(native_text: str) -> bool:
    metrics = native_text_quality_score(native_text)
    t = native_text.lower()

    if metrics["char_count"] < 1000:
        return False
    if metrics["ligatures"] > 12:
        return False
    if metrics["hyphen_ends"] > 18:
        return False
    if metrics["short_line_ratio"] > 0.55:
        return False

    if looks_like_continued_table(native_text):
        return False

    if re.search(r"^\s*table\s+[a-z0-9]+\b", native_text, flags=re.IGNORECASE | re.MULTILINE):
        return False

    return True


def transcribe_page_with_llm(
    client: OpenAI,
    page_images: dict[str, bytes],
    pdf_name: str,
    page_num: int,
    previous_table_context: str | None = None,
) -> tuple[str, dict[str, int]]:
    system_prompt = (
        "You are a precise medical-journal document transcription engine. "
        "Transcribe visible page content into clean Markdown. "
        "Do not summarize. Do not explain. Do not infer missing text. "
        "Ignore non-table figures and diagrams. "
        "Do not transform content into LaTeX, Mermaid, or other synthetic formats. "
        "Return only the Markdown transcription for that page."
    )

    user_prompt = build_page_prompt(
        pdf_name=pdf_name,
        page_num=page_num,
        previous_table_context=previous_table_context,
    )

    content_parts: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
    for key in ("full", "main"):
        if key in page_images:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_bytes_to_data_url(page_images[key])},
                }
            )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    content = (response.choices[0].message.content or "").strip()
    content = clean_llm_markdown(content)

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
) -> tuple[str, dict[str, int]]:
    doc = fitz.open(pdf_path)
    page_texts: list[str] = []
    total_usage = {"input": 0, "output": 0, "total": 0}
    previous_table_context: str | None = None

    try:
        total_pages = len(doc)

        for page_index in range(total_pages):
            page_num = page_index + 1
            page = doc.load_page(page_index)

            print(f"  Page {page_num}/{total_pages}")

            native_text = normalize_unicode_text(page.get_text("text"))
            quality = native_text_quality_score(native_text)
            use_native = should_use_native_text(native_text)

            append_log(
                log_path,
                f"  Page {page_num}/{total_pages}\n"
                f"    Render DPI: {RENDER_DPI}\n"
                f"    Native chars: {int(quality['char_count'])}\n"
                f"    Ligatures: {int(quality['ligatures'])}\n"
                f"    Hyphen-ended lines: {int(quality['hyphen_ends'])}\n"
                f"    Short-line ratio: {quality['short_line_ratio']:.3f}\n"
                f"    Use native text: {use_native}\n",
            )

            if use_native:
                page_md = convert_native_text_to_md(native_text, page_num)
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

                append_log(
                    log_path,
                    "    Extraction mode: native_text\n"
                    "    Input tokens: 0\n"
                    "    Output tokens: 0\n"
                    "    Total tokens: 0\n\n",
                )
            else:
                page_images = render_pdf_page_variants(page, dpi=RENDER_DPI)
                page_md, usage = transcribe_page_with_llm(
                    client=client,
                    page_images=page_images,
                    pdf_name=pdf_path.name,
                    page_num=page_num,
                    previous_table_context=previous_table_context,
                )

                append_log(
                    log_path,
                    "    Extraction mode: vision_llm\n"
                    f"    Input tokens: {usage['input_tokens']}\n"
                    f"    Output tokens: {usage['output_tokens']}\n"
                    f"    Total tokens: {usage['total_tokens']}\n\n",
                )

            total_usage["input"] += usage["input_tokens"]
            total_usage["output"] += usage["output_tokens"]
            total_usage["total"] += usage["total_tokens"]

            if looks_like_continued_table(page_md):
                previous_table_context = page_md[:3000]
            else:
                previous_table_context = None

            block = [
                f"<!-- PAGE {page_num} START -->",
                "",
                f"## Page {page_num}",
                "",
                page_md,
                "",
                f"<!-- PAGE {page_num} END -->",
            ]
            page_texts.append("\n".join(block))

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    finally:
        doc.close()

    document_header = [
        f"# {pdf_path.stem}",
        "",
        f"Source PDF: `{pdf_path.name}`",
        "",
    ]

    full_markdown = "\n".join(document_header) + "\n\n" + "\n\n".join(page_texts)
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

    client = OpenAI(api_key=api_key, base_url=base_url)
    global_usage = {"input": 0, "output": 0, "total": 0}

    append_log(log_path, f"Input directory: {INPUT_DIR}\n")
    append_log(log_path, f"Output directory: {OUTPUT_DIR}\n")
    append_log(log_path, f"Model: {MODEL_NAME}\n")
    append_log(log_path, f"Render DPI: {RENDER_DPI}\n")
    append_log(log_path, f"Max files: {MAX_FILES}\n\n")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        append_log(log_path, f"Processing: {pdf_file.name}\n")

        try:
            full_markdown, usage = extract_pdf_with_vision_llm(
                pdf_file,
                client,
                log_path,
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