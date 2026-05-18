#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("artifacts/elsevier/filtered_xml")
OUTPUT_DIR = Path("artifacts/elsevier/factoids")
FAILED_DIR = Path("artifacts/elsevier/factoids_failed")
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

MAX_INPUT_CHARS_PER_CHUNK = 120_000
CHUNK_OVERLAP_CHARS = 4_000
MAX_TABLE_ROWS_PER_TABLE = 250

# Set to an integer like 100, 1000, etc. to limit the run.
# Leave as None to process all pending files.
MAX_FILES: Optional[int] = None

# If True, skip files whose output JSON already exists and is readable.
SKIP_ALREADY_PROCESSED = True


# ============================================================
# LOGGING
# ============================================================

class Tee:
    def __init__(self, filepath: Path) -> None:
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

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
    log_path = LOG_DIR / f"factoids_elsevier_{time.strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 80)
    print("ELSEVIER FACTOID EXTRACTION LOG")
    print("=" * 80)
    print(f"Log file: {log_path}")
    print(f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    script_path = Path(__file__).resolve()
    print("=" * 80)
    print("SCRIPT SOURCE")
    print("=" * 80)
    try:
        print(script_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read script source from {script_path}: {e}")
    print()
    print("=" * 80)
    print("RUN OUTPUT")
    print("=" * 80)

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

def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def clean_text(text: str) -> str:
    return normalize(html.unescape((text or "").replace("\xa0", " ")))


def extract_blocks(text: str) -> List[str]:
    pattern = re.compile(
        re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
        re.DOTALL,
    )
    return [b.strip() for b in pattern.findall(text)]


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def output_path_for(input_file: Path) -> Path:
    return OUTPUT_DIR / f"{input_file.stem}_factoids.json"


def fail_path_for(input_file: Path) -> Path:
    return FAILED_DIR / f"{input_file.stem}_failed.json"


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def short_error_message(exc: Exception) -> str:
    msg = str(exc).strip()
    if not msg:
        msg = exc.__class__.__name__
    return f"{exc.__class__.__name__}: {msg}"


def local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def direct_children(elem: Optional[ET.Element], name: str) -> List[ET.Element]:
    if elem is None:
        return []
    return [child for child in list(elem) if local_name(child.tag) == name]


def first_child(elem: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for child in list(elem):
        if local_name(child.tag) == name:
            return child
    return None


def iter_elems(elem: Optional[ET.Element], name: str):
    if elem is None:
        return
    for child in elem.iter():
        if local_name(child.tag) == name:
            yield child


def elem_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    parts: List[str] = []
    for t in elem.itertext():
        t = clean_text(t)
        if t:
            parts.append(t)
    return normalize(" ".join(parts))


def first_text_from_children(elem: Optional[ET.Element], name: str) -> str:
    child = first_child(elem, name)
    return elem_text(child)


def first_text_in_subtree(elem: Optional[ET.Element], name: str) -> str:
    if elem is None:
        return ""
    for child in elem.iter():
        if local_name(child.tag) == name:
            text = elem_text(child)
            if text:
                return text
    return ""


def extract_year(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def split_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0

            start = 0
            step = max_chars - overlap_chars if max_chars > overlap_chars else max_chars
            while start < para_len:
                end = min(start + max_chars, para_len)
                chunks.append(para[start:end])
                if end >= para_len:
                    break
                start += step
            continue

        projected = current_len + (2 if current else 0) + para_len
        if current and projected > max_chars:
            chunks.append("\n\n".join(current))

            if overlap_chars > 0:
                overlap_parts: List[str] = []
                overlap_len = 0
                for old_para in reversed(current):
                    overlap_parts.insert(0, old_para)
                    overlap_len += len(old_para) + 2
                    if overlap_len >= overlap_chars:
                        break
                current = overlap_parts
                current_len = len("\n\n".join(current))
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len = len("\n\n".join(current))

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def is_already_processed(input_file: Path) -> bool:
    out_path = output_path_for(input_file)
    if not out_path.exists():
        return False

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "factoids" in data
    except Exception:
        return False


def collect_files_for_run() -> Tuple[List[Path], List[Path], int]:
    all_files = sorted(INPUT_DIR.glob("*.xml"))

    if not SKIP_ALREADY_PROCESSED:
        pending_files = all_files[:]
        already_done = 0
    else:
        pending_files = []
        already_done = 0

        for f in all_files:
            if is_already_processed(f):
                already_done += 1
            else:
                pending_files.append(f)

    if MAX_FILES is not None:
        pending_files = pending_files[:MAX_FILES]

    return all_files, pending_files, already_done


# ============================================================
# XML DEBUG
# ============================================================

def describe_xml_structure(root: ET.Element) -> str:
    top = [local_name(child.tag) for child in list(root)]
    return f"root={local_name(root.tag)} | children={top[:20]}"


# ============================================================
# ELSEVIER XML PARSING
# ============================================================

def parse_xml_root(path: Path) -> ET.Element:
    tree = ET.parse(path)
    return tree.getroot()


def extract_authors(coredata: Optional[ET.Element], article: Optional[ET.Element]) -> List[str]:
    authors: List[str] = []

    if coredata is not None:
        for creator in direct_children(coredata, "creator"):
            name = clean_text(creator.text or "")
            if name:
                authors.append(name)

    if authors:
        return authors

    head = first_child(article, "head") if article is not None else None
    author_group = first_child(head, "author-group") if head is not None else None
    if author_group is None:
        return authors

    for author in direct_children(author_group, "author"):
        given = first_text_from_children(author, "given-name")
        surname = first_text_from_children(author, "surname")
        name = normalize(" ".join(x for x in [given, surname] if x))
        if name:
            authors.append(name)

    return authors


def extract_metadata(
    root: ET.Element,
    input_file: Path,
) -> Tuple[Dict[str, Any], Optional[ET.Element], Optional[ET.Element]]:
    coredata = next(iter_elems(root, "coredata"), None)
    article = next(iter_elems(root, "article"), None)
    head = first_child(article, "head") if article is not None else None

    title = (
        first_text_from_children(coredata, "title")
        or first_text_from_children(head, "title")
        or input_file.stem
    )

    doi = first_text_from_children(coredata, "doi")
    pii = first_text_from_children(coredata, "pii")
    eid = first_text_from_children(coredata, "eid")
    journal = first_text_from_children(coredata, "publicationName")
    aggregation_type = first_text_from_children(coredata, "aggregationType")
    pub_type = first_text_from_children(coredata, "pubType")
    issn = first_text_from_children(coredata, "issn")
    volume = first_text_from_children(coredata, "volume")
    issue = (
        first_text_from_children(coredata, "issueIdentifier")
        or first_text_from_children(coredata, "number")
    )
    starting_page = first_text_from_children(coredata, "startingPage")
    ending_page = first_text_from_children(coredata, "endingPage")
    page_range = first_text_from_children(coredata, "pageRange")
    cover_date = first_text_from_children(coredata, "coverDate")
    cover_display_date = first_text_from_children(coredata, "coverDisplayDate")
    publisher = first_text_from_children(coredata, "publisher")
    copyright_text = first_text_from_children(coredata, "copyright")
    description = first_text_from_children(coredata, "description")
    openaccess = first_text_from_children(coredata, "openaccess")
    openaccess_article = first_text_from_children(coredata, "openaccessArticle")

    scopus_id = first_text_in_subtree(root, "scopus-id")
    pubmed_id = first_text_in_subtree(root, "pubmed-id")
    document_type = first_text_in_subtree(root, "document-type")
    document_subtype = first_text_in_subtree(root, "document-subtype")

    publication_year = (
        extract_year(cover_date)
        or extract_year(cover_display_date)
        or extract_year(first_text_in_subtree(root, "cover-date-start"))
    )

    authors = extract_authors(coredata, article)

    metadata: Dict[str, Any] = {
        "source_family": "Elsevier",
        "source_format": "XML",
        "file_name": input_file.name,
        "document_title": title,
        "document_type": document_type or aggregation_type or "article",
        "document_subtype": document_subtype or pub_type,
        "document_year": publication_year,
        "journal": journal,
        "doi": doi,
        "pii": pii,
        "eid": eid,
        "pubmed_id": pubmed_id,
        "scopus_id": scopus_id,
        "issn": issn,
        "volume": volume,
        "issue": issue,
        "starting_page": starting_page,
        "ending_page": ending_page,
        "page_range": page_range,
        "cover_date": cover_date,
        "cover_display_date": cover_display_date,
        "publisher": publisher,
        "copyright": copyright_text,
        "openaccess": openaccess,
        "openaccess_article": openaccess_article,
        "authors": authors,
        "abstract_from_coredata": description,
    }

    return metadata, article, coredata


def extract_abstract_text(article: Optional[ET.Element], coredata: Optional[ET.Element]) -> str:
    head = first_child(article, "head") if article is not None else None
    if head is not None:
        blocks: List[str] = []
        for abstract in direct_children(head, "abstract"):
            sec_blocks: List[str] = []
            for child in list(abstract):
                lname = local_name(child.tag)

                if lname == "abstract-sec":
                    title = first_text_from_children(child, "section-title")
                    paras: List[str] = []
                    for gc in list(child):
                        if local_name(gc.tag) in {"simple-para", "para"}:
                            text = elem_text(gc)
                            if text:
                                paras.append(text)

                    part: List[str] = []
                    if title:
                        part.append(title)
                    part.extend(paras)

                    if part:
                        sec_blocks.append("\n".join(part))

                elif lname in {"simple-para", "para"}:
                    text = elem_text(child)
                    if text:
                        sec_blocks.append(text)

            if sec_blocks:
                blocks.append("\n\n".join(sec_blocks))

        if blocks:
            return "\n\n".join(blocks)

    if coredata is not None:
        return first_text_from_children(coredata, "description")

    return ""


def render_section(section: ET.Element) -> str:
    parts: List[str] = []

    title = first_text_from_children(section, "section-title")
    if title:
        parts.append(title)

    for child in list(section):
        lname = local_name(child.tag)

        if lname == "section-title":
            continue

        if lname == "para":
            text = elem_text(child)
            if text:
                parts.append(text)

        elif lname == "section":
            nested = render_section(child)
            if nested:
                parts.append(nested)

    return "\n\n".join(parts)


def extract_body_text(article: Optional[ET.Element]) -> str:
    body = first_child(article, "body") if article is not None else None
    if body is None:
        return ""

    parts: List[str] = []

    for child in list(body):
        lname = local_name(child.tag)

        if lname == "sections":
            for section in direct_children(child, "section"):
                text = render_section(section)
                if text:
                    parts.append(text)

        elif lname == "section":
            text = render_section(child)
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def extract_tables_text(article: Optional[ET.Element]) -> str:
    if article is None:
        return ""

    table_blocks: List[str] = []

    for table in iter_elems(article, "table"):
        block: List[str] = []

        label = first_text_from_children(table, "label")
        caption = elem_text(first_child(table, "caption"))
        if label and caption:
            block.append(f"{label}: {caption}")
        elif label or caption:
            block.append(label or caption)

        row_lines: List[str] = []
        for row in iter_elems(table, "row"):
            cells: List[str] = []
            for entry in list(row):
                if local_name(entry.tag) == "entry":
                    cells.append(elem_text(entry))

            if any(cells):
                row_lines.append(" | ".join(cell if cell else " " for cell in cells))

            if len(row_lines) >= MAX_TABLE_ROWS_PER_TABLE:
                row_lines.append("[table truncated]")
                break

        if row_lines:
            block.append("\n".join(row_lines))

        legend = elem_text(first_child(table, "legend"))
        if legend:
            block.append(f"Legend: {legend}")

        footnotes: List[str] = []
        for fn in direct_children(table, "table-footnote"):
            text = elem_text(fn)
            if text:
                footnotes.append(text)

        if footnotes:
            block.append("Footnotes:\n" + "\n".join(footnotes))

        final_block = "\n\n".join(part for part in block if part)
        if final_block:
            table_blocks.append(final_block)

    return "\n\n".join(table_blocks)


def extract_textboxes_text(article: Optional[ET.Element]) -> str:
    if article is None:
        return ""

    blocks: List[str] = []

    for textbox in iter_elems(article, "textbox"):
        parts: List[str] = []

        label = first_text_from_children(textbox, "label")
        caption = elem_text(first_child(textbox, "caption"))
        if label and caption:
            parts.append(f"{label}: {caption}")
        elif label or caption:
            parts.append(label or caption)

        paras: List[str] = []
        for para in iter_elems(textbox, "para"):
            text = elem_text(para)
            if text:
                paras.append(text)

        if paras:
            parts.append("\n".join(paras))

        block = "\n\n".join(parts)
        if block:
            blocks.append(block)

    return "\n\n".join(blocks)


def extract_fallback_text_from_article(article: Optional[ET.Element]) -> str:
    if article is None:
        return ""

    parts: List[str] = []

    for tag_name in ["title", "abstract", "section-title", "para", "simple-para"]:
        for elem in iter_elems(article, tag_name):
            text = elem_text(elem)
            if text:
                parts.append(text)

    seen = set()
    deduped: List[str] = []
    for p in parts:
        key = p.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return "\n\n".join(deduped).strip()


def extract_fallback_text_from_root(root: ET.Element) -> str:
    parts: List[str] = []

    for tag_name in ["title", "description", "abstract", "section-title", "para", "simple-para"]:
        for elem in iter_elems(root, tag_name):
            text = elem_text(elem)
            if text:
                parts.append(text)

    seen = set()
    deduped: List[str] = []
    for p in parts:
        key = p.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return "\n\n".join(deduped).strip()


def build_full_text(
    metadata: Dict[str, Any],
    abstract: str,
    body: str,
    tables: str,
    textboxes: str,
) -> str:
    parts: List[str] = []

    title = metadata.get("document_title") or ""
    journal = metadata.get("journal") or ""
    doi = metadata.get("doi") or ""

    header_bits = [f"Title: {title}"]
    if journal:
        header_bits.append(f"Journal: {journal}")
    if doi:
        header_bits.append(f"DOI: {doi}")
    parts.append("\n".join(header_bits))

    if abstract:
        parts.append("ABSTRACT\n" + abstract)

    if body:
        parts.append("BODY\n" + body)

    if tables:
        parts.append("TABLES\n" + tables)

    if textboxes:
        parts.append("PANELS\n" + textboxes)

    return "\n\n".join(part for part in parts if part).strip()


def parse_elsevier_xml(input_file: Path) -> Tuple[Dict[str, Any], str]:
    root = parse_xml_root(input_file)
    metadata, article, coredata = extract_metadata(root, input_file)

    abstract = extract_abstract_text(article, coredata)
    body = extract_body_text(article)
    tables = extract_tables_text(article)
    textboxes = extract_textboxes_text(article)

    full_text = build_full_text(metadata, abstract, body, tables, textboxes)

    if not abstract and not body:
        fallback_article_text = extract_fallback_text_from_article(article)
        fallback_root_text = extract_fallback_text_from_root(root)
        fallback_text = fallback_article_text or fallback_root_text

        if fallback_text:
            print(
                f"[{now_ts()}] [WARN] {input_file.name} -> "
                "standard abstract/body extraction failed; using fallback text extraction"
            )
            full_text = build_full_text(
                metadata=metadata,
                abstract=abstract,
                body=fallback_text,
                tables=tables,
                textboxes=textboxes,
            )
        else:
            print(f"[{now_ts()}] [DEBUG] {input_file.name} -> {describe_xml_structure(root)}")
            raise ValueError("Could not extract abstract/body text from Elsevier XML")

    if not full_text.strip():
        raise ValueError("Extracted full_text is empty")

    return metadata, full_text


# ============================================================
# PROMPT
# ============================================================

def build_prompts(document_title: str, full_text: str) -> Tuple[str, str]:
    system = (
        "You extract atomic, self-contained clinical factoids from breast-cancer-related full text.\n"
        "Each factoid must be understandable on its own, without relying on previous factoids.\n"
        "Use only information explicitly stated in the text.\n"
        "Do not invent, infer, or generalize beyond the text.\n"
        "Avoid duplicate or near-duplicate factoids.\n"
        "Output ONLY tagged factoids in the exact format below.\n"
    )

    user = (
        f"Document title: {document_title}\n\n"
        f"Task:\n"
        f"Extract atomic, self-contained factoids from the full text below.\n\n"
        f"Critical rules:\n"
        f"- Every factoid must stand alone.\n"
        f"- Pretend each factoid will be read independently in a retrieval system.\n"
        f"- Do not assume the reader has seen any previous factoid.\n"
        f"- Expand every abbreviation or acronym inside the same factoid where it appears.\n"
        f"- If needed, restate the full disease, modality, biomarker, drug, or population in each factoid.\n"
        f"- Avoid pronouns or vague references unless the referent is explicitly named in the same sentence.\n"
        f"- Use precise wording that preserves the original meaning.\n"
        f"- If the source text uses shorthand like 'PA', 'MRI', 'HER2', or a trial acronym, write the expanded form in that factoid unless the abbreviation is universally standard and the full term is also included.\n"
        f"- Good example: 'Photoacoustic (PA) imaging can be integrated into conventional ultrasound systems for breast cancer assessment.'\n"
        f"- Bad example: 'PA imaging can be integrated into conventional ultrasound systems.'\n"
        f"- If the text contains no useful self-contained factoids, output nothing.\n\n"
        f"Required format:\n"
        f"{FACTOID_START}\n"
        f"<factoid>\n"
        f"{FACTOID_END}\n\n"
        f"Full text:\n{full_text}"
    )

    return system, user


# ============================================================
# FACTOID EXTRACTION
# ============================================================

async def request_factoids_for_chunk(
    client: AsyncOpenAI,
    document_title: str,
    chunk_text: str,
) -> List[str]:
    system, user = build_prompts(document_title, chunk_text)

    async def _do_request() -> str:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
        )
        return resp.choices[0].message.content or ""

    content = await asyncio.wait_for(_do_request(), timeout=REQUEST_TIMEOUT_SECONDS)
    blocks = [normalize(b) for b in extract_blocks(content)]
    return [b for b in blocks if b]


async def extract_factoids(
    client: AsyncOpenAI,
    document_title: str,
    full_text: str,
) -> List[str]:
    chunks = split_text(
        full_text,
        max_chars=MAX_INPUT_CHARS_PER_CHUNK,
        overlap_chars=CHUNK_OVERLAP_CHARS,
    )

    if not chunks:
        return []

    all_blocks: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(
                f"[{now_ts()}] [CHUNK] {document_title[:80]} | "
                f"chunk={i}/{len(chunks)} | chars={len(chunk)}"
            )
        chunk_blocks = await request_factoids_for_chunk(client, document_title, chunk)
        all_blocks.extend(chunk_blocks)

    seen = set()
    deduped: List[str] = []
    for b in all_blocks:
        key = b.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(b)

    return deduped


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

    async def record_success(self, filename: str, factoid_count: int, worker_id: int) -> int:
        async with self.lock:
            self.succeeded += 1
            self.processed += 1
            if factoid_count == 0:
                self.empty_factoids += 1
            current = self.processed
            print(
                f"[{now_ts()}] [{current}/{self.total}] [OK] worker={worker_id} "
                f"{filename} -> {factoid_count} factoids"
            )
            self._maybe_print()
            return current

    async def record_failure(self, filename: str, err: str, worker_id: int) -> int:
        async with self.lock:
            self.failed += 1
            self.processed += 1
            current = self.processed
            print(
                f"[{now_ts()}] [{current}/{self.total}] [ERROR] worker={worker_id} "
                f"{filename} -> {err}"
            )
            self._maybe_print()
            return current

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
                f"empty={self.empty_factoids} | "
                f"rate={rate:.2f} files/sec | eta={eta_sec/60:.1f} min"
            )

    def final_print(self) -> None:
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0.0
        print("\n========== PERFORMANCE ==========")
        print(f"Discovered files in this run: {self.discovered}")
        print(f"Processed:                    {self.processed}")
        print(f"Succeeded:                    {self.succeeded}")
        print(f"Failed:                       {self.failed}")
        print(f"Skipped:                      {self.skipped}")
        print(f"Empty factoids:               {self.empty_factoids}")
        print(f"Total time:                   {elapsed:.2f}s")
        print(f"Throughput:                   {rate:.2f} files/sec")
        print("=================================")


# ============================================================
# FILE PROCESSING
# ============================================================

async def process_one_file(
    client: AsyncOpenAI,
    input_file: Path,
) -> Tuple[bool, str, int]:
    metadata, full_text = parse_elsevier_xml(input_file)

    document_title = metadata.get("document_title") or input_file.stem
    factoids = await extract_factoids(client, document_title, full_text)

    output = {
        "metadata": metadata,
        "factoids": [
            {"id": i + 1, "factoid_text": f}
            for i, f in enumerate(factoids)
        ],
    }

    out_path = output_path_for(input_file)
    safe_write_json(out_path, output)

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

        except ValueError as e:
            last_error = short_error_message(e)
            fail_payload = {
                "input_file": str(input_file),
                "error": last_error,
                "timestamp": now_ts(),
            }
            safe_write_json(fail_path_for(input_file), fail_payload)
            return False, input_file.name, 0, last_error

        except Exception as e:
            last_error = short_error_message(e)

            if attempt < MAX_RETRIES:
                backoff = min(2 ** (attempt - 1), 30)
                print(
                    f"[{now_ts()}] [RETRY] {input_file.name} | "
                    f"attempt={attempt}/{MAX_RETRIES} | "
                    f"error={last_error} | sleeping={backoff}s"
                )
                await asyncio.sleep(backoff)
            else:
                fail_payload = {
                    "input_file": str(input_file),
                    "error": last_error,
                    "timestamp": now_ts(),
                }
                safe_write_json(fail_path_for(input_file), fail_payload)
                return False, input_file.name, 0, last_error

    return False, input_file.name, 0, last_error


# ============================================================
# WORKER
# ============================================================

async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    stats: Stats,
) -> None:
    while True:
        try:
            input_file = await queue.get()
        except asyncio.CancelledError:
            return

        try:
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

    all_files, files_to_run, already_done = collect_files_for_run()

    if not all_files:
        print(f"No input files found in {INPUT_DIR}")
        return

    print(f"Input dir:              {INPUT_DIR}")
    print(f"Output dir:             {OUTPUT_DIR}")
    print(f"Failed dir:             {FAILED_DIR}")
    print(f"Log dir:                {LOG_DIR}")
    print(f"Model:                  {MODEL_NAME}")
    print(f"Concurrency:            {CONCURRENCY}")
    print(f"Max retries:            {MAX_RETRIES}")
    print(f"Request timeout:        {REQUEST_TIMEOUT_SECONDS}s")
    print(f"Max files this run:     {MAX_FILES if MAX_FILES is not None else 'ALL pending'}")
    print(f"Discovered XML files:   {len(all_files)}")
    print(f"Already processed:      {already_done}")
    print(f"Queued this run:        {len(files_to_run)}")
    print(f"Chunk size:             {MAX_INPUT_CHARS_PER_CHUNK} chars")
    print(f"Chunk overlap:          {CHUNK_OVERLAP_CHARS} chars")
    print(f"Restart skip enabled:   {SKIP_ALREADY_PROCESSED}")
    print("Mode:                   Elsevier XML -> factoids")
    print()

    if not files_to_run:
        print("No pending files to process.")
        return

    stats = Stats(total=len(files_to_run))
    queue: asyncio.Queue = asyncio.Queue()

    for f in files_to_run:
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

    stats.final_print()


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