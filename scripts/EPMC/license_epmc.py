#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from openai import OpenAI


INPUT_DIR = Path("artifacts/EPMC/filtered_xml")
OUTPUT_JSON = Path("artifacts/EPMC/license_summary.json")

# Set to an integer like 500 for testing.
# Set to None to run on all files.
MAX_FILES = None

MODEL_NAME = "GPT-OSS-120B"
MAX_COMPLETION_TOKENS = 200
MAX_WORKERS = min(16, (os.cpu_count() or 4) * 2)
MAX_PENDING_MULTIPLIER = 4
CHECKPOINT_EVERY = 500
PROGRESS_EVERY = 100

LLM_ALLOWED_LABELS = [
    "CC_BY",
    "CC_BY_SA",
    "CC_BY_NC",
    "CC_BY_NC_SA",
    "CC_BY_ND",
    "CC_BY_NC_ND",
    "CC_UNKNOWN",
]

SUPPORTED_CC_VERSIONS = {"1.0", "2.0", "2.5", "3.0", "4.0"}


THREAD_LOCAL = threading.local()


def get_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("Missing environment variable VIRTUAL_API_KEY.")
    if not base_url:
        raise RuntimeError("Missing environment variable BASE_URL.")

    return OpenAI(api_key=api_key, base_url=base_url)


def get_thread_client() -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = get_client()
        THREAD_LOCAL.client = client
    return client


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def collect_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return clean_text(" ".join(elem.itertext()))


def extract_permissions_text(root: ET.Element) -> str:
    article_meta = root.find(".//article-meta")
    if article_meta is None:
        return ""

    parts: list[str] = []

    permissions = article_meta.find("permissions")
    if permissions is not None:
        parts.append(collect_text(permissions))

    for tag in [".//license", ".//copyright-statement", ".//copyright-holder"]:
        for elem in article_meta.findall(tag):
            txt = collect_text(elem)
            if txt:
                parts.append(txt)

    return clean_text(" ".join(parts))


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def detect_article_license_rule(text: str) -> str:
    t = text.lower()

    # Temporary PMC / COVID reuse statements
    if (
        "pmc open access subset" in t
        and "world health organization (who) declaration of covid-19 as a global pandemic" in t
    ):
        return "PMC_OPEN_ACCESS_TEMP"

    if (
        "pmc open access subset" in t
        and "duration of the covid-19 pandemic" in t
    ):
        return "PMC_OPEN_ACCESS_TEMP"

    if (
        "pmc open access subset" in t
        and "until permissions are revoked in writing" in t
    ):
        return "PMC_OPEN_ACCESS_TEMP"

    # Most specific first
    if contains_any(t, [
        "creativecommons.org/licenses/by-nc-nd/4.0",
        "cc by-nc-nd 4.0",
        "creative commons attribution-noncommercial-noderivs 4.0",
        "creative commons attribution-noncommercial-no derivatives 4.0",
    ]):
        return "CC_BY_NC_ND_4_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nc-sa/4.0",
        "cc by-nc-sa 4.0",
        "creative commons attribution-noncommercial-sharealike 4.0",
    ]):
        return "CC_BY_NC_SA_4_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nd/4.0",
        "cc by-nd 4.0",
        "creative commons attribution-noderivs 4.0",
        "creative commons attribution-no derivatives 4.0",
    ]):
        return "CC_BY_ND_4_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-sa/4.0",
        "cc by-sa 4.0",
        "creative commons attribution-sharealike 4.0",
    ]):
        return "CC_BY_SA_4_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nc/4.0",
        "cc by-nc 4.0",
        "creative commons attribution-noncommercial 4.0",
    ]):
        return "CC_BY_NC_4_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by/4.0",
        "cc by 4.0",
        "cc-by 4.0",
        "creative commons attribution 4.0",
        "licensed under a creative commons attribution 4.0 international license",
        "creative commons attribution license (cc by)",
    ]):
        return "CC_BY_4_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nc-nd/3.0",
        "cc by-nc-nd 3.0",
    ]):
        return "CC_BY_NC_ND_3_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nc-sa/3.0",
        "cc by-nc-sa 3.0",
    ]):
        return "CC_BY_NC_SA_3_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nd/3.0",
        "cc by-nd 3.0",
    ]):
        return "CC_BY_ND_3_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-sa/3.0",
        "cc by-sa 3.0",
    ]):
        return "CC_BY_SA_3_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nc/3.0",
        "cc by-nc 3.0",
    ]):
        return "CC_BY_NC_3_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by/3.0",
        "cc by 3.0",
        "cc-by 3.0",
    ]):
        return "CC_BY_3_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by/2.0",
        "cc by 2.0",
        "cc-by 2.0",
    ]):
        return "CC_BY_2_0"

    if contains_any(t, [
        "creativecommons.org/licenses/by-nc-sa/1.0",
        "cc by-nc-sa 1.0",
        "cc-by-nc-sa 1.0",
    ]):
        return "CC_BY_NC_SA_1_0"

    if contains_any(t, [
        "creativecommons.org/publicdomain/mark/1.0",
    ]):
        return "PUBLIC_DOMAIN_MARK_1_0"

    if contains_any(t, [
        "creativecommons.org/publicdomain/zero/1.0",
        "cc0 1.0",
        "creative commons public domain dedication waiver",
    ]):
        return "CC0_1_0"

    if "creative commons" in t:
        return "CREATIVE_COMMONS_UNCLEAR"

    if "open access" in t:
        return "OPEN_ACCESS_UNCLEAR"

    return "UNKNOWN"


def build_llm_prompt(permissions_text: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a license classifier for biomedical article permissions text. "
        "Your task is to identify the Creative Commons family only. "
        "Choose exactly one label from this list:\n"
        "CC_BY\n"
        "CC_BY_SA\n"
        "CC_BY_NC\n"
        "CC_BY_NC_SA\n"
        "CC_BY_ND\n"
        "CC_BY_NC_ND\n"
        "CC_UNKNOWN\n\n"
        "Important rules:\n"
        "- Only classify Creative Commons article licenses.\n"
        "- Ignore CC0/data waivers and focus on the article license.\n"
        "- If the text says 'Creative Commons Attribution License' with no version, return CC_BY.\n"
        "- If the text says Attribution-NonCommercial, return CC_BY_NC.\n"
        "- If the text says Attribution-NonCommercial-ShareAlike, return CC_BY_NC_SA.\n"
        "- If the text says Attribution-NonCommercial-NoDerivs, return CC_BY_NC_ND.\n"
        "- If the text says Attribution-NoDerivs, return CC_BY_ND.\n"
        "- If the text says Attribution-ShareAlike, return CC_BY_SA.\n"
        "- If you are not confident it is a Creative Commons license family, return CC_UNKNOWN.\n"
        "- Reply with exactly one label and nothing else."
    )

    user_prompt = (
        "Classify this permissions text into one Creative Commons family label.\n\n"
        f"{permissions_text}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def classify_cc_unclear_with_llm(
    permissions_text: str,
    cache: dict[str, str],
    cache_lock: threading.Lock,
) -> str:
    key = permissions_text.strip()
    with cache_lock:
        cached = cache.get(key)
    if cached is not None:
        return cached

    response = get_thread_client().chat.completions.create(
        messages=build_llm_prompt(permissions_text),
        model=MODEL_NAME,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    content = clean_text(response.choices[0].message.content or "")
    label = content.strip().upper()

    if label not in LLM_ALLOWED_LABELS:
        match = re.search(
            r"\b(CC_BY_NC_ND|CC_BY_NC_SA|CC_BY_NC|CC_BY_ND|CC_BY_SA|CC_BY|CC_UNKNOWN)\b",
            label,
        )
        if match:
            label = match.group(1)
        else:
            label = "CC_UNKNOWN"

    with cache_lock:
        cache[key] = label
    return label


def detect_supported_cc_version(text: str) -> str | None:
    t = text.lower()

    patterns = [
        r"creativecommons\.org/licenses/by-nc-nd/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nc-sa/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nc/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nd/(\d\.\d)",
        r"creativecommons\.org/licenses/by-sa/(\d\.\d)",
        r"creativecommons\.org/licenses/by/(\d\.\d)",
        r"\bcc by-nc-nd\s+(\d\.\d)\b",
        r"\bcc by-nc-sa\s+(\d\.\d)\b",
        r"\bcc by-nc\s+(\d\.\d)\b",
        r"\bcc by-nd\s+(\d\.\d)\b",
        r"\bcc by-sa\s+(\d\.\d)\b",
        r"\bcc by\s+(\d\.\d)\b",
        r"creative commons attribution-noncommercial-noderivs\s+(\d\.\d)",
        r"creative commons attribution-noncommercial-sharealike\s+(\d\.\d)",
        r"creative commons attribution-noncommercial\s+(\d\.\d)",
        r"creative commons attribution-no derivatives\s+(\d\.\d)",
        r"creative commons attribution-noderivs\s+(\d\.\d)",
        r"creative commons attribution-sharealike\s+(\d\.\d)",
        r"creative commons attribution\s+(\d\.\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            version = match.group(1)
            if version in SUPPORTED_CC_VERSIONS:
                return version

    return None


def combine_rule_and_llm_label(
    rule_label: str,
    llm_label: str,
    permissions_text: str,
) -> str:
    if rule_label != "CREATIVE_COMMONS_UNCLEAR":
        return rule_label

    version = detect_supported_cc_version(permissions_text)
    if llm_label == "CC_UNKNOWN":
        return "CREATIVE_COMMONS_UNCLEAR"

    if version:
        return f"{llm_label}_{version.replace('.', '_')}"

    return f"{llm_label}_UNSPECIFIED"


def refine_existing_record(record: dict) -> dict:
    updated = dict(record)
    current_label = str(updated.get("license_label", "")).strip()
    rule_label = str(updated.get("rule_label", "")).strip()
    llm_label = str(updated.get("llm_cc_family_label", "")).strip().upper()
    permissions_text = str(updated.get("raw_permissions_text", "")).strip()

    if (
        current_label.endswith("_UNSPECIFIED")
        and rule_label == "CREATIVE_COMMONS_UNCLEAR"
        and llm_label in LLM_ALLOWED_LABELS
        and llm_label != "CC_UNKNOWN"
        and permissions_text
    ):
        updated["license_label"] = combine_rule_and_llm_label(
            rule_label=rule_label,
            llm_label=llm_label,
            permissions_text=permissions_text,
        )

    return updated


def load_existing_output(output_path: Path) -> tuple[list[dict], dict[str, str]]:
    if not output_path.exists():
        return [], {}

    try:
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load existing output {output_path}: {e}")
        return [], {}

    existing_records = data.get("per_file", [])
    if not isinstance(existing_records, list):
        print(f"[WARN] Existing output has invalid 'per_file'; starting fresh.")
        return [], {}

    llm_cache: dict[str, str] = {}
    normalized_records: list[dict] = []

    for record in existing_records:
        if not isinstance(record, dict):
            continue

        normalized_records.append(refine_existing_record(record))

        raw_permissions_text = str(record.get("raw_permissions_text", "")).strip()
        llm_label = str(record.get("llm_cc_family_label", "")).strip().upper()
        if raw_permissions_text and llm_label in LLM_ALLOWED_LABELS:
            llm_cache[raw_permissions_text] = llm_label

    return normalized_records, llm_cache


def build_output_payload(
    xml_files: list[Path],
    per_file: list[dict],
    llm_cache: dict[str, str],
) -> dict:
    counts: Counter[str] = Counter()
    for record in per_file:
        label = record.get("license_label", "")
        if label:
            counts[str(label)] += 1

    processed_files = {
        str(record.get("file", ""))
        for record in per_file
        if str(record.get("file", "")).strip()
    }

    return {
        "input_dir": str(INPUT_DIR),
        "files_scanned": len(xml_files),
        "files_processed": len(processed_files),
        "files_remaining": max(len(xml_files) - len(processed_files), 0),
        "max_files": MAX_FILES,
        "model_name": MODEL_NAME,
        "license_counts": dict(counts.most_common()),
        "llm_cache_size": len(llm_cache),
        "per_file": per_file,
    }


def save_output(
    output_path: Path,
    xml_files: list[Path],
    per_file: list[dict],
    llm_cache: dict[str, str],
) -> None:
    payload = build_output_payload(
        xml_files=xml_files,
        per_file=per_file,
        llm_cache=llm_cache,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    temp_path.replace(output_path)


def extract_ids(root: ET.Element) -> dict[str, str]:
    result = {
        "pmcid": "",
        "pmid": "",
        "doi": "",
        "title": "",
    }

    article_meta = root.find(".//article-meta")
    if article_meta is None:
        return result

    for article_id in article_meta.findall("article-id"):
        pub_id_type = article_id.attrib.get("pub-id-type", "").lower()
        text = clean_text(article_id.text)
        if pub_id_type == "pmcid":
            result["pmcid"] = text
        elif pub_id_type == "pmid":
            result["pmid"] = text
        elif pub_id_type == "doi":
            result["doi"] = text

    title_elem = article_meta.find("title-group/article-title")
    result["title"] = collect_text(title_elem)

    return result


def analyze_file(
    xml_path: Path,
    llm_enabled: bool,
    llm_cache: dict[str, str],
    llm_cache_lock: threading.Lock,
) -> dict:
    record = {
        "file": xml_path.name,
        "pmcid": "",
        "pmid": "",
        "doi": "",
        "title": "",
        "license_label": "",
        "license_label_source": "",
        "rule_label": "",
        "llm_cc_family_label": "",
        "raw_permissions_text": "",
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        record["license_label"] = "PARSE_ERROR"
        record["license_label_source"] = "parse_error"
        record["raw_permissions_text"] = str(e)
        return record

    ids = extract_ids(root)
    record.update(ids)

    permissions_text = extract_permissions_text(root)
    record["raw_permissions_text"] = permissions_text

    if not permissions_text:
        record["license_label"] = "NO_LICENSE_TEXT"
        record["license_label_source"] = "rule"
        return record

    rule_label = detect_article_license_rule(permissions_text)
    record["rule_label"] = rule_label

    if rule_label == "CREATIVE_COMMONS_UNCLEAR":
        if not llm_enabled:
            record["license_label"] = "CREATIVE_COMMONS_UNCLEAR"
            record["license_label_source"] = "rule"
            return record

        llm_label = classify_cc_unclear_with_llm(
            permissions_text=permissions_text,
            cache=llm_cache,
            cache_lock=llm_cache_lock,
        )
        final_label = combine_rule_and_llm_label(
            rule_label=rule_label,
            llm_label=llm_label,
            permissions_text=permissions_text,
        )

        record["llm_cc_family_label"] = llm_label
        record["license_label"] = final_label
        record["license_label_source"] = "llm"
        return record

    record["license_label"] = rule_label
    record["license_label_source"] = "rule"
    return record


def main() -> None:
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    if MAX_FILES is not None:
        xml_files = xml_files[:MAX_FILES]

    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {INPUT_DIR}")

    llm_enabled = True
    try:
        get_client()
    except Exception as e:
        llm_enabled = False
        print(f"[WARN] LLM client unavailable: {e}")
        print("[WARN] CREATIVE_COMMONS_UNCLEAR records will remain unresolved.")

    per_file, llm_cache = load_existing_output(OUTPUT_JSON)
    llm_cache_lock = threading.Lock()
    processed_files = {
        str(record.get("file", ""))
        for record in per_file
        if str(record.get("file", "")).strip()
    }

    if processed_files:
        print(
            f"Loaded existing progress for {len(processed_files)} file(s) from {OUTPUT_JSON}"
        )

    remaining_files = [xml_file for xml_file in xml_files if xml_file.name not in processed_files]
    if not remaining_files:
        print("All files already processed. Rebuilding output summary.")
    else:
        print(
            f"Processing {len(remaining_files)} remaining file(s) "
            f"with {MAX_WORKERS} worker(s)"
        )

    total_files = len(xml_files)
    completed_since_save = 0
    max_pending = max(MAX_WORKERS * MAX_PENDING_MULTIPLIER, MAX_WORKERS)

    def submit_next_file(
        executor: ThreadPoolExecutor,
        pending: dict[Future[dict], Path],
        next_index: int,
    ) -> int:
        while next_index < len(remaining_files) and len(pending) < max_pending:
            xml_path = remaining_files[next_index]
            future = executor.submit(
                analyze_file,
                xml_path,
                llm_enabled,
                llm_cache,
                llm_cache_lock,
            )
            pending[future] = xml_path
            next_index += 1
        return next_index

    pending: dict[Future[dict], Path] = {}
    next_index = 0

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            next_index = submit_next_file(executor, pending, next_index)

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    xml_path = pending.pop(future)
                    rec = future.result()
                    per_file.append(rec)
                    processed_files.add(xml_path.name)
                    completed_since_save += 1

                    if (
                        completed_since_save >= CHECKPOINT_EVERY
                        or len(processed_files) == total_files
                    ):
                        save_output(
                            output_path=OUTPUT_JSON,
                            xml_files=xml_files,
                            per_file=per_file,
                            llm_cache=llm_cache,
                        )
                        completed_since_save = 0

                    if (
                        len(processed_files) % PROGRESS_EVERY == 0
                        or len(processed_files) == total_files
                    ):
                        print(
                            f"Processed {len(processed_files)}/{total_files} | "
                            f"remaining={total_files - len(processed_files)} | "
                            f"unique_llm_cache={len(llm_cache)}"
                        )

                next_index = submit_next_file(executor, pending, next_index)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted. Saving current progress before exit...")
        save_output(
            output_path=OUTPUT_JSON,
            xml_files=xml_files,
            per_file=per_file,
            llm_cache=llm_cache,
        )
        raise

    if completed_since_save > 0 or not OUTPUT_JSON.exists():
        save_output(
            output_path=OUTPUT_JSON,
            xml_files=xml_files,
            per_file=per_file,
            llm_cache=llm_cache,
        )

    output = build_output_payload(
        xml_files=xml_files,
        per_file=per_file,
        llm_cache=llm_cache,
    )

    save_output(
        output_path=OUTPUT_JSON,
        xml_files=xml_files,
        per_file=per_file,
        llm_cache=llm_cache,
    )

    print()
    print(f"Done. Wrote JSON to: {OUTPUT_JSON}")
    print("License counts:")
    for label, count in output["license_counts"].items():
        print(f"{label} = {count}")


if __name__ == "__main__":
    main()
