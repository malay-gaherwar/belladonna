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


INPUT_DIR = Path("artifacts/Elsevier/filtered_xml")
OUTPUT_JSON = Path("artifacts/Elsevier/license_summary.json")

# Set to an integer like 500 for testing.
# Set to None to run on all files.
MAX_FILES = None

MODEL_NAME = "Qwen3.5-397B-A17B-FP8"
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


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def attr_local_name(attr_name: str) -> str:
    return attr_name.rsplit("}", 1)[-1].lower()


def elem_attr(elem: ET.Element, names: list[str]) -> str:
    wanted = {name.lower() for name in names}
    for key, value in elem.attrib.items():
        if attr_local_name(key) in wanted:
            return clean_text(value)
    return ""


def normalize_boolish(values: list[str]) -> str:
    joined = " ".join(str(v).strip().lower() for v in values if str(v).strip())

    if not joined:
        return ""

    if re.search(r"\b(true|yes|y|1|full)\b", joined):
        return "true"

    if re.search(r"\b(false|no|n|0|none)\b", joined):
        return "false"

    return ""


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def extract_elsevier_article_ids(root: ET.Element) -> dict[str, str]:
    result = {
        "doi": "",
        "title": "",
    }

    for elem in root.iter():
        name = local_name(elem.tag)
        text = collect_text(elem)

        if not text:
            continue

        if name == "doi" and not result["doi"]:
            result["doi"] = text

        elif name in {"title", "article-title"} and not result["title"]:
            result["title"] = text

    # Some Elsevier XMLs may carry DOI in an attribute.
    for elem in root.iter():
        for key, value in elem.attrib.items():
            k = attr_local_name(key)
            v = clean_text(value)

            if k == "doi" and v and not result["doi"]:
                result["doi"] = v

    return result


def extract_elsevier_license_fields(root: ET.Element) -> dict[str, object]:
    """
    Extract Elsevier-specific license / access fields.

    Important Elsevier signals:
    - openaccessArticle
    - openaccess
    - openaccessType
    - openaccessUserLicense
    - oa-user-license
    - sa-user-license
    - openArchiveArticle
    - license / license-text
    - copyright / copyright-statement
    """

    text_fields: dict[str, list[str]] = {
        "openaccessArticle": [],
        "openaccess": [],
        "openaccessType": [],
        "openaccessUserLicense": [],
        "oa-user-license": [],
        "sa-user-license": [],
        "openArchiveArticle": [],
        "license_text": [],
        "copyright": [],
        "copyright_statement": [],
        "copyright_holder": [],
        "self_archiving": [],
        "sponsor_type": [],
    }

    local_name_to_field = {
        "openaccessarticle": "openaccessArticle",
        "openaccess": "openaccess",
        "openaccesstype": "openaccessType",
        "openaccessuserlicense": "openaccessUserLicense",
        "oa-user-license": "oa-user-license",
        "sa-user-license": "sa-user-license",
        "openarchivearticle": "openArchiveArticle",
        "license": "license_text",
        "license-text": "license_text",
        "copyright": "copyright",
        "copyright-statement": "copyright_statement",
        "copyright-holder": "copyright_holder",
        "self-archiving": "self_archiving",
        "selfarchiving": "self_archiving",
        "sponsortype": "sponsor_type",
        "sponsor-type": "sponsor_type",
    }

    for elem in root.iter():
        name = local_name(elem.tag)
        field = local_name_to_field.get(name)

        if field:
            txt = collect_text(elem)
            if txt:
                text_fields[field].append(txt)

        # License URLs often live in href / xlink:href attributes.
        if name == "license":
            href = elem_attr(elem, ["href", "xlink:href"])
            if href:
                text_fields["license_text"].append(href)

        for key, value in elem.attrib.items():
            k = attr_local_name(key)
            v = clean_text(value)

            if not v:
                continue

            if k in {"openaccessarticle", "openaccess"}:
                text_fields["openaccessArticle"].append(v)

            elif k == "openaccesstype":
                text_fields["openaccessType"].append(v)

            elif k == "openaccessuserlicense":
                text_fields["openaccessUserLicense"].append(v)

            elif k == "oa-user-license":
                text_fields["oa-user-license"].append(v)

            elif k == "sa-user-license":
                text_fields["sa-user-license"].append(v)

            elif k == "openarchivearticle":
                text_fields["openArchiveArticle"].append(v)

            elif k in {"href", "xlink:href", "license_ref", "license-ref"}:
                if (
                    "creativecommons.org/" in v.lower()
                    or "elsevier.com/open-access/userlicense" in v.lower()
                ):
                    text_fields["license_text"].append(v)

    compact_fields = {
        key: sorted(set(values))
        for key, values in text_fields.items()
        if values
    }

    raw_parts: list[str] = []
    for values in compact_fields.values():
        raw_parts.extend(values)

    raw_license_text = clean_text(" ".join(raw_parts))

    return {
        "fields": compact_fields,
        "raw_license_text": raw_license_text,
    }


def detect_supported_cc_version(text: str) -> str | None:
    t = text.lower()

    patterns = [
        r"creativecommons\.org/licenses/by-nc-nd/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nc-sa/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nc/(\d\.\d)",
        r"creativecommons\.org/licenses/by-nd/(\d\.\d)",
        r"creativecommons\.org/licenses/by-sa/(\d\.\d)",
        r"creativecommons\.org/licenses/by/(\d\.\d)",
        r"\bcc[- ]by[- ]nc[- ]nd\s+(\d\.\d)\b",
        r"\bcc[- ]by[- ]nc[- ]sa\s+(\d\.\d)\b",
        r"\bcc[- ]by[- ]nc\s+(\d\.\d)\b",
        r"\bcc[- ]by[- ]nd\s+(\d\.\d)\b",
        r"\bcc[- ]by[- ]sa\s+(\d\.\d)\b",
        r"\bcc[- ]by\s+(\d\.\d)\b",
        r"creative commons attribution-noncommercial-noderivs\s+(\d\.\d)",
        r"creative commons attribution-noncommercial-no derivatives\s+(\d\.\d)",
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


def detect_cc_license_rule(text: str) -> str:
    t = text.lower()

    cc_patterns = [
        (
            "CC_BY_NC_ND",
            [
                "creativecommons.org/licenses/by-nc-nd/",
                "cc by-nc-nd",
                "cc-by-nc-nd",
                "creative commons attribution-noncommercial-noderivs",
                "creative commons attribution-noncommercial-no derivatives",
                "creative commons attribution non-commercial no-derivatives",
                "creative commons attribution noncommercial noderivatives",
            ],
        ),
        (
            "CC_BY_NC_SA",
            [
                "creativecommons.org/licenses/by-nc-sa/",
                "cc by-nc-sa",
                "cc-by-nc-sa",
                "creative commons attribution-noncommercial-sharealike",
                "creative commons attribution non-commercial sharealike",
            ],
        ),
        (
            "CC_BY_ND",
            [
                "creativecommons.org/licenses/by-nd/",
                "cc by-nd",
                "cc-by-nd",
                "creative commons attribution-noderivs",
                "creative commons attribution-no derivatives",
                "creative commons attribution no-derivatives",
            ],
        ),
        (
            "CC_BY_SA",
            [
                "creativecommons.org/licenses/by-sa/",
                "cc by-sa",
                "cc-by-sa",
                "creative commons attribution-sharealike",
            ],
        ),
        (
            "CC_BY_NC",
            [
                "creativecommons.org/licenses/by-nc/",
                "cc by-nc",
                "cc-by-nc",
                "creative commons attribution-noncommercial",
                "creative commons attribution non-commercial",
            ],
        ),
        (
            "CC_BY",
            [
                "creativecommons.org/licenses/by/",
                "cc by ",
                "cc-by ",
                "creative commons attribution license",
                "creative commons attribution 4.0",
                "creative commons attribution international license",
            ],
        ),
    ]

    family = ""

    for candidate, patterns in cc_patterns:
        if contains_any(t, patterns):
            family = candidate
            break

    if not family:
        if "creativecommons.org/licenses/" in t or "creative commons" in t:
            return "CREATIVE_COMMONS_UNCLEAR"
        return ""

    version = detect_supported_cc_version(text)

    if version:
        return f"{family}_{version.replace('.', '_')}"

    return f"{family}_UNSPECIFIED"


def detect_elsevier_license_rule(
    raw_license_text: str,
    fields: dict[str, list[str]],
) -> str:
    """
    Returns a single clean license label.

    Conservative Elsevier rules:
    - Prefer openaccessUserLicense / oa-user-license for current article license.
    - Do not treat sa-user-license as the current license unless no better information exists.
    - Closed or unknown non-OA articles are labelled NO_OPEN_ACCESS.
    - OA articles without a clear recognized license are labelled OPEN_ACCESS_LICENSE_UNSURE.
    """

    text = raw_license_text.lower()

    oa_license_text = clean_text(
        " ".join(
            fields.get("openaccessUserLicense", [])
            + fields.get("oa-user-license", [])
            + fields.get("license_text", [])
        )
    )

    sa_license_text = clean_text(" ".join(fields.get("sa-user-license", [])))

    openaccess_status = normalize_boolish(
        fields.get("openaccessArticle", []) + fields.get("openaccess", [])
    )

    openarchive_status = normalize_boolish(fields.get("openArchiveArticle", []))

    # 1. Explicit OA CC license.
    explicit_cc = detect_cc_license_rule(oa_license_text)
    if explicit_cc:
        return explicit_cc

    # 2. CC license in general license text, but only if current OA is true.
    broad_cc = detect_cc_license_rule(raw_license_text)
    if broad_cc and openaccess_status == "true":
        return broad_cc

    # 3. Elsevier proprietary OA user license.
    if (
        "elsevier.com/open-access/userlicense" in text
        or "elsevier user license" in text
        or "userlicense/1.0" in text
    ):
        if openaccess_status == "true" or "open access" in text:
            return "ELSEVIER_USER_LICENSE"

    # 4. Other/non-CC OA license or OA with no recognized license.
    # This includes rare cases like Open Government Licence.
    if openaccess_status == "true":
        return "OPEN_ACCESS_LICENSE_UNSURE"

    # 5. Self-archiving license only. This is not the current article license.
    sa_cc = detect_cc_license_rule(sa_license_text)
    if sa_cc:
        return f"SELF_ARCHIVING_{sa_cc}"

    # 6. Non-OA, closed, unknown, or no usable OA signal.
    if (
        "all rights reserved" in text
        or "all rights are reserved" in text
        or "text and data mining" in text
        or "ai training" in text
        or openaccess_status == "false"
        or openarchive_status == "false"
    ):
        return "NO_OPEN_ACCESS"

    # 7. Open archive but no clear current OA license.
    if openarchive_status == "true":
        return "OPEN_ACCESS_LICENSE_UNSURE"

    if "open access" in text:
        return "OPEN_ACCESS_LICENSE_UNSURE"

    if not raw_license_text:
        return "NO_OPEN_ACCESS"

    return "NO_OPEN_ACCESS"


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
        "- Ignore self-archiving licenses unless the text clearly says this is the current article license.\n"
        "- Ignore Elsevier proprietary user licenses unless they explicitly mention a Creative Commons family.\n"
        "- If the text says 'Creative Commons Attribution License' with no version, return CC_BY.\n"
        "- If the text says Attribution-NonCommercial, return CC_BY_NC.\n"
        "- If the text says Attribution-NonCommercial-ShareAlike, return CC_BY_NC_SA.\n"
        "- If the text says Attribution-NonCommercial-NoDerivs, return CC_BY_NC_ND.\n"
        "- If the text says Attribution-NoDerivs, return CC_BY_ND.\n"
        "- If the text says Attribution-ShareAlike, return CC_BY_SA.\n"
        "- If you are not confident it is a Creative Commons article license family, return CC_UNKNOWN.\n"
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


def combine_llm_label_with_version(
    llm_label: str,
    permissions_text: str,
) -> str:
    if llm_label == "CC_UNKNOWN":
        return "CREATIVE_COMMONS_UNCLEAR"

    version = detect_supported_cc_version(permissions_text)

    if version:
        return f"{llm_label}_{version.replace('.', '_')}"

    return f"{llm_label}_UNSPECIFIED"


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
        print("[WARN] Existing output has invalid 'per_file'; starting fresh.")
        return [], {}

    normalized_records: list[dict] = []
    llm_cache: dict[str, str] = {}

    for record in existing_records:
        if not isinstance(record, dict):
            continue

        # Keep only the simplified output fields.
        normalized = {
            "file": str(record.get("file", "")),
            "doi": str(record.get("doi", "")),
            "title": str(record.get("title", "")),
            "license_label": str(record.get("license_label", "")),
            "openaccess_status": str(record.get("openaccess_status", "")),
            "elsevier_license_fields": record.get("elsevier_license_fields", {}),
            "raw_license_text": str(record.get("raw_license_text", "")),
        }

        normalized_records.append(normalized)

    return normalized_records, llm_cache


def build_output_payload(
    xml_files: list[Path],
    per_file: list[dict],
    llm_cache: dict[str, str],
) -> dict:
    counts: Counter[str] = Counter()

    for record in per_file:
        label = str(record.get("license_label", "")).strip()
        if label:
            counts[label] += 1

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


def analyze_file(
    xml_path: Path,
    llm_enabled: bool,
    llm_cache: dict[str, str],
    llm_cache_lock: threading.Lock,
) -> dict:
    record = {
        "file": xml_path.name,
        "doi": "",
        "title": "",
        "license_label": "",
        "openaccess_status": "",
        "elsevier_license_fields": {},
        "raw_license_text": "",
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        record["license_label"] = "PARSE_ERROR"
        record["raw_license_text"] = str(e)
        return record

    ids = extract_elsevier_article_ids(root)
    record.update(ids)

    extracted = extract_elsevier_license_fields(root)
    fields = extracted["fields"]
    raw_license_text = str(extracted["raw_license_text"])

    record["elsevier_license_fields"] = fields
    record["raw_license_text"] = raw_license_text
    record["openaccess_status"] = normalize_boolish(
        fields.get("openaccessArticle", []) + fields.get("openaccess", [])
    )

    license_label = detect_elsevier_license_rule(
        raw_license_text=raw_license_text,
        fields=fields,
    )

    if license_label == "CREATIVE_COMMONS_UNCLEAR" and llm_enabled:
        llm_label = classify_cc_unclear_with_llm(
            permissions_text=raw_license_text,
            cache=llm_cache,
            cache_lock=llm_cache_lock,
        )
        license_label = combine_llm_label_with_version(
            llm_label=llm_label,
            permissions_text=raw_license_text,
        )

    record["license_label"] = license_label

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

    remaining_files = [
        xml_file for xml_file in xml_files
        if xml_file.name not in processed_files
    ]

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

                    try:
                        rec = future.result()
                    except Exception as e:
                        rec = {
                            "file": xml_path.name,
                            "doi": "",
                            "title": "",
                            "license_label": "PROCESSING_ERROR",
                            "openaccess_status": "",
                            "elsevier_license_fields": {},
                            "raw_license_text": str(e),
                        }

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