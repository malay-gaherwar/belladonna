#!/usr/bin/env python3
"""AGO pipeline agent.

Runs the full AGO pipeline end to end and verifies that every stage of the
PDF -> processed markdown -> factoids -> embeddings flow actually produced
valid output.

Stages (each is an existing script in scripts/AGO/):
  1. process    processAGO.py      downloaded/*.pdf  -> processed/*.md
  2. factoids   factoids_AGO.py    processed/*.md    -> factoids/*_factoids.json
  3. embeddings ago_embedding.py   factoids/*.json   -> embeddings/ (Chroma)

By default a stage is skipped when its output is already complete; pass
--force to re-run everything, or --validate-only to skip running entirely.
The agent exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

# Project root = two levels up from scripts/AGO/ago_agent.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOWNLOADED_DIR = PROJECT_ROOT / "artifacts/AGO/downloaded"
PROCESSED_DIR = PROJECT_ROOT / "artifacts/AGO/processed"
FACTOIDS_DIR = PROJECT_ROOT / "artifacts/AGO/factoids"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts/AGO/embeddings"

PROCESS_SCRIPT = PROJECT_ROOT / "scripts/AGO/processAGO.py"
FACTOIDS_SCRIPT = PROJECT_ROOT / "scripts/AGO/factoids_AGO.py"
EMBEDDING_SCRIPT = PROJECT_ROOT / "scripts/AGO/ago_embedding.py"

# Mirrors ago_embedding.py
CHROMA_COLLECTION = "ago_factoids_qwen_embeddings"

EXPECTED_LICENSE_INFO = {
    "copyright": "German Copyright law",
    "commercial_use": "no",
    "personal_use": "yes",
}


# ============================================================
# PYDANTIC MODELS
# ============================================================

class LicenseInfo(BaseModel):
    copyright: str
    commercial_use: str
    personal_use: str


class FactoidMetadata(BaseModel):
    source_family: str
    document_title: str
    document_type: str
    document_year: Optional[int] = None
    file_name: str
    license_info: LicenseInfo


class Factoid(BaseModel):
    id: int
    factoid_text: str = Field(min_length=1)


class FactoidFile(BaseModel):
    metadata: FactoidMetadata
    factoids: List[Factoid]


class StageReport(BaseModel):
    name: str
    ran: bool
    skipped_reason: Optional[str] = None
    return_code: Optional[int] = None
    ok: bool
    details: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class PipelineReport(BaseModel):
    started_at: str
    finished_at: Optional[str] = None
    project_root: str
    pdf_count: int = 0
    processed_md_count: int = 0
    factoid_file_count: int = 0
    total_factoids: int = 0
    embedded_count: int = 0
    overall_ok: bool = False
    stages: List[StageReport] = Field(default_factory=list)


# ============================================================
# HELPERS
# ============================================================

class _Tee:
    """Writes to several streams at once, flushing after every write so the
    on-disk log stays current line-by-line."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def setup_log_file() -> Path:
    """Create logs/ago_agent_run_<ts>.log and tee stdout/stderr into it."""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"ago_agent_run_{ts}.log"
    fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, fh)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, fh)  # type: ignore[assignment]
    return log_path


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def run_script(script: Path, stage_prefix: str) -> int:
    """Run a pipeline script as a subprocess, streaming its output line-by-line.

    Each child line is prefixed with [HH:MM:SS][stage] so you can tail -f the
    log and see exactly which stage is doing what.
    """
    log(f"  $ {sys.executable} -u {script.relative_to(PROJECT_ROOT)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"[{_ts()}][{stage_prefix}] {line.rstrip()}\n")
        sys.stdout.flush()
    proc.wait()
    log(f"  {script.name} finished with rc={proc.returncode}")
    return proc.returncode


def pdf_stems() -> List[str]:
    return sorted(p.stem for p in DOWNLOADED_DIR.glob("*.pdf"))


def processed_stems() -> List[str]:
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.md"))


def factoid_files() -> List[Path]:
    return sorted(FACTOIDS_DIR.glob("*_factoids.json"))


def get_process_cap() -> Optional[int]:
    """Read MAX_FILES from processAGO.py without executing its main()."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts/AGO"))
        import processAGO  # type: ignore

        return processAGO.MAX_FILES
    except Exception:
        return None
    finally:
        if str(PROJECT_ROOT / "scripts/AGO") in sys.path:
            sys.path.remove(str(PROJECT_ROOT / "scripts/AGO"))


# ============================================================
# STAGE 1: PROCESS (PDF -> MD)
# ============================================================

def expected_processed_count() -> int:
    pdfs = len(pdf_stems())
    cap = get_process_cap()
    return min(pdfs, cap) if cap is not None else pdfs


def process_complete() -> bool:
    return PROCESSED_DIR.is_dir() and len(processed_stems()) >= expected_processed_count() > 0


def validate_process(report: StageReport) -> None:
    pdfs = pdf_stems()
    mds = set(processed_stems())
    expected = expected_processed_count()

    report.details.append(f"PDFs in downloaded/: {len(pdfs)}")
    report.details.append(f"Markdown in processed/: {len(mds)} (expected >= {expected})")

    if not PROCESSED_DIR.is_dir():
        report.ok = False
        report.errors.append(f"processed dir missing: {PROCESSED_DIR}")
        return

    # Which of the processed-eligible PDFs are missing a markdown file.
    eligible = pdfs[:expected] if expected else pdfs
    missing = [s for s in eligible if s not in mds]

    empty = [p.name for p in PROCESSED_DIR.glob("*.md") if p.stat().st_size == 0]
    no_pages = [
        p.name
        for p in PROCESSED_DIR.glob("*.md")
        if "<!-- PAGE 1 START -->" not in p.read_text(encoding="utf-8", errors="ignore")
    ]

    if missing:
        report.ok = False
        report.errors.append(
            f"{len(missing)} eligible PDF(s) have no markdown: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    if empty:
        report.ok = False
        report.errors.append(f"empty markdown file(s): {empty}")
    if no_pages:
        report.ok = False
        report.errors.append(f"markdown without page markers: {no_pages}")

    if len(pdfs) > expected:
        report.details.append(
            f"NOTE: {len(pdfs)} PDFs but processAGO cap is {expected}; "
            f"{len(pdfs) - expected} PDF(s) are intentionally not processed."
        )


# ============================================================
# STAGE 2: FACTOIDS (MD -> JSON)
# ============================================================

def factoids_complete() -> bool:
    if not FACTOIDS_DIR.is_dir():
        return False
    produced = {p.name.replace("_factoids.json", "") for p in factoid_files()}
    return bool(processed_stems()) and all(s in produced for s in processed_stems())


def validate_factoids(report: StageReport) -> int:
    """Validate every factoid JSON with pydantic. Returns total factoid count."""
    files = factoid_files()
    report.details.append(f"Factoid JSON files: {len(files)}")

    if not FACTOIDS_DIR.is_dir():
        report.ok = False
        report.errors.append(f"factoids dir missing: {FACTOIDS_DIR}")
        return 0

    produced = {p.name.replace("_factoids.json", "") for p in files}
    missing = [s for s in processed_stems() if s not in produced]
    if missing:
        report.ok = False
        report.errors.append(
            f"{len(missing)} processed md without factoids json: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    total_factoids = 0
    bad_schema: List[str] = []
    empty_factoids: List[str] = []
    bad_license: List[str] = []

    for fp in files:
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
            model = FactoidFile.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            bad_schema.append(f"{fp.name}: {str(e).splitlines()[0]}")
            continue

        if len(model.factoids) == 0:
            empty_factoids.append(fp.name)

        li = model.metadata.license_info.model_dump()
        if li != EXPECTED_LICENSE_INFO:
            bad_license.append(f"{fp.name}: {li}")

        total_factoids += len(model.factoids)

    report.details.append(f"Total factoids across files: {total_factoids}")

    if bad_schema:
        report.ok = False
        report.errors.append(f"schema-invalid file(s): {bad_schema[:5]}")
    if empty_factoids:
        report.ok = False
        report.errors.append(f"file(s) with zero factoids: {empty_factoids}")
    if bad_license:
        report.ok = False
        report.errors.append(
            f"file(s) with wrong/missing license_info: {bad_license[:5]}"
        )
    else:
        report.details.append("license_info present and correct in all files")

    return total_factoids


# ============================================================
# STAGE 3: EMBEDDINGS (JSON -> CHROMA)
# ============================================================

def embeddings_complete(total_factoids: int) -> bool:
    if not EMBEDDINGS_DIR.is_dir() or total_factoids == 0:
        return False
    try:
        return chroma_count() >= total_factoids
    except Exception:
        return False


def chroma_count() -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION, embedding_function=None
    )
    return collection.count()


def validate_embeddings(report: StageReport, total_factoids: int) -> int:
    if not EMBEDDINGS_DIR.is_dir():
        report.ok = False
        report.errors.append(f"embeddings dir missing: {EMBEDDINGS_DIR}")
        return 0

    try:
        count = chroma_count()
    except Exception as e:
        report.ok = False
        report.errors.append(f"could not open Chroma collection: {e}")
        return 0

    report.details.append(
        f"Chroma collection '{CHROMA_COLLECTION}' count: {count} "
        f"(expected >= {total_factoids})"
    )

    if total_factoids == 0:
        report.ok = False
        report.errors.append("no factoids to embed (upstream stage produced nothing)")
    elif count < total_factoids:
        report.ok = False
        report.errors.append(
            f"embedded {count} < {total_factoids} factoids — embeddings incomplete"
        )
    return count


# ============================================================
# ORCHESTRATION
# ============================================================

def need_api_keys() -> Optional[str]:
    import os

    missing = [k for k in ("VIRTUAL_API_KEY", "BASE_URL") if not os.getenv(k)]
    if missing:
        return (
            f"Missing env var(s): {missing}. "
            f"Run inside a shell that sourced ~/.bashrc, or export them first."
        )
    return None


def run_stage(
    name: str,
    script: Path,
    is_complete,
    validate_fn,
    *,
    force: bool,
    validate_only: bool,
    needs_keys: bool,
) -> StageReport:
    log(f"\n=== STAGE: {name} ===")
    sr = StageReport(name=name, ran=False, ok=True)

    already = is_complete()
    if validate_only:
        sr.skipped_reason = "validate-only mode"
    elif already and not force:
        sr.skipped_reason = "output already complete (use --force to re-run)"
    else:
        if needs_keys:
            key_err = need_api_keys()
            if key_err:
                sr.ok = False
                sr.errors.append(key_err)
                log(f"  ABORT: {key_err}")
                return sr
        sr.ran = True
        rc = run_script(script, stage_prefix=name)
        sr.return_code = rc
        if rc != 0:
            sr.ok = False
            sr.errors.append(f"{script.name} exited with code {rc}")

    if sr.skipped_reason:
        log(f"  (skipped: {sr.skipped_reason})")

    validate_fn(sr)
    status = "OK" if sr.ok else "FAIL"
    log(f"  -> {name}: {status}")
    for d in sr.details:
        log(f"     - {d}")
    for e in sr.errors:
        log(f"     ! {e}")
    return sr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and verify the AGO pipeline.")
    parser.add_argument(
        "--force", action="store_true", help="re-run stages even if output exists"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="do not run any stage, only validate existing output",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=["process", "factoids", "embeddings"],
        default="process",
        help="start the pipeline from this stage",
    )
    args = parser.parse_args()

    log_path = setup_log_file()

    order = ["process", "factoids", "embeddings"]
    start_idx = order.index(args.from_stage)

    report = PipelineReport(
        started_at=datetime.datetime.now().isoformat(timespec="seconds"),
        project_root=str(PROJECT_ROOT),
    )

    log(f"AGO agent — project root: {PROJECT_ROOT}")
    log(f"Live log:                 {log_path}")
    log(f"Stages to consider: {order[start_idx:]}")

    total_factoids = 0

    # Stage 1
    if start_idx <= 0:
        sr = run_stage(
            "process",
            PROCESS_SCRIPT,
            process_complete,
            validate_process,
            force=args.force,
            validate_only=args.validate_only,
            needs_keys=True,
        )
        report.stages.append(sr)

    # Stage 2
    if start_idx <= 1:
        sr = run_stage(
            "factoids",
            FACTOIDS_SCRIPT,
            factoids_complete,
            lambda r: validate_factoids(r),
            force=args.force,
            validate_only=args.validate_only,
            needs_keys=True,
        )
        report.stages.append(sr)

    total_factoids = validate_factoids(StageReport(name="_count", ran=False, ok=True))

    # Stage 3
    if start_idx <= 2:
        sr = run_stage(
            "embeddings",
            EMBEDDING_SCRIPT,
            lambda: embeddings_complete(total_factoids),
            lambda r: validate_embeddings(r, total_factoids),
            force=args.force,
            validate_only=args.validate_only,
            needs_keys=True,
        )
        report.stages.append(sr)
        report.embedded_count = validate_embeddings(
            StageReport(name="_count", ran=False, ok=True), total_factoids
        )

    # Summary numbers
    report.pdf_count = len(pdf_stems())
    report.processed_md_count = len(processed_stems())
    report.factoid_file_count = len(factoid_files())
    report.total_factoids = total_factoids
    report.overall_ok = all(s.ok for s in report.stages)
    report.finished_at = datetime.datetime.now().isoformat(timespec="seconds")

    # Persist report
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = logs_dir / f"ago_agent_report_{ts}.json"
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log("\n================ SUMMARY ================")
    log(f"PDFs:            {report.pdf_count}")
    log(f"Processed md:    {report.processed_md_count}")
    log(f"Factoid files:   {report.factoid_file_count}")
    log(f"Total factoids:  {report.total_factoids}")
    log(f"Embedded:        {report.embedded_count}")
    for s in report.stages:
        log(f"Stage {s.name:<10} {'OK' if s.ok else 'FAIL'}")
    log(f"OVERALL: {'OK — PDF->embeddings verified' if report.overall_ok else 'FAIL'}")
    log(f"Report written: {report_path}")

    return 0 if report.overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
