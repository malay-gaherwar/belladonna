"""Targeted retry: classify ONLY the 10 known-failed factoid files."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import add_source_hierarchy as ash

from openai import AsyncOpenAI


FAILED: list[tuple[str, str]] = [
    ("EPMC",     "PMC12740911"),
    ("EPMC",     "PMC5941327"),
    ("EPMC",     "PMC6472763"),
    ("EPMC",     "PMC6604336"),
    ("EPMC",     "PMC7534806"),
    ("EPMC",     "PMC8495573"),
    ("EPMC",     "PMC8531794"),
    ("EPMC",     "PMC9026406"),
    ("EPMC",     "PMC9809245"),
    ("Elsevier", "85070501004"),
]


def factoid_path_for(source: str, doc_id: str) -> Path:
    cfg = ash.SOURCES[source]
    for rel in cfg["paths"]:
        d = Path(rel)
        if d.exists() and d.is_dir():
            p = d / f"{doc_id}_factoids.json"
            if p.exists():
                return p
    raise FileNotFoundError(f"No factoid file for {source} {doc_id}")


async def process(client: AsyncOpenAI, source: str, doc_id: str) -> None:
    cfg = ash.SOURCES[source]
    xml_dirs = ash.resolve_xml_dirs(cfg, Path.cwd())
    xml_path = ash.stem_to_xml_path(doc_id, xml_dirs)
    factoid_path = factoid_path_for(source, doc_id)

    data = ash.load_json(factoid_path)

    if ash.already_labelled(data):
        print(f"[{source} {doc_id}] already labelled -> skip")
        return

    used_xml = False
    if xml_path is not None:
        try:
            signals = ash.extract_signals_from_xml(xml_path, cfg["xml_kind"])
            used_xml = True
        except Exception as e:
            print(f"[{source} {doc_id}] xml parse failed: {e} -> fallback")
            signals = ash.signals_from_factoid_json(data)
    else:
        signals = ash.signals_from_factoid_json(data)

    level = await ash.classify_one(client, signals)
    if level is None:
        print(f"[{source} {doc_id}] CLASSIFIER STILL FAILS")
        return

    ash.inject_hierarchy(data, level)
    ash.write_json_atomic(factoid_path, data)
    print(f"[{source} {doc_id}] level={level} ({ash.HIERARCHY_LABELS[level]}) "
          f"used_xml={used_xml}")


async def main() -> int:
    api_key = os.environ.get("VIRTUAL_API_KEY")
    base_url = os.environ.get("BASE_URL")
    if not api_key or not base_url:
        print("VIRTUAL_API_KEY / BASE_URL not set", file=sys.stderr)
        return 1

    print(f"MAX_COMPLETION_TOKENS = {ash.MAX_COMPLETION_TOKENS}")
    print(f"MODEL_NAME            = {ash.MODEL_NAME}")
    print(f"target files          = {len(FAILED)}")
    print()

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    await asyncio.gather(*(process(client, s, d) for s, d in FAILED))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
