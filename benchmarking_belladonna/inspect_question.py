#!/usr/bin/env python3
"""Re-query the RAG for one benchmark question and dump the FULL picture:
the question, the options (gold/predicted marked), every retrieved evidence
factoid (with its source), and the backing model's grounded reasoning.

This is the "why did it answer that?" tool — it shows exactly what AGO/ESMO
(or whatever was routed) factoids the model reasoned over.

    python3 inspect_question.py 142
    python3 inspect_question.py 142 --model GPT-OSS-120B
"""
import argparse, json, sys, textwrap, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bench"))
from run_rag import load_questions, present_question, parse_answer, DEFAULT_QUESTIONS, SEED  # noqa
import random

RAG_URL = "http://127.0.0.1:8001/query"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--model", default="GPT-OSS-120B")
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    questions = load_questions(DEFAULT_QUESTIONS)
    rng = random.Random(SEED)
    pres_all = {str(q["id"]): present_question(q, False, rng) for q in questions}
    q = next(q for q in questions if str(q["id"]) == str(args.qid))
    pres = pres_all[str(args.qid)]

    payload = {"question": pres["prompt"], "top_k": args.top_k, "model": args.model}
    req = urllib.request.Request(RAG_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)

    predicted = parse_answer(resp.get("answer", ""), pres["num_options"])
    W = 100
    print("═" * W)
    print(f"Q{args.qid}  [{q.get('category')}]   model={args.model}")
    print(f"routing={resp.get('routing_method')}  sources={resp.get('routed_sources')}")
    verdict = resp.get("evidence_verdict") or {}
    print(f"evidence_sufficient={verdict.get('sufficient')}  "
          f"(critic notes: {verdict.get('notes','')[:120]})")
    print(f"PREDICTED={predicted}   GOLD={pres['gold']}   "
          f"{'✓ correct' if predicted==pres['gold'] else '✗ WRONG'}")
    print("═" * W)
    print("\nQUESTION:\n" + textwrap.fill(q["question"], W, initial_indent="  ", subsequent_indent="  "))
    print("\nOPTIONS:")
    for k in sorted(pres["label_map"]):
        txt = q["options"][pres["label_map"][k]]
        mark = "  <== GOLD" if k == pres["gold"] else ("  <== model picked" if k == predicted else "")
        print(textwrap.fill(f"{k}) {txt}{mark}", W, subsequent_indent="     "))

    print(f"\nRETRIEVED EVIDENCE ({len(resp.get('evidence', []))} factoids):")
    for i, e in enumerate(resp.get("evidence", []), 1):
        src = e.get("source", "?")
        cite = e.get("citation_label") or f"{src} {e.get('document_year','')}"
        title = (e.get("display_title") or e.get("file_name") or "").strip()[:80]
        print(f"\n  [{i}] {src}  ({cite})  {title}")
        body = (e.get("factoid_text") or "").strip()
        print(textwrap.fill(body, W, initial_indent="      ", subsequent_indent="      "))

    print("\n" + "─" * W)
    print("MODEL GROUNDED ANSWER (its reasoning + [[citations]]):\n")
    print(resp.get("answer", "(empty)"))
    print("═" * W)


if __name__ == "__main__":
    main()
