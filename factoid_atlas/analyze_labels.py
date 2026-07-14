#!/usr/bin/env python3
"""Quick QA of a labels.jsonl against the factoids it labelled.

Prints per-dimension distributions, per-source 'none' coverage, the drug-name
dictionary agreement (LLM drug_class vs bc_drugs_reference seeds), and a few
sample rows per dimension so we can eyeball quality.
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import taxonomy as tax


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factoids", default="sample.jsonl")
    ap.add_argument("--labels", default="sample.labels.jsonl")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    fac = {f["id"]: f for f in (json.loads(l) for l in open(args.factoids, encoding="utf-8") if l.strip())}
    lab = load(args.labels)
    ids = [i for i in fac if i in lab]
    N = len(ids)
    print(f"factoids={len(fac)} labels={len(lab)} joined={N}\n")

    names = {d: {e["code"]: e["name"] for e in tax.TAXONOMY[d]} for d in tax.DIMENSIONS}

    # per-dimension distribution
    for d in tax.DIMENSIONS:
        c = Counter(lab[i][d] for i in ids)
        print(f"== {d} ==")
        for code, _ in [(e["code"], 0) for e in tax.TAXONOMY[d]]:
            n = c.get(code, 0)
            bar = "█" * round(40 * n / max(N, 1))
            print(f"  {names[d][code]:26s} {n:5d} {100*n/N:5.1f}%  {bar}")
        print()

    # per-source 'none' rate for the entity dims (lower = better coverage)
    print("== coverage by source (% NOT 'none') ==")
    bysrc = defaultdict(lambda: Counter())
    for i in ids:
        s = fac[i]["source"]; bysrc[s]["n"] += 1
        for d in ("drug_class", "biomarker", "setting"):
            if lab[i][d] != "none":
                bysrc[s][d] += 1
    hdr = f"  {'source':10s} {'n':>5s} {'drug%':>7s} {'biom%':>7s} {'set%':>7s}"
    print(hdr)
    for s in sorted(bysrc):
        n = bysrc[s]["n"]
        print(f"  {s:10s} {n:5d} {100*bysrc[s]['drug_class']/n:6.0f}% "
              f"{100*bysrc[s]['biomarker']/n:6.0f}% {100*bysrc[s]['setting']/n:6.0f}%")
    print()

    # dictionary agreement: when a factoid mentions a known drug name, does the
    # LLM's drug_class match the bc_drugs_reference bucket?
    seeds = tax.DRUG_NAME_SEEDS
    pat = {name: re.compile(r"\b" + re.escape(name) + r"\b", re.I) for name in seeds}
    agree = miss = total = 0
    examples = []
    for i in ids:
        txt = fac[i]["text"].lower()
        hit = None
        for name, bucket in seeds.items():
            if pat[name].search(txt):
                hit = (name, bucket); break
        if not hit:
            continue
        total += 1
        name, bucket = hit
        got = lab[i]["drug_class"]
        if got == bucket:
            agree += 1
        else:
            miss += 1
            if len(examples) < 8:
                examples.append((name, bucket, got, fac[i]["text"][:90]))
    if total:
        print(f"== drug dictionary check (factoids mentioning a known drug) ==")
        print(f"  matched {total}; LLM agrees with bc_drugs_reference bucket: "
              f"{agree} ({100*agree/total:.0f}%), differs: {miss}")
        for name, bucket, got, t in examples:
            print(f"   '{name}' seed={bucket} llm={got} :: {t}")
        print()

    # sample rows per dimension (non-none) so we can eyeball
    print("== sample labelled factoids ==")
    seen = defaultdict(int)
    for i in ids:
        l = lab[i]
        key = (l["drug_class"], l["biomarker"], l["setting"], l["evidence"])
        if seen[key] >= 1:
            continue
        seen[key] += 1
        if sum(1 for v in seen.values()) > 22:
            break
        tags = f"{l['drug_class']}/{l['biomarker']}/{l['setting']}/{l['evidence']}"
        print(f"  [{fac[i]['source']:8s}] {tags}\n      {fac[i]['text'][:140]}")


if __name__ == "__main__":
    main()
