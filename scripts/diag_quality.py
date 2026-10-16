"""Diagnose what drags specpure / unimodal on the BEST-model mapping, to find
the next high-leverage split. Prints the worst clusters with their content.
"""
import sys, os, json, collections, statistics as st, math, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
import ab_lab as L
from ab_search import KNOBS, DEFAULTS

CACHE = ROOT / "data" / "ab_cache"
best = json.load(open(CACHE / "best_model.json"))["full_config"]
for k in KNOBS:
    setattr(cl, k, best.get(k, DEFAULTS[k]))

items, emb, img = L.load_subset()
raw = L.raw_partition(items, emb)
m = L.refine(items, raw, emb, img)
by = collections.defaultdict(list)
for it in items:
    by[m[it["id"]]].append(it)
by = {k: v for k, v in by.items() if len(v) >= 12}

def specpure(v):
    nums_per = [cl._lex_sig(it.get("title",""), it.get("brand"))[1] for it in v]
    flat = [n for nums in nums_per for n in nums]
    if not flat: return 1.0, None
    dom, c = collections.Counter(flat).most_common(1)[0]
    return sum(1 for nums in nums_per if dom in nums)/len(v), dom

# worst specpure clusters
rows = []
for k, v in by.items():
    sp, dom = specpure(v)
    rows.append((sp, len(v), k, dom, v))
rows.sort()
print("=== 12 WORST specpure clusters (size>=12) ===")
for sp, n, k, dom, v in rows[:12]:
    toks = collections.Counter()
    for it in v:
        _, nums = cl._lex_sig(it.get("title",""), it.get("brand"))
        toks.update(nums)
    top = ", ".join(f"{t}×{c}" for t, c in toks.most_common(6))
    titles = " | ".join((it.get("title") or "")[:30] for it in v[:3])
    print(f"  specpure={sp:.2f} n={n}  key={k[:32]}")
    print(f"     nums: {top}")
    print(f"     ex: {titles}")

# how much specpure could a model-number split recover?
print("\n=== model-number split potential (split by dominant numeric token) ===")
gain = 0; splits = 0
for k, v in by.items():
    nums_per = [cl._lex_sig(it.get("title",""), it.get("brand"))[1] for it in v]
    # group by the single most-frequent num each item carries
    sigs = collections.Counter()
    for nums in nums_per:
        if nums:
            sigs.update([max(nums, key=lambda t:(sum(c.isdigit() for c in t), len(t)))])
    big = [t for t,c in sigs.items() if c>=4]
    if len(big) >= 2:
        splits += 1
        gain += len(v)
print(f"  {splits} clusters (>=12) have >=2 distinct dominant model-nums (>=4 items each)")
print(f"  → {gain} items ({100*gain/sum(len(v) for v in by.values()):.0f}% of big-cluster items) could be model-split")
