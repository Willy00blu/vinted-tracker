"""Controlled comparison of de-mixing strategies on the subset, WITH inspection
so we judge correctness (not just OBJ). Compares: default, v1 (no de-mix),
mixture_split conservative (ratio 3.0), +ejection (user's idea), and the
aggressive search winner (ratio 2.5).
"""
import sys, os, json, re, collections, statistics as st
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl, ab_lab as L
from bench import benchmark
from ab_search import KNOBS, DEFAULTS

items, emb, img = L.load_subset()
raw = L.raw_partition(items, emb)
V1 = {"DISABLE_BITAIL": True, "STABLE_MERGE_KEY": True, "TUNE_JENKS_COS": 0.97,
      "TUNE_SIBLING_RATIO": 1.5, "TUNE_SIBLING_COS": 0.985, "TUNE_LEXSIG_COS": 0.96,
      "TUNE_LEXSIG_RATIO": 1.25, "RESCUE_GATE_HI": 0.92}


def setflags(d):
    for k in KNOBS:
        setattr(cl, k, DEFAULTS[k])
    for k, v in d.items():
        setattr(cl, k, v)


def run(name, cfg):
    setflags(cfg)
    m = L.refine(items, raw, emb, img)
    b = benchmark(items, m, emb)
    print(f"{name:38} OBJ={b['OBJ']:.3f} ABQ={b['ABQ']:.3f} HARM={b['HARM']:.3f} "
          f"MASK={b['MASK']:.3f} cov={b['coverage']:4.1f}% mix={b['mix_pct']:4.1f}% "
          f"nclu={b['n_clusters']}")
    return m


# bi-tail stays ON (honest benchmark showed DISABLE_BITAIL → mix explodes)
CAND = {"STABLE_MERGE_KEY": True, "TUNE_JENKS_COS": 0.97}
print(f"subset {len(items)} items\n")
run("DEFAULT (production-like)", {})
run("CAND + mixture_split @3.0", {**CAND, "ENABLE_MIXTURE_SPLIT": True, "TUNE_MIX_RATIO": 3.0})
run("CAND + mixture_split @2.5", {**CAND, "ENABLE_MIXTURE_SPLIT": True, "TUNE_MIX_RATIO": 2.5})
mR = run("CAND + mix@3.0 + eject(floor.90) (REC)",
         {**CAND, "ENABLE_MIXTURE_SPLIT": True, "TUNE_MIX_RATIO": 3.0,
          "ENABLE_OUTLIER_EJECT": True})
run("search4 winner (mix@2.5)",
    {"STABLE_MERGE_KEY": True, "ENABLE_MIXTURE_SPLIT": True, "TUNE_MIX_RATIO": 2.5,
     "TUNE_JENKS_RATIO": 2.5, "TUNE_JENKS_COS": 0.92})

# ---- INSPECTION of the recommended config ----
print("\n=== INSPECTION: recommended config ===")
by = collections.defaultdict(list)
for it in items:
    by[mR[it["id"]]].append(it)
base = lambda k: re.sub(r' \[€[^\]]*\]| #out\d+$', '', k)
fam = collections.defaultdict(list)
for k, v in by.items():
    fam[base(k)].append((k, v))
splits = [(b, ps) for b, ps in fam.items() if len(ps) >= 2]
print(f"\n{len(splits)} base-clusters split/ejected. 6 examples (verify correctness):")
for b, ps in sorted(splits, key=lambda x: -sum(len(v) for _, v in x[1]))[:6]:
    ps = sorted(ps, key=lambda kv: st.median([it['price'] for it in kv[1] if it.get('price')] or [0]))
    print(f"■ {b[:44]}")
    for k, v in ps:
        pr = [it['price'] for it in v if it.get('price')]
        med = st.median(pr) if pr else 0
        tag = "OUT" if "#out" in k else f"€{med:.0f}"
        print(f"   {tag:>6} n={len(v):3d}: " +
              " | ".join((it.get('title') or '')[:22] for it in v[:3]))
