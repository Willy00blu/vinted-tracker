"""Find price-bimodal MIXTURE clusters (console+game+accessory under one brand
token) in the best-model mapping and explain why they aren't split + whether
image/lexical signals could separate them. Drives the next improvement.
"""
import sys, os, json, math, collections, statistics as st
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
import ab_lab as L
from ab_search import KNOBS, DEFAULTS

best = json.load(open(ROOT / "data/ab_cache/best_model.json"))["full_config"]
for k in KNOBS:
    setattr(cl, k, best.get(k, DEFAULTS[k]))
items, emb, img = L.load_subset()
raw = L.raw_partition(items, emb)
m = L.refine(items, raw, emb, img)
by = collections.defaultdict(list)
for it in items:
    by[m[it["id"]]].append(it)

def norm(v): return v / max(float(np.linalg.norm(v)), 1e-12)

# Find strongly bimodal clusters (Jenks2 ratio high) of decent size.
flagged = []
for k, v in by.items():
    pr = [it["price"] for it in v if it.get("price") and it["price"] > 0]
    if len(v) < 25 or len(pr) < 10:
        continue
    stats = cl._jenks2_stats(pr, min_band=3)
    if not stats:
        continue
    idx, gvf, ratio = stats
    if ratio >= 3.0:                  # clearly two price populations
        sp = sorted([it for it in v if it.get("price") and it["price"]>0],
                    key=lambda it: it["price"])
        A = sp[:idx]; B = sp[idx:]
        ca = norm(np.mean([emb[it["id"]] for it in A], axis=0))
        cb = norm(np.mean([emb[it["id"]] for it in B], axis=0))
        cos = float(ca @ cb)
        nimg = sum(1 for it in v if it["id"] in img)
        flagged.append((ratio, gvf, cos, len(v), k, A, B, nimg))

flagged.sort(reverse=True)
print(f"{len(flagged)} mixture clusters (size>=25, price-ratio>=3) in best model\n")
for ratio, gvf, cos, n, k, A, B, nimg in flagged[:10]:
    medA = st.median([it["price"] for it in A]); medB = st.median([it["price"] for it in B])
    print(f"■ {k[:42]}  n={n} ratio={ratio:.1f} gvf={gvf:.2f} textcos={cos:.3f} img={nimg}/{n}")
    print(f"   CHEAP €{medA:.0f} (n={len(A)}): " +
          " | ".join((it.get('title') or '')[:26] for it in A[:4]))
    print(f"   EXPEN €{medB:.0f} (n={len(B)}): " +
          " | ".join((it.get('title') or '')[:26] for it in B[:4]))
    # would the high-conf Jenks2 bypass fire? (gvf>=0.65 ratio>=4)
    hc = "YES" if (gvf >= 0.65 and ratio >= 4.0) else "no"
    print(f"   jenks high-conf bypass: {hc}  (blocked by emb-cos≥0.95: {cos>=0.95})")
