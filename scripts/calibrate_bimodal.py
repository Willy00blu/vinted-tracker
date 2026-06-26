"""Test the BIMODAL split hypothesis: split a cluster only when its prices form
TWO substantial groups (each >= min_frac AND >= min_count), separated by a gap
that is large relative to the within-group robust spread (MAD). A lone cheap
outlier (a deal) is NOT two modes → not split. Ground truth = product-type
difference between the two sides.
"""
import sys, collections, statistics as st, math, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_ab2 import base_partition, CACHE
from scripts.cluster_typesplit import _detect_type

STRIP = re.compile(r" \[[^\]]*\]$")


def mad(xs):
    m = st.median(xs)
    return st.median([abs(x - m) for x in xs]) or 1e-9


def main():
    d = np.load(CACHE, allow_pickle=True)
    emb = {int(i): vv for i, vv in zip(d["ids"], d["emb"])}
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items WHERE removed=0 AND title IS NOT NULL")]
    items = [it for it in items if it["id"] in emb]
    with connect(Path("/tmp/bim.db")) as conn:
        base = base_partition(items, emb, "catbrand")
        m1 = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)
    pre = collections.defaultdict(list)
    for it in items:
        pre[STRIP.sub("", m1[it["id"]])].append(it)

    # candidate bimodal boundaries
    cands = []   # (sep_mad, minfrac, label, k, A, B)
    for k, v in pre.items():
        pr = [it["price"] for it in v if it.get("price")]
        if len(v) < 10 or len(pr) < 10 or not st.mean(pr):
            continue
        if st.pstdev(pr) / st.mean(pr) <= 0.6:     # some spread needed
            continue
        priced = sorted((it for it in v if it.get("price")), key=lambda it: it["price"])
        lp = [math.log(it["price"]) for it in priced]
        N = len(priced)
        mincount = max(5, int(0.2 * N))            # each mode substantial
        best = None
        for i in range(mincount - 1, N - mincount):
            g = lp[i + 1] - lp[i]
            if best is None or g > best[0]:
                best = (g, i)
        if best is None:
            continue
        g, i = best
        A, B = priced[:i + 1], priced[i + 1:]
        within = mad([x for x in lp])              # robust spread of whole cluster
        sep = g / within                           # MAD-units of the gap
        tA = collections.Counter(t for t in (_detect_type(x["title"]) for x in A) if t)
        tB = collections.Counter(t for t in (_detect_type(x["title"]) for x in B) if t)
        label = None
        if tA and tB:
            label = 1 if tA.most_common(1)[0][0] != tB.most_common(1)[0][0] else 0
        cands.append((sep, g, label, k, A, B))

    labeled = [c for c in cands if c[2] is not None]
    pos = [c for c in labeled if c[2] == 1]
    neg = [c for c in labeled if c[2] == 0]
    print(f"cluster bimodali (2 mode sostanziali): {len(cands)}  etichettati: {len(labeled)} "
          f"(diversi={len(pos)}, stesso={len(neg)})\n")
    print(f"{'sep_MAD>':>8} {'gap>':>5} | {'prec%':>6} {'rec%':>5} {'split':>6} {'falsi':>6}")
    for sepmin in (2.0, 3.0, 4.0, 5.0):
        for gapmin in (0.5, 0.7):
            tp = sum(1 for c in labeled if c[0] > sepmin and c[1] > gapmin and c[2] == 1)
            fp = sum(1 for c in labeled if c[0] > sepmin and c[1] > gapmin and c[2] == 0)
            fn = sum(1 for c in labeled if not (c[0] > sepmin and c[1] > gapmin) and c[2] == 1)
            prec = tp/(tp+fp) if tp+fp else 0
            rec = tp/(tp+fn) if tp+fn else 0
            print(f"{sepmin:>8} {gapmin:>5} | {100*prec:>6.0f} {100*rec:>5.0f} {tp+fp:>6} {fp:>6}")

    # eyeball bimodal clusters at sep>3, gap>0.5
    print("\n=== esempi bimodali (sep_MAD>3, gap>0.5) ===")
    shown = 0
    for sep, g, lab, k, A, B in sorted(cands, key=lambda c: -c[0]):
        if sep <= 3 or g <= 0.5:
            continue
        ma = st.median([x["price"] for x in A]); mb = st.median([x["price"] for x in B])
        print(f"  [{k[-30:]}] sep={sep:.1f} | basso n={len(A)} med€{ma:.0f} ({A[0]['title'][:18]}) "
              f"| alto n={len(B)} med€{mb:.0f} ({B[-1]['title'][:18]})")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
