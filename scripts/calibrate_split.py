"""Data-driven calibration of the two price-split thresholds:
  - PRICE gap (log ratio) between the two tiers
  - TITLE difference (1 - embedding cosine) between the two tiers

Ground truth (weak): the detected product-TYPE of each side (console/game/...).
Different type on the two sides = a TRUE split (different product); same type =
should NOT split. We find the thresholds that best predict this, leaning toward
high precision (few false splits = few masked deals).
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
MIN_BAND = 3


def _norm(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def main():
    d = np.load(CACHE, allow_pickle=True)
    emb = {int(i): vv for i, vv in zip(d["ids"], d["emb"])}
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items WHERE removed=0 AND title IS NOT NULL")]
    items = [it for it in items if it["id"] in emb]
    with connect(Path("/tmp/calib.db")) as conn:
        base = base_partition(items, emb, "catbrand")
        m1 = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)
    pre = collections.defaultdict(list)
    for it in items:
        pre[STRIP.sub("", m1[it["id"]])].append(it)

    # one candidate boundary per cluster: largest valid price gap
    cands = []  # (gap_log, cos, label)  label: 1=diff type(split), 0=same type
    for k, v in pre.items():
        pr = [it["price"] for it in v if it.get("price")]
        if len(v) < 8 or len(pr) < 2 * MIN_BAND or not st.mean(pr):
            continue
        if st.pstdev(pr) / st.mean(pr) <= 0.9:
            continue
        priced = sorted((it for it in v if it.get("price")), key=lambda it: it["price"])
        lp = [math.log(it["price"]) for it in priced]
        best = None
        for i in range(MIN_BAND - 1, len(priced) - MIN_BAND):
            g = lp[i + 1] - lp[i]
            if best is None or g > best[0]:
                best = (g, i)
        if best is None or best[0] <= 0:
            continue
        g, i = best
        A, B = priced[:i + 1], priced[i + 1:]
        cos = float(_norm(np.mean([emb[it["id"]] for it in A], axis=0))
                    @ _norm(np.mean([emb[it["id"]] for it in B], axis=0)))
        tA = collections.Counter(t for t in (_detect_type(x["title"]) for x in A) if t)
        tB = collections.Counter(t for t in (_detect_type(x["title"]) for x in B) if t)
        if not tA or not tB:
            continue
        label = 1 if tA.most_common(1)[0][0] != tB.most_common(1)[0][0] else 0
        cands.append((g, cos, label))

    pos = [c for c in cands if c[2] == 1]
    neg = [c for c in cands if c[2] == 0]
    print(f"candidati etichettati: {len(cands)}  (prodotti-diversi={len(pos)}, stesso-tipo={len(neg)})\n")
    def pct(xs, q): return np.percentile(xs, q) if xs else float("nan")
    print("COSINO tra i due lati (più basso = titoli più diversi):")
    print(f"  prodotti-diversi: p25={pct([c[1] for c in pos],25):.3f} p50={pct([c[1] for c in pos],50):.3f} p75={pct([c[1] for c in pos],75):.3f}")
    print(f"  stesso-tipo:      p25={pct([c[1] for c in neg],25):.3f} p50={pct([c[1] for c in neg],50):.3f} p75={pct([c[1] for c in neg],75):.3f}")
    print("GAP prezzo (log; e^gap = rapporto):")
    print(f"  prodotti-diversi: p25={pct([c[0] for c in pos],25):.2f} p50={pct([c[0] for c in pos],50):.2f}")
    print(f"  stesso-tipo:      p25={pct([c[0] for c in neg],25):.2f} p50={pct([c[0] for c in neg],50):.2f}\n")

    print(f"{'gap*':>5} {'cos*':>6} | {'prec%':>6} {'recall%':>7} {'F1':>5} {'split':>6} {'falsi':>6}")
    best_f1 = (0, None)
    for gap in (0.4, 0.55, 0.7, 0.9):
        for cos in (0.93, 0.95, 0.97, 0.98, 0.99):
            tp = sum(1 for g, c, l in cands if g > gap and c < cos and l == 1)
            fp = sum(1 for g, c, l in cands if g > gap and c < cos and l == 0)
            fn = sum(1 for g, c, l in cands if not (g > gap and c < cos) and l == 1)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            if f1 > best_f1[0]:
                best_f1 = (f1, (gap, cos, prec, rec, tp, fp))
            print(f"{gap:>5} {cos:>6} | {100*prec:>6.0f} {100*rec:>7.0f} {f1:>5.2f} {tp+fp:>6} {fp:>6}")
    g, c, p, r, tp, fp = best_f1[1]
    print(f"\n→ MIGLIORE F1: gap>{g} (~{math.exp(g):.1f}x) & cos<{c} → precisione {100*p:.0f}%, recall {100*r:.0f}%, {fp} falsi")


if __name__ == "__main__":
    main()
