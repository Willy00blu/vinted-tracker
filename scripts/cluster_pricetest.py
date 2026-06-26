"""Test conservative price-split variants on the cat+brand e5 clusters.

Goal: lift price-coherence by splitting ONLY clusters with a real price chasm
(a big multiplicative gap = console-vs-games boundary), never continuous
variation (same product, different condition → would mask deals). Compares
against the current (quantile-fallback) split. Reuses cached e5 embeddings.
"""
import sys, collections, statistics as st, re, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_experiment import _tokenize
from scripts.cluster_ab2 import base_partition, CACHE

STRIP = re.compile(r" \[[^\]]*\]$")   # remove " [10-590€]" / " [?]" suffixes


def split_gap_only(by, cv_thresh, gap_log, max_bands=4):
    """Split a cluster ONLY at genuine multiplicative gaps (no quantile)."""
    out = {}; n_split = 0; real_chasm = 0
    for k, v in by.items():
        pr = [it["price"] for it in v if it.get("price")]
        if len(v) < 8 or len(pr) < 4 or not st.mean(pr) or st.pstdev(pr)/st.mean(pr) <= cv_thresh:
            out[k] = v; continue
        priced = sorted((it for it in v if it.get("price")), key=lambda it: it["price"])
        noprice = [it for it in v if not it.get("price")]
        lp = [math.log(it["price"]) for it in priced]
        gaps = sorted(((lp[i+1]-lp[i], i) for i in range(len(lp)-1)), reverse=True)
        cuts = sorted(i for g, i in gaps[:max_bands-1] if g > gap_log)
        if not cuts:
            out[k] = v; continue
        n_split += 1
        # record the biggest gap ratio for diagnostics
        if math.exp(gaps[0][0]) >= 2.0:
            real_chasm += 1
        prev = 0
        for ci in cuts + [len(priced)-1]:
            if ci >= prev:
                band = priced[prev:ci+1]
                if band:
                    out[f"{k} [{int(band[0]['price'])}-{int(band[-1]['price'])}€]"] = band
                prev = ci+1
        if noprice:
            out.setdefault(f"{k} [?]", []).extend(noprice)
    return out, n_split, real_chasm


def price_coherent(by, n):
    pok = 0
    for v in by.values():
        if len(v) < 2: continue
        pr = [it["price"] for it in v if it["price"]]
        if len(pr) > 1 and st.mean(pr) and st.pstdev(pr)/st.mean(pr) <= 0.9:
            pok += len(v)
    return 100*pok/n


def main():
    d = np.load(CACHE, allow_pickle=True)
    emb = {int(i): vv for i, vv in zip(d["ids"], d["emb"])}
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items WHERE removed=0 AND title IS NOT NULL")]
    items = [it for it in items if it["id"] in emb]
    n = len(items)
    print(f"items: {n}")

    with connect(Path("/tmp/pricetest.db")) as conn:
        base = base_partition(items, emb, "catbrand")
        m1 = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)

    # reconstruct PRE-price-split groups by stripping the band suffixes
    pre = collections.defaultdict(list)
    for it in items:
        pre[STRIP.sub("", m1[it["id"]])].append(it)
    print(f"cluster pre-split: {len(pre)}  price-coherence(no split): {price_coherent(pre, n):.0f}%\n")

    print(f"{'variante':40} {'price%':>6} {'split':>6} {'chasm≥2x':>8}")
    for name, cv, gap in [
        ("solo-gap cv0.8 gap0.5 (~1.6x)", 0.8, 0.5),
        ("solo-gap cv0.8 gap0.7 (~2.0x)", 0.8, 0.7),
        ("solo-gap cv0.9 gap0.9 (~2.5x) CONSERV", 0.9, 0.9),
        ("solo-gap cv0.9 gap0.7 (~2.0x)", 0.9, 0.7),
    ]:
        out, ns, rc = split_gap_only(dict(pre), cv, gap)
        print(f"{name:40} {price_coherent(out, n):>6.0f} {ns:>6} {rc:>8}")

    # current (with quantile fallback) for reference
    cur = collections.defaultdict(list)
    for it in items:
        cur[m1[it["id"]]].append(it)
    print(f"{'ATTUALE (quantile fallback, gap0.4)':40} {price_coherent(cur, n):>6.0f}")

    # eyeball: what gets split by the conservative variant
    out, ns, rc = split_gap_only(dict(pre), 0.9, 0.9)
    print("\n=== esempi spezzati dal CONSERVATIVO (cv0.9 gap0.9) ===")
    parents = collections.defaultdict(list)
    for k in out:
        parents[STRIP.sub("", k)].append(k)
    shown = 0
    for p, bands in parents.items():
        if len(bands) < 2: continue
        print(f"  [{p[-34:]}] →")
        for b in sorted(bands):
            pr = [it["price"] for it in out[b] if it["price"]]
            print(f"      {b[-14:]:16} n={len(out[b])} med€{st.median(pr) if pr else 0:.0f}")
        shown += 1
        if shown >= 6: break


if __name__ == "__main__":
    main()
