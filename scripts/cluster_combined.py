"""Test the user's combined rule: split a cluster only when there's a price gap
AND the two sides also differ in embedding (different product), with both sides
substantial (>=min_band). This avoids shaving single price outliers (= deals)
of the SAME product, which a price-only gap split wrongly isolates.
"""
import sys, collections, statistics as st, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import re
from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_ab2 import base_partition, CACHE

STRIP = re.compile(r" \[[^\]]*\]$")


def _norm(x):
    return x / np.maximum(np.linalg.norm(x), 1e-12)


def combined_split(by, emb, cv_thresh, gap_log, min_band, max_cos):
    out = {}; n_split = 0; rejected_sim = 0
    for k, v in by.items():
        pr = [it["price"] for it in v if it.get("price")]
        if len(v) < 8 or len(pr) < 6 or not st.mean(pr) or st.pstdev(pr)/st.mean(pr) <= cv_thresh:
            out[k] = v; continue
        priced = sorted((it for it in v if it.get("price")), key=lambda it: it["price"])
        noprice = [it for it in v if not it.get("price")]
        lp = [math.log(it["price"]) for it in priced]
        # largest gap with both sides >= min_band
        best = None
        for i in range(min_band-1, len(priced)-min_band):
            g = lp[i+1]-lp[i]
            if g > gap_log and (best is None or g > best[0]):
                best = (g, i)
        if not best:
            out[k] = v; continue
        i = best[1]
        A, B = priced[:i+1], priced[i+1:]
        ca = _norm(np.mean([emb[it["id"]] for it in A], axis=0))
        cb = _norm(np.mean([emb[it["id"]] for it in B], axis=0))
        cos = float(ca @ cb)
        if cos > max_cos:                 # titles ~same product → DON'T split
            rejected_sim += 1
            out[k] = v; continue
        n_split += 1
        out[f"{k} [{int(A[0]['price'])}-{int(A[-1]['price'])}€]"] = A
        out[f"{k} [{int(B[0]['price'])}-{int(B[-1]['price'])}€]"] = B
        if noprice:
            out.setdefault(f"{k} [?]", []).extend(noprice)
    return out, n_split, rejected_sim


def price_coh(by, n):
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
    with connect(Path("/tmp/comb.db")) as conn:
        base = base_partition(items, emb, "catbrand")
        m1 = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)
    pre = collections.defaultdict(list)
    for it in items:
        pre[STRIP.sub("", m1[it["id"]])].append(it)
    print(f"items: {n}  pre-split price-coherence: {price_coh(pre, n):.0f}%\n")

    print(f"{'max_cos (più basso=più severo)':32} {'price%':>6} {'split':>6} {'rifiut(simili)':>14}")
    for mc in (0.995, 0.99, 0.985, 0.98):
        out, ns, rj = combined_split(dict(pre), emb, 0.9, 0.7, 3, mc)
        print(f"{'gap0.7 min3 max_cos='+str(mc):32} {price_coh(out, n):>6.0f} {ns:>6} {rj:>14}")

    # eyeball at a middle threshold
    out, ns, rj = combined_split(dict(pre), emb, 0.9, 0.7, 3, 0.99)
    print(f"\n=== ACCETTATI (split veri, max_cos 0.99) — primi 8 ===")
    parents = collections.defaultdict(list)
    for k in out:
        parents[STRIP.sub("", k)].append(k)
    shown = 0
    for p, bands in parents.items():
        if len(bands) < 2: continue
        rows = []
        for b in sorted(bands):
            pr = [it["price"] for it in out[b] if it["price"]]
            ex = out[b][0]["title"][:22]
            rows.append(f"{b[-12:]:14} n={len(out[b])} med€{st.median(pr) if pr else 0:.0f} es:{ex}")
        print(f"  [{p[-28:]}]"); [print("     ", r) for r in rows]
        shown += 1
        if shown >= 8: break


if __name__ == "__main__":
    main()
