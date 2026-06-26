"""Targeted: do images help the MESSY (high price-CV) text clusters specifically?
For each text cluster with high CV, run image sub-clustering and measure how much
the within-group price-CV drops (image finding price-coherent sub-products).
Reuses /tmp/mm_sample.npz.
"""
import sys, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import connect
from scripts.cluster_ab2 import base_partition

CACHE = Path("/tmp/mm_sample.npz")


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def cv(prices):
    p = [x for x in prices if x]
    return (st.pstdev(p) / st.mean(p)) if len(p) > 1 and st.mean(p) else 0


def main():
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv, iv = norm(d["tv"]), norm(d["iv"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    tcache = {it["id"]: tv[emap[it["id"]]] for it in items}

    with connect(Path("/tmp/imreft.db")) as conn:
        base = base_partition(items, tcache, "catbrand")
        text_map = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=tcache)
    by = collections.defaultdict(list)
    for it in items:
        by[text_map[it["id"]]].append(it)

    # the MESSY text clusters: >=6 items and high price-CV
    messy = [(k, v) for k, v in by.items() if len(v) >= 6 and cv([it["price"] for it in v]) > 0.7]
    print(f"cluster-testo MESSY (n>=6, CV>0.7): {len(messy)}\n")

    H = cl._get_hdbscan().HDBSCAN
    improved = same = 0
    tot_before = tot_after = tot_items = 0
    for k, v in messy:
        before_cv = cv([it["price"] for it in v])
        X = norm(np.array([iv[emap[it["id"]]] for it in v]))
        lab = H(min_cluster_size=3, metric="euclidean",
                cluster_selection_method="leaf", core_dist_n_jobs=-1).fit_predict(X)
        sub = collections.defaultdict(list)
        for it, l in zip(v, lab):
            sub[int(l)].append(it)
        real = {l: g for l, g in sub.items() if l != -1 and len(g) >= 2}
        if len(real) < 1:
            after_cv = before_cv
        else:
            # weighted mean CV of the image sub-groups (noise excluded)
            num = sum(cv([it["price"] for it in g]) * len(g) for g in real.values())
            den = sum(len(g) for g in real.values())
            after_cv = num / den if den else before_cv
        tot_before += before_cv * len(v); tot_after += after_cv * len(v); tot_items += len(v)
        if after_cv < before_cv - 0.1:
            improved += 1
        else:
            same += 1
        if improved <= 8 and after_cv < before_cv - 0.1:
            meds = " | ".join(f"€{st.median([it['price'] for it in g]):.0f}(n{len(g)})"
                              for g in sorted(real.values(), key=lambda g: -len(g)))
            print(f"  [{k[-26:]}] CV {before_cv:.2f}→{after_cv:.2f}  fasce: {meds}")
    print(f"\nMESSY migliorati dall'immagine (CV-0.1+): {improved}/{len(messy)}")
    print(f"CV medio pesato: {tot_before/max(1,tot_items):.2f} → {tot_after/max(1,tot_items):.2f}")


if __name__ == "__main__":
    main()
