"""Definitive two-stage multimodal test (coarse-to-fine, like the papers):
  Stage 1: TEXT clusters (e5 + cat/brand partition + refine)  [blocking/coarse]
  Stage 2: inside each text cluster, refine/split by IMAGE (L-14 multi-photo)
           into distinct visual products.
Reuses /tmp/multiphoto.npz (no scrape). Measures text-only vs two-stage.
"""
import sys, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import connect
from scripts.cluster_ab2 import base_partition
from scripts.fashion_clip_test import report

CACHE = Path("/tmp/multiphoto.npz")


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def cv(items):
    p = [it["price"] for it in items if it["price"]]
    return (st.pstdev(p)/st.mean(p)) if len(p) > 1 and st.mean(p) else 0


def main():
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv = norm(d["tv"]); img = norm(d["l14_multi"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    tcache = {it["id"]: tv[emap[it["id"]]] for it in items}
    print(f"campione: {len(items)}  (L-14 multi-foto)\n")

    # STAGE 1: text clusters
    with connect(Path("/tmp/ts.db")) as conn:
        base = base_partition(items, tcache, "catbrand")
        text_map = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=tcache)
    report(text_map, items, "STADIO 1: solo TESTO")

    # STAGE 2: image refine within each text cluster
    by = collections.defaultdict(list)
    for it in items:
        by[text_map[it["id"]]].append(it)
    H = cl._get_hdbscan().HDBSCAN
    ref = dict(text_map)
    n_split = 0; splits = []
    for k, v in by.items():
        if len(v) < 6:
            continue
        X = norm(np.array([img[emap[it["id"]]] for it in v]))
        lab = H(min_cluster_size=3, metric="euclidean", cluster_selection_method="leaf",
                core_dist_n_jobs=-1).fit_predict(X)
        sub = collections.Counter(l for l in lab if l != -1)
        if len(sub) < 2:
            continue
        n_split += 1; splits.append((k, v, lab))
        for it, l in zip(v, lab):
            if l != -1:
                ref[it["id"]] = f"{k} ·img{int(l)}"
    print(f"  (cluster-testo spezzati dall'immagine: {n_split})")
    report(ref, items, "STADIO 2: TESTO + img-refine (L-14 multi)")

    # targeted: CV reduction on messy text clusters
    messy = [(k, v) for k, v in by.items() if len(v) >= 6 and cv(v) > 0.7]
    improved = 0
    print(f"\n=== MESSY (n>=6,CV>0.7): {len(messy)} ===")
    for k, v, lab in splits:
        if cv(v) <= 0.7:
            continue
        sub = collections.defaultdict(list)
        for it, l in zip(v, lab):
            sub[int(l)].append(it)
        real = {l: g for l, g in sub.items() if l != -1 and len(g) >= 2}
        if not real:
            continue
        after = sum(cv(g)*len(g) for g in real.values())/sum(len(g) for g in real.values())
        if after < cv(v) - 0.1:
            improved += 1
            meds = " | ".join(f"€{st.median([it['price'] for it in g]):.0f}(n{len(g)})"
                              for g in sorted(real.values(), key=lambda g: -len(g)))
            print(f"  [{k[-26:]}] CV {cv(v):.2f}→{after:.2f}  {meds}")
    print(f"\nMESSY sistemati dall'immagine L-14-multi: {improved}/{len(messy)}")


if __name__ == "__main__":
    main()
