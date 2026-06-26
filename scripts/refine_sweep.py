"""Refine the text+image recipe to the max. On cached tv + L-14 multi-photo
(499 items), compare: text-only, combined-concat (various weights), and
two-stage (image-only vs text+image fused in stage 2, various min_cluster_size).
Reports price-coherence, coverage, MESSY fixed (good) and CLEAN split (over-split,
bad). Reuses /tmp/multiphoto.npz — no scrape.
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

CACHE = Path("/tmp/multiphoto.npz")


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def cvp(items):
    p = [it["price"] for it in items if it["price"]]
    return (st.pstdev(p)/st.mean(p)) if len(p) > 1 and st.mean(p) else 0


def metrics(mapping, items):
    n = len(items); by = collections.defaultdict(list)
    for it in items: by[mapping[it["id"]]].append(it)
    cov = sum(len(v) for v in by.values() if len(v) >= 2)
    pok = 0
    for v in by.values():
        if len(v) >= 2 and cvp(v) <= 0.9:
            pok += len(v)
    return 100*cov/n, 100*pok/n


def two_stage(text_map, items, img, emap, mcs, fuse_tv=None, cv_gate=0.0):
    by = collections.defaultdict(list)
    for it in items: by[text_map[it["id"]]].append(it)
    H = cl._get_hdbscan().HDBSCAN
    out = dict(text_map)
    messy_fixed = clean_split = 0
    for k, v in by.items():
        if len(v) < 6:
            continue
        if cv_gate and cvp(v) <= cv_gate:    # only refine ALREADY-MESSY clusters
            continue
        if fuse_tv is not None:
            X = norm(np.array([np.concatenate([0.4*fuse_tv[emap[it["id"]]],
                                               0.6*img[emap[it["id"]]]]) for it in v]))
        else:
            X = norm(np.array([img[emap[it["id"]]] for it in v]))
        lab = H(min_cluster_size=mcs, metric="euclidean", cluster_selection_method="leaf",
                core_dist_n_jobs=-1).fit_predict(X)
        if len(set(l for l in lab if l != -1)) < 2:
            continue
        before = cvp(v)
        sub = collections.defaultdict(list)
        for it, l in zip(v, lab):
            if l != -1:
                out[it["id"]] = f"{k}#i{int(l)}"; sub[int(l)].append(it)
        real = [g for g in sub.values() if len(g) >= 2]
        if real:
            after = sum(cvp(g)*len(g) for g in real)/sum(len(g) for g in real)
            if before > 0.7 and after < before - 0.1:
                messy_fixed += 1
            elif before <= 0.5:           # split a clean cluster = over-split
                clean_split += 1
    return out, messy_fixed, clean_split


def main():
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv = norm(d["tv"]); img = norm(d["l14_multi"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    tcache = {it["id"]: tv[emap[it["id"]]] for it in items}
    n = len(items)
    print(f"campione: {n}\n")

    with connect(Path("/tmp/rs.db")) as conn:
        base = base_partition(items, tcache, "catbrand")
        text_map = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=tcache)
    cov, pc = metrics(text_map, items)
    print(f"{'TESTO-only':40} cov={cov:3.0f}% price-coh={pc:3.0f}%")

    # combined-concat
    def run_cache(vecs, name):
        cache = {it["id"]: vecs[emap[it["id"]]] for it in items}
        with connect(Path("/tmp/rs.db")) as conn:
            base = base_partition(items, cache, "catbrand")
            m = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=cache)
        cov, pc = metrics(m, items)
        print(f"{name:40} cov={cov:3.0f}% price-coh={pc:3.0f}%")
    for a in (0.3, 0.5, 0.7):
        run_cache(np.concatenate([a*tv, (1-a)*img], axis=1), f"CONCAT testo={a}")

    print()
    for mcs in (2, 3):
        m, mf, cs = two_stage(text_map, items, img, emap, mcs)
        cov, pc = metrics(m, items)
        print(f"{'2STAGE img-only mcs='+str(mcs):40} cov={cov:3.0f}% price-coh={pc:3.0f}% "
              f"messy_fix={mf} clean_split(BAD)={cs}")
    for mcs in (2, 3):
        m, mf, cs = two_stage(text_map, items, img, emap, mcs, fuse_tv=tv)
        cov, pc = metrics(m, items)
        print(f"{'2STAGE fuso(t+i) mcs='+str(mcs):40} cov={cov:3.0f}% price-coh={pc:3.0f}% "
              f"messy_fix={mf} clean_split(BAD)={cs}")
    print("  --- gated: raffina SOLO i cluster già sporchi (CV>0.7) ---")
    for fuse, lbl in ((None, "img-only"), (tv, "fuso(t+i)")):
        m, mf, cs = two_stage(text_map, items, img, emap, 2, fuse_tv=fuse, cv_gate=0.7)
        cov, pc = metrics(m, items)
        print(f"{'2STAGE '+lbl+' mcs=2 gated':40} cov={cov:3.0f}% price-coh={pc:3.0f}% "
              f"messy_fix={mf} clean_split(BAD)={cs}")


if __name__ == "__main__":
    main()
