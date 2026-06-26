"""Two-stage test: TEXT clusters first (real pipeline), THEN images split each
text-cluster into visual sub-products (console vs games inside 'PS4'). Image is
used only WITHIN an already-coherent text cluster, so it can't confuse different
products across the population (the failure of image-as-primary). Measures
price-coherence + coverage before/after. Reuses /tmp/mm_sample.npz.
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
from scripts.cluster_experiment import _tokenize

CACHE = Path("/tmp/mm_sample.npz")


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def metrics(mapping, items, name):
    n = len(items)
    by = collections.defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    cov = sum(len(v) for v in by.values() if len(v) >= 2)
    pok = num = den = 0
    for v in by.values():
        if len(v) < 2:
            continue
        pr = [it["price"] for it in v if it["price"]]
        if len(pr) > 1 and st.mean(pr) and st.pstdev(pr) / st.mean(pr) <= 0.9:
            pok += len(v)
        brs = [_tokenize(it["title"], it["brand"])[1] for it in v]
        brs = [b for b in brs if b]
        if brs:
            num += collections.Counter(brs).most_common(1)[0][1]; den += len(brs)
    print(f"{name:34} cluster={len([1 for v in by.values() if len(v)>=2]):4} "
          f"copertura={100*cov/n:3.0f}% brand-purity={100*num/max(1,den):3.0f}% "
          f"price-coher={100*pok/n:3.0f}%")
    return by


def image_refine(text_map, items, ivec, emap, min_cluster=6, min_sub=3):
    """Within each text cluster, split into visual sub-groups by image HDBSCAN."""
    by = collections.defaultdict(list)
    for it in items:
        by[text_map[it["id"]]].append(it)
    H = cl._get_hdbscan().HDBSCAN
    out = dict(text_map)
    n_split = 0
    splits = []
    for k, v in by.items():
        if len(v) < min_cluster:
            continue
        X = norm(np.array([ivec[emap[it["id"]]] for it in v]))
        lab = H(min_cluster_size=min_sub, metric="euclidean",
                cluster_selection_method="leaf", core_dist_n_jobs=-1).fit_predict(X)
        sub = collections.Counter(l for l in lab if l != -1)
        if len(sub) < 2:                      # no ≥2 substantial visual groups
            continue
        n_split += 1
        for it, l in zip(v, lab):
            if l != -1:
                out[it["id"]] = f"{k} ·img{int(l)}"
            # image-noise items keep the parent text key (not lost)
        splits.append((k, v, lab))
    print(f"  cluster-testo spezzati per immagine: {n_split}")
    return out, splits


def main():
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv, iv = norm(d["tv"]), norm(d["iv"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    tcache = {it["id"]: tv[emap[it["id"]]] for it in items}

    with connect(Path("/tmp/imref.db")) as conn:
        base = base_partition(items, tcache, "catbrand")
        text_map = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=tcache)
    print(f"campione: {len(items)}\n=== STADIO 1: solo TESTO ===")
    metrics(text_map, items, "solo TESTO")

    print("\n=== STADIO 2: + raffinamento IMMAGINE dentro i cluster-testo ===")
    ref_map, splits = image_refine(text_map, items, iv, emap)
    metrics(ref_map, items, "TESTO + img-refine")

    print("\n=== esempi di cluster-testo spezzati dall'immagine ===")
    shown = 0
    for k, v, lab in splits:
        groups = collections.defaultdict(list)
        for it, l in zip(v, lab):
            groups[int(l)].append(it)
        if len([g for g in groups if g != -1]) < 2:
            continue
        print(f"\n[{k[-30:]}] {len(v)} item →")
        for l, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            pr = [it["price"] for it in g]
            nm = "rumore" if l == -1 else f"img{l}"
            print(f"   {nm:7} n={len(g)} med€{st.median(pr):.0f}: " +
                  " | ".join(it["title"][:20] for it in g[:3]))
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()
