"""Honest evaluation of clustering candidates. Embeds all items ONCE (cached to
/tmp), then scores any {item_id: key} assignment on THREE axes:

  - coverage:        % items in a cluster of size>=2
  - product cohesion: size-weighted mean cosine(member, its centroid) over
                      multi-item clusters  → are they the SAME product?
  - price coherence:  % items in multi-clusters with price-CV<=0.9

A combined score = product_cohesion * price_coverage rewards clusters that are
both the same product AND price-coherent (so we can't game it by lumping
unrelated items and price-splitting).
"""
from __future__ import annotations
import sys, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
from core.category_db import shared_db_path, connect
from core.clustering import _get_model, _embed_text

CACHE = Path("/tmp/cluster_eval_emb.npz")


def load_items():
    with connect(shared_db_path()) as c:
        return [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category,cluster_key FROM items "
            "WHERE removed=0 AND title IS NOT NULL")]


def embeddings(items):
    ids = np.array([it["id"] for it in items])
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if len(d["ids"]) == len(ids) and (d["ids"] == ids).all():
            return d["emb"]
    model = _get_model()
    emb = model.encode([_embed_text(it) for it in items], show_progress_bar=False,
                       convert_to_numpy=True, batch_size=256)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    np.savez(CACHE, ids=ids, emb=emb)
    return emb


def evaluate(items, emb, keys: dict, name: str):
    n = len(items)
    idx = {it["id"]: i for i, it in enumerate(items)}
    by = collections.defaultdict(list)
    for it in items:
        by[keys[it["id"]]].append(it)
    sizes = [len(v) for v in by.values()]
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    cov = sum(len(v) for v in multi.values())
    # product cohesion (size-weighted) + price coherence
    cohes_num = 0.0; cohes_den = 0
    price_ok = 0; price_items = 0
    for k, v in multi.items():
        vecs = emb[[idx[it["id"]] for it in v]]
        cen = vecs.mean(axis=0); cen /= max(np.linalg.norm(cen), 1e-12)
        sims = vecs @ cen
        cohes_num += float(sims.sum()); cohes_den += len(v)
        pr = [it["price"] for it in v if it["price"]]
        if len(pr) > 1 and st.mean(pr):
            price_items += len(v)
            if st.pstdev(pr) / st.mean(pr) <= 0.9:
                price_ok += len(v)
    cohesion = cohes_num / max(1, cohes_den)
    price_cov = 100 * price_ok / n
    print(f"\n=== {name} ===")
    print(f"  cluster:{len(sizes)} coverage(size>=2):{100*cov/n:.0f}%  "
          f"product_cohesion:{cohesion:.3f}  price_coherent:{price_cov:.0f}%  "
          f"score:{cohesion*price_cov:.1f}")
    return cohesion * price_cov


if __name__ == "__main__":
    from scripts.cluster_experiment import price_split, final_keys, strong_key_v2
    items = load_items(); n = len(items)
    print(f"items:{n}")
    emb = embeddings(items)
    idx = {it["id"]: i for i, it in enumerate(items)}

    cur = {it["id"]: it["cluster_key"] for it in items}
    evaluate(items, emb, cur, "ATTUALE")

    # current + price split
    by = collections.defaultdict(list)
    for it in items: by[it["cluster_key"]].append(it)
    sp = price_split({k: list(v) for k, v in by.items()}, cv_thresh=0.9, gap_log=0.4)
    kp = {it["id"]: k for k, v in sp.items() for it in v}
    evaluate(items, emb, kp, "ATTUALE + price_split")

    # singleton absorption sweep + price split (honest scoring now)
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    sings = [v[0] for k, v in by.items() if len(v) == 1]
    mk = list(multi.keys())
    cm = np.array([emb[[idx[it["id"]] for it in multi[k]]].mean(axis=0) for k in mk])
    cm = cm / np.maximum(np.linalg.norm(cm, axis=1, keepdims=True), 1e-12)
    se = emb[[idx[it["id"]] for it in sings]]
    sims = se @ cm.T; best = sims.argmax(axis=1); bestsim = sims.max(axis=1)
    for absorb in (0.70, 0.78, 0.85):
        merged = {k: list(v) for k, v in multi.items()}
        for i, it in enumerate(sings):
            if bestsim[i] >= absorb: merged[mk[best[i]]].append(it)
            else: merged[f"__s_{it['id']}"] = [it]
        sp2 = price_split(merged, cv_thresh=0.9, gap_log=0.4)
        kp2 = {it["id"]: k for k, v in sp2.items() for it in v}
        evaluate(items, emb, kp2, f"absorb>={absorb} + price_split")
