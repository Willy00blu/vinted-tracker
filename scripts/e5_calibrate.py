"""Calibrate the rescue gates for the e5 embedding model.

e5 cosine sims sit on a higher scale than MiniLM, so the MiniLM-tuned gates
(0.86/0.72) over-merge. We embed once (cached), compute the HDBSCAN base
partition once, then sweep (gate_hi, gate_lo) measuring MODEL-INDEPENDENT
quality (brand-purity, coverage, price-coherence, singleton%).

Run with:  VINTED_EMBED_MODEL=intfloat/multilingual-e5-large \
           python3 scripts/e5_calibrate.py
"""
import sys, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_experiment import _tokenize

CACHE = Path("/tmp/e5_emb.npz")


def load_items():
    with connect(shared_db_path()) as c:
        return [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items "
            "WHERE removed=0 AND title IS NOT NULL")]


def embed(items):
    ids = np.array([it["id"] for it in items])
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if len(d["ids"]) == len(ids) and (d["ids"] == ids).all():
            print(f"  (embedding e5 da cache: {len(ids)})", flush=True)
            return {i: e for i, e in zip(d["ids"], d["emb"])}
    print(f"  embedding {len(ids)} con {cl.MODEL_NAME} … (lento)", flush=True)
    m = cl._get_model()
    e = m.encode([cl._embed_text(it) for it in items], show_progress_bar=False,
                 convert_to_numpy=True, batch_size=128)
    np.savez(CACHE, ids=ids, emb=e)
    return {i: v for i, v in zip(ids, e)}


def quick(mapping, items):
    n = len(items)
    by = collections.defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    sizes = [len(v) for v in by.values()]
    sing = sum(1 for s in sizes if s == 1)
    cov = sum(s for s in sizes if s >= 2)
    pok = 0; num = den = 0
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
    return dict(clusters=len(sizes), sing_item=100*sing/n,
               cov=100*cov/n, price=100*pok/n, purity=100*num/max(1, den))


def main():
    items = load_items()
    print(f"items: {len(items)}  model: {cl.MODEL_NAME}")
    emb = embed(items)

    # base HDBSCAN partition (gate-independent) — one throwaway DB for centroids
    with connect(Path("/tmp/e5_sweep_base.db")) as base_conn:
        base = cl.cluster_partitioned(items, conn=base_conn, log_fn=lambda *a: None,
                                      emb_cache=emb)
    bsizes = collections.Counter(base.values())
    nsing = sum(1 for k, c in bsizes.items() if c == 1)
    print(f"base HDBSCAN: {len(bsizes)} cluster, {nsing} singleton")

    # diagnostic: distribution of singleton→nearest-multi-centroid sims (e5 scale)
    multi = collections.defaultdict(list)
    for it in items:
        multi[base[it["id"]]].append(it)
    multi = {k: v for k, v in multi.items() if len(v) >= 2}
    sing_keys = {k for k, c in bsizes.items() if c == 1}
    sings = [it for it in items if base[it["id"]] in sing_keys]
    mk = list(multi.keys())
    cmat = np.array([np.mean([emb[it["id"]] for it in multi[k]], axis=0) for k in mk])
    cmat = cmat / np.maximum(np.linalg.norm(cmat, axis=1, keepdims=True), 1e-12)
    se = np.array([emb[it["id"]] for it in sings])
    se = se / np.maximum(np.linalg.norm(se, axis=1, keepdims=True), 1e-12)
    best = (se @ cmat.T).max(axis=1)
    print(f"singleton→best-centroid sim (e5): "
          f"p50={np.percentile(best,50):.3f} p75={np.percentile(best,75):.3f} "
          f"p90={np.percentile(best,90):.3f} max={best.max():.3f}")

    print(f"\n{'HI':>5} {'LO':>5} | {'clusters':>8} {'cov%':>5} {'price%':>6} {'purity%':>7} {'sing%':>5}")
    print(f"  (MiniLM rif:        23828   68     64      94      32)")
    with connect(Path("/tmp/e5_sweep.db")) as sweep_conn:
        for hi in (0.88, 0.90, 0.92, 0.94, 0.96):
            for lo in (0.84, 0.86, 0.88, 0.90):
                if lo >= hi:
                    continue
                m = cl.refine_clusters(items, dict(base), sweep_conn,
                                       log_fn=lambda *a: None, emb_cache=emb,
                                       gate_hi=hi, gate_lo=lo)
                r = quick(m, items)
                print(f"{hi:>5} {lo:>5} | {r['clusters']:>8} {r['cov']:>5.0f} "
                      f"{r['price']:>6.0f} {r['purity']:>7.0f} {r['sing_item']:>5.0f}")


if __name__ == "__main__":
    main()
