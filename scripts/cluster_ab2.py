"""A/B test two clustering improvements on the real DB, reusing cached e5
embeddings (no re-embedding):

  #1  partition by category+brand jointly (vs current category-OR-brand)
  #3  embedding sub-split of big clusters (console vs games inside 'PS4')

Metrics are MODEL-INDEPENDENT (brand-purity, coverage, price-coherence) so they
can't be gamed by the embedding choice. Run:
    python3 scripts/cluster_ab2.py
"""
import sys, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")

from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_experiment import _tokenize

CACHE = Path("/tmp/e5_emb.npz")


def load():
    d = np.load(CACHE, allow_pickle=True)
    emb = {int(i): v for i, v in zip(d["ids"], d["emb"])}
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items "
            "WHERE removed=0 AND title IS NOT NULL")]
    items = [it for it in items if it["id"] in emb]   # only cached ones
    return items, emb


def part_key(it, mode):
    cat = (it.get("category") or "").strip()
    brand, _ = cl._lex_sig(it.get("title", ""), it.get("brand"))
    if mode == "current":
        return cat if cat else (brand or "__nobrand__")
    # cat+brand: use BOTH when category known
    if cat and brand:
        return f"{cat}|{brand}"
    if cat:
        return cat
    return brand or "__nobrand__"


def base_partition(items, emb, mode):
    by = collections.defaultdict(list)
    for it in items:
        by[part_key(it, mode)].append(it)
    mapping, used = {}, set()
    for part, grp in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ns = "" if part == "__nobrand__" else f"{part}/"
        m, _rows = cl._hdbscan_compute(grp, ns, 2, lambda *a: None, emb)
        # global key de-dup like cluster_partitioned
        ren = {}
        for k in set(m.values()):
            nk, c = k, 0
            while nk in used:
                c += 1; nk = f"{k}~{c}"
            used.add(nk)
            if nk != k:
                ren[k] = nk
        if ren:
            m = {i: ren.get(k, k) for i, k in m.items()}
        mapping.update(m)
    return mapping


def subsplit(mapping, items, emb, min_size=40, sim_keep=0.0):
    """Split clusters larger than min_size with an internal HDBSCAN(leaf)."""
    by = collections.defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    H = cl._get_hdbscan().HDBSCAN
    out = dict(mapping)
    n_split = 0
    for k, v in by.items():
        if len(v) <= min_size:
            continue
        X = np.array([emb[it["id"]] for it in v])
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        lab = H(min_cluster_size=2, metric="euclidean",
                cluster_selection_method="leaf", core_dist_n_jobs=-1).fit_predict(X)
        if len(set(lab) - {-1}) <= 1:
            continue                       # no useful split
        n_split += 1
        for it, l in zip(v, lab):
            if l != -1:
                out[it["id"]] = f"{k}#s{int(l)}"
            # noise (-1) keeps parent key k
    return out, n_split


def metrics(mapping, items, name):
    n = len(items)
    by = collections.defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    sizes = [len(v) for v in by.values()]
    sing = sum(1 for s in sizes if s == 1)
    cov = sum(s for s in sizes if s >= 2)
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
    print(f"{name:42} clusters={len(sizes):6} cov={100*cov/n:4.0f}% "
          f"price={100*pok/n:4.0f}% purity={100*num/max(1,den):4.0f}% "
          f"sing={100*sing/n:3.0f}%")


def main():
    items, emb = load()
    print(f"items (cached): {len(items)}\n")
    tmp = Path("/tmp/ab2.db")
    with connect(tmp) as conn:
        # BASELINE: current partition + refine
        base = base_partition(items, emb, "current")
        m0 = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)
        metrics(m0, items, "BASELINE (cat-or-brand + refine)")

        # #1: cat+brand partition + refine
        base1 = base_partition(items, emb, "catbrand")
        m1 = cl.refine_clusters(items, dict(base1), conn, log_fn=lambda *a: None, emb_cache=emb)
        metrics(m1, items, "#1 cat+brand partition + refine")

        # #3: baseline + sub-split big clusters
        m3, ns = subsplit(m0, items, emb, min_size=40)
        metrics(m3, items, f"#3 baseline + sub-split (>40, split {ns})")

        # #1 + #3 combined
        m13, ns2 = subsplit(m1, items, emb, min_size=40)
        metrics(m13, items, f"#1 + #3 combined (split {ns2})")


if __name__ == "__main__":
    main()
