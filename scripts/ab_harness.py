"""A/B harness for the clustering/refine pipeline (review interventions).

Runs the REAL pipeline on FIXED cached e5 embeddings (no re-embedding, no
touching the shared DB). Splits the work so the expensive stage-2 raw
partition is computed ONCE and cached; refine-stage interventions reuse it.

Caches (built by scripts/ab_build_cache.py / inline):
  /tmp/ab_emb.npz    ids + 1024-d e5 text vectors
  /tmp/ab_items.pkl  [{id,title,brand,price,category}]
  /tmp/ab_img.pkl    {id: clip vector}
  /tmp/ab_raw_map.json  raw HDBSCAN partition mapping (stage 2 output)
"""
from __future__ import annotations
import sys, os, pickle, json, collections, statistics as st, math, time, tempfile, sqlite3
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")

import core.clustering as cl          # noqa: E402
from core.category_db import SCHEMA   # noqa: E402

EMB_NPZ   = "/tmp/ab_emb.npz"
ITEMS_PKL = "/tmp/ab_items.pkl"
IMG_PKL   = "/tmp/ab_img.pkl"
RAW_MAP   = "/tmp/ab_raw_map.json"


def load():
    d = np.load(EMB_NPZ, allow_pickle=True)
    emb = {int(i): v for i, v in zip(d["ids"], d["emb"])}
    items = pickle.load(open(ITEMS_PKL, "rb"))
    img = pickle.load(open(IMG_PKL, "rb"))
    items = [it for it in items if it["id"] in emb]
    return items, emb, img


def _temp_conn():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        conn.execute("ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT")
    except sqlite3.OperationalError:
        pass
    return conn, path


def raw_partition(items, emb, cache=True):
    if cache and os.path.exists(RAW_MAP):
        m = json.load(open(RAW_MAP))
        return {int(k): v for k, v in m.items()}
    conn, path = _temp_conn()
    t0 = time.time()
    m = cl.cluster_partitioned(items, conn=conn, min_cluster_size=2,
                               log_fn=lambda *a, **k: None, emb_cache=emb)
    conn.close(); os.unlink(path)
    print(f"  [raw_partition] {len(set(m.values()))} clusters in {time.time()-t0:.0f}s")
    if cache:
        json.dump({str(k): v for k, v in m.items()}, open(RAW_MAP, "w"))
    return m


def refine(items, raw_map, emb, img=None, do_image=True):
    """Run the refine + image stages on a copy of the raw mapping."""
    conn, path = _temp_conn()
    for it in items:
        it["cluster_key"] = raw_map.get(it["id"], "unknown")
    m = cl.refine_clusters(items, dict(raw_map), conn,
                           log_fn=lambda *a, **k: None, emb_cache=emb)
    if do_image and img:
        img_of = {i: v for i, v in img.items() if i in {it["id"] for it in items}}
        m = cl.image_refine(m, items, img_of, log_fn=lambda *a, **k: None)
    conn.close(); os.unlink(path)
    return m


# ── partition key (mirror cluster_partitioned base; family-shard ignored: we
#    want the full confusion scope for separation) ────────────────────────────
def part_key(it):
    cat = (it.get("category") or "").strip()
    brand, _ = cl._lex_sig(it.get("title", ""), it.get("brand"))
    brand = (cl._normalize_brand(brand) or brand or "").lower()
    if cat and brand:
        return f"{cat}|{brand}"
    if cat:
        return cat
    return brand or "__nobrand__"


def sep_metric(items, mapping, emb_of, sample=40000, seed=0):
    """Within-partition separation margin: for each item,
       cos(x, own_centroid) - max cos(x, other cluster centroid in same partition).
    Returns (mean_margin, p10_margin, frac_negative)."""
    by = collections.defaultdict(list)
    for it in items:
        k = mapping.get(it["id"])
        if k is not None:
            by[k].append(it["id"])
    # cluster centroid (normalized) + cluster->partition (modal)
    cen = {}
    cpart = {}
    itpart = {it["id"]: part_key(it) for it in items}
    for k, ids in by.items():
        V = np.array([emb_of[i] for i in ids if i in emb_of], dtype=np.float32)
        if len(V) == 0:
            continue
        Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        c = Vn.mean(0); c /= max(float(np.linalg.norm(c)), 1e-12)
        cen[k] = c
        pc = collections.Counter(itpart[i] for i in ids).most_common(1)[0][0]
        cpart[k] = pc
    # group cluster centroids by partition
    part_cens = collections.defaultdict(list)   # part -> [(k, c)]
    for k, c in cen.items():
        part_cens[cpart[k]].append((k, c))
    part_mat = {p: (np.array([c for _, c in lst], dtype=np.float32),
                    [k for k, _ in lst]) for p, lst in part_cens.items()}
    rng = np.random.default_rng(seed)
    pool = [it for it in items if mapping.get(it["id"]) in cen and it["id"] in emb_of]
    if sample and len(pool) > sample:
        idx = rng.choice(len(pool), sample, replace=False)
        pool = [pool[i] for i in idx]
    margins = []
    for it in pool:
        iid = it["id"]; k = mapping[iid]; p = cpart.get(k)
        x = emb_of[iid]; x = x / max(float(np.linalg.norm(x)), 1e-12)
        M, keys = part_mat[p]
        sims = M @ x
        own_i = keys.index(k)
        own = float(sims[own_i])
        if len(keys) == 1:
            other = own            # singleton-partition: no rival -> margin 0
        else:
            sims2 = sims.copy(); sims2[own_i] = -2.0
            other = float(sims2.max())
        margins.append(own - other)
    margins = np.array(margins)
    return (float(margins.mean()), float(np.percentile(margins, 10)),
            float((margins < 0).mean()))


def evaluate(items, mapping, emb, label, do_sep=True):
    m = cl.eval_metrics(items, mapping, emb)
    line = (f"{label:30} SCORE={m['score']:5.1f} ANOM={m['anom_ready']:4.1f}% "
            f"N_IDEAL={m['n_ideal']:4d} cov={m['coverage']:4.1f}% "
            f"coh={m['cohesion']:.3f} bpure={m['brand_pure']:4.1f}% "
            f"sing={m['singleton_pct']:.0f}% nclu={m['cluster']} "
            f"BLOB={m['n_blob']} GIANT={m['n_giant']}")
    if do_sep:
        smean, sp10, sneg = sep_metric(items, mapping, emb)
        line += f"\n{'':30} SEP={smean:.3f} SEP_p10={sp10:.3f} neg={100*sneg:.1f}%"
    print(line)
    return m


if __name__ == "__main__":
    items, emb, img = load()
    print(f"loaded {len(items)} items, {len(img)} image vecs")
    raw = raw_partition(items, emb)
    base = refine(items, raw, emb, img, do_image=True)
    evaluate(items, base, emb, "BASELINE (full pipeline)")
