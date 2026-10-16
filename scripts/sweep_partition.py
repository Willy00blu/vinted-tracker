"""Partition-stage sweep: re-partition the subset under different HDBSCAN /
price-feature settings (each ~2 min), refine with the BEST refine config, and
benchmark. Tests levers the refine search can't touch (they shape the raw
clusters): PRICE_FEATURE_LAMBDA (same-product fragmentation), min_cluster_size,
selection method. Appends to leaderboard; updates best_model.json on new record.
"""
import sys, os, json, time, tempfile, sqlite3, pickle, itertools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import SCHEMA
import ab_lab as L
from bench import benchmark
from ab_search import KNOBS, DEFAULTS, BEST, LEADER

CACHE = ROOT / "data" / "ab_cache"


def temp_conn():
    fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd)
    c = sqlite3.connect(p); c.row_factory = sqlite3.Row; c.executescript(SCHEMA)
    try: c.execute("ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT")
    except sqlite3.OperationalError: pass
    return c, p


def partition(items, emb, mcs):
    c, p = temp_conn()
    cl.PARALLEL_SHARD_LARGE = True
    cl.DISABLE_POOL = True            # main-thread partition (no pool deadlock)
    m = cl.cluster_partitioned(items, conn=c, min_cluster_size=mcs,
                               log_fn=lambda *a, **k: None, emb_cache=emb)
    cl.PARALLEL_SHARD_LARGE = False
    cl.DISABLE_POOL = False
    c.close(); os.unlink(p)
    return m


def refine(items, raw, emb):
    c, p = temp_conn()
    for it in items:
        it["cluster_key"] = raw.get(it["id"], "unknown")
    m = cl.refine_clusters(items, dict(raw), c, log_fn=lambda *a, **k: None, emb_cache=emb)
    c.close(); os.unlink(p)
    return m


def main():
    items, emb, _img = L.load_subset()
    best = json.load(open(BEST))["full_config"]
    best_obj = json.load(open(BEST))["OBJ"]
    print(f"loaded {len(items)} items; current best OBJ={best_obj:.4f}\n", flush=True)

    grid = list(itertools.product(
        [("lambda", v) for v in (0.0, 0.30)],
        [("mcs", v) for v in (2,)],
        [("sel", v) for v in ("eom",)],
    ))
    n = 0
    for (_, lam), (_, mcs), (_, sel) in grid:
        # apply refine best config + partition knobs
        for k in KNOBS:
            setattr(cl, k, best.get(k, DEFAULTS[k]))
        cl.PRICE_FEATURE_LAMBDA = lam
        cl.HDBSCAN_SELECTION = sel
        t0 = time.time()
        print(f"... partitioning lambda={lam} mcs={mcs} sel={sel}", flush=True)
        raw = partition(items, emb, mcs)
        m = refine(items, raw, emb)
        b = benchmark(items, m, emb)
        n += 1
        label = f"lambda={lam} mcs={mcs} sel={sel}"
        rec = {"label": "PART " + label, "OBJ": b["OBJ"], "ABQ": b["ABQ"],
               "HARM": b["HARM"], "cov": b["coverage"], "specpure": b["specpure"],
               "unimodal": b["unimodal"], "mix": b["mix_pct"], "sep": b["sep"],
               "nclu": b["n_clusters"],
               "config": {"PRICE_FEATURE_LAMBDA": lam, "HDBSCAN_SELECTION": sel,
                          "min_cluster_size": mcs,
                          **{k: best[k] for k in best if best[k] != DEFAULTS[k]}}}
        with open(LEADER, "a") as f:
            f.write(json.dumps(rec) + "\n")
        star = ""
        if b["OBJ"] > best_obj:
            best_obj = b["OBJ"]
            json.dump({**rec, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "full_config": {**{k: best.get(k, DEFAULTS[k]) for k in KNOBS},
                                       "PRICE_FEATURE_LAMBDA": lam,
                                       "HDBSCAN_SELECTION": sel, "min_cluster_size": mcs}},
                      open(BEST, "w"), indent=2)
            star = "  ★ NEW BEST"
        print(f"[{n}] {label:30} OBJ={b['OBJ']:.4f} ABQ={b['ABQ']:.4f} cov={b['coverage']:.1f}% "
              f"uni={b['unimodal']:.3f} mix={b['mix_pct']:.1f}% ({time.time()-t0:.0f}s){star}", flush=True)
    # restore defaults
    for k in KNOBS:
        setattr(cl, k, DEFAULTS[k])
    cl.PRICE_FEATURE_LAMBDA = 0.30; cl.HDBSCAN_SELECTION = "eom"
    print(f"\nDONE. best OBJ={best_obj:.4f}")


if __name__ == "__main__":
    main()
