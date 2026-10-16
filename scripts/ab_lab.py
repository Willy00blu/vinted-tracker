"""Fast A/B lab: load cache once, partition the subset once (cached), then run
any flag config through refine+benchmark in seconds.

Usage:
  python3.13 scripts/ab_lab.py                 # run the full variant battery
  python3.13 scripts/ab_lab.py --full <map.json>   # benchmark a precomputed full map
The expensive HDBSCAN partition is cached to data/ab_cache/raw_sub.json; every
refine-stage variant reuses it (instant). Memory-frugal: float16 base matrix,
float32 dict only for the ~60k subset.
"""
import sys, os, json, pickle, time, tempfile, sqlite3
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import SCHEMA
from bench import benchmark
from ab_harness import sep_metric

CACHE = ROOT / "data" / "ab_cache"
# CLI flags (preferred — keeps every command starting with `python3.13` so the
# permission allowlist matches) with env fallback.
RUN_IMAGE = ("--image" in sys.argv) or os.environ.get("AB_IMAGE", "0") == "1"
SHARD = ("--no-shard" not in sys.argv) and os.environ.get("AB_SHARD", "1") == "1"
RAW_SUB = CACHE / ("raw_sub_shard.json" if SHARD else "raw_sub.json")


def load_subset():
    E = np.load(CACHE / "emb_f16.npy", mmap_mode="r")
    ids = np.load(CACHE / "ids.npy")
    row = {int(i): j for j, i in enumerate(ids)}
    items_all = pickle.load(open(CACHE / "items.pkl", "rb"))
    sub = set(json.load(open(CACHE / "subset.json"))["ids"])
    items = [it for it in items_all if it["id"] in sub]
    emb = {it["id"]: E[row[it["id"]]].astype(np.float32) for it in items}
    img_ids = np.load(CACHE / "img_ids.npy")
    IMG = np.load(CACHE / "img_f16.npy", mmap_mode="r")
    irow = {int(i): j for j, i in enumerate(img_ids)}
    img = {it["id"]: IMG[irow[it["id"]]].astype(np.float32)
           for it in items if it["id"] in irow}
    return items, emb, img


def _temp_conn():
    fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd)
    conn = sqlite3.connect(p); conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try: conn.execute("ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT")
    except sqlite3.OperationalError: pass
    return conn, p


def raw_partition(items, emb):
    if RAW_SUB.exists():
        return {int(k): v for k, v in json.load(open(RAW_SUB)).items()}
    conn, p = _temp_conn(); t0 = time.time()
    cl.PARALLEL_SHARD_LARGE = SHARD          # parallelize the giant partitions
    m = cl.cluster_partitioned(items, conn=conn, min_cluster_size=2,
                               log_fn=lambda *a, **k: None, emb_cache=emb)
    cl.PARALLEL_SHARD_LARGE = False
    conn.close(); os.unlink(p)
    json.dump({str(k): v for k, v in m.items()}, open(RAW_SUB, "w"))
    print(f"  [partition] {len(set(m.values()))} clusters in {time.time()-t0:.0f}s "
          f"→ cached", flush=True)
    return m


def refine(items, raw, emb, img):
    conn, p = _temp_conn()
    for it in items:
        it["cluster_key"] = raw.get(it["id"], "unknown")
    m = cl.refine_clusters(items, dict(raw), conn, log_fn=lambda *a, **k: None,
                           emb_cache=emb)
    if RUN_IMAGE and img:
        m = cl.image_refine(m, items, img, log_fn=lambda *a, **k: None)
    conn.close(); os.unlink(p)
    return m


def reset():
    for f in ("ENABLE_MODEL_TOKEN_SPLIT", "STABLE_MERGE_KEY", "LOGSPACE_BITAIL",
              "CONSOLIDATE_PRICE_SPLIT", "PARALLEL_SHARD_LARGE", "DISABLE_PRICE_SPLIT",
              "DISABLE_BITAIL"):
        setattr(cl, f, False)


def evaluate(items, m, emb, label):
    b = benchmark(items, m, emb)
    print(f"{label:24} ABQ={b['ABQ']:.3f} cov={b['coverage']:4.1f}% "
          f"bpure={b['bpure']:.3f} spec={b['specpure']:.3f} uni={b['unimodal']:.3f} "
          f"sep={b['sep']:.3f} mix={b['mix_pct']:4.1f}% nclu={b['n_clusters']}",
          flush=True)
    return b


def main():
    items, emb, img = load_subset()
    print(f"subset: {len(items)} items, {len(img)} img vecs, image_refine={RUN_IMAGE}\n", flush=True)
    raw = raw_partition(items, emb)
    # sanity: the raw partition (no refine) is the floor; refine must beat it.
    evaluate(items, {it["id"]: raw.get(it["id"], "unknown") for it in items},
             emb, "RAW (no refine)")
    variants = [
        ("BASELINE", {}),
        ("#2 model-token", {"ENABLE_MODEL_TOKEN_SPLIT": True}),
        ("#8 stable-merge", {"STABLE_MERGE_KEY": True}),
        ("#6 bitail-log", {"LOGSPACE_BITAIL": True}),
        ("#10 drop-2nd-jenks", {"CONSOLIDATE_PRICE_SPLIT": True}),
        ("no-bitail (keep Jenks2)", {"DISABLE_BITAIL": True}),
        ("no-price-split", {"DISABLE_PRICE_SPLIT": True}),
        ("#8 + no-bitail", {"STABLE_MERGE_KEY": True, "DISABLE_BITAIL": True}),
        ("ALL", {"ENABLE_MODEL_TOKEN_SPLIT": True, "STABLE_MERGE_KEY": True,
                 "LOGSPACE_BITAIL": True}),
    ]
    for name, flags in variants:
        reset()
        for f, val in flags.items():
            setattr(cl, f, val)
        t0 = time.time()
        m = refine(items, raw, emb, img)
        b = evaluate(items, m, emb, name)
        print(f"{'':26} ({time.time()-t0:.0f}s)", flush=True)
    reset()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--full":
        items, emb, _ = load_subset()  # emb dict only for subset; full needs its own
        print("use bench.py for full maps", flush=True)
    else:
        main()
