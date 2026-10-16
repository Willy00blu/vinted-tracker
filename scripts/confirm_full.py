"""Confirm the searched best model on the FULL dataset (all ~208k items), vs the
production baseline. Partitions the full corpus once (sharded → cached
raw_full.json) and benchmarks baseline vs best_model.json.

Memory note: builds a float32 emb dict for all items (~850MB). Run alone.
"""
import sys, os, json, time, tempfile, sqlite3, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import SCHEMA
from bench import benchmark
from ab_search import KNOBS, DEFAULTS

CACHE = ROOT / "data" / "ab_cache"
RAW_FULL = CACHE / "raw_full.json"


def load_full():
    E = np.load(CACHE / "emb_f16.npy", mmap_mode="r")
    ids = np.load(CACHE / "ids.npy"); row = {int(i): j for j, i in enumerate(ids)}
    items = [it for it in pickle.load(open(CACHE / "items.pkl", "rb")) if it["id"] in row]
    emb = {it["id"]: E[row[it["id"]]].astype(np.float32) for it in items}
    img_ids = np.load(CACHE / "img_ids.npy"); IMG = np.load(CACHE / "img_f16.npy", mmap_mode="r")
    irow = {int(i): j for j, i in enumerate(img_ids)}
    img = {it["id"]: IMG[irow[it["id"]]].astype(np.float32) for it in items if it["id"] in irow}
    return items, emb, img


def temp_conn():
    fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd)
    c = sqlite3.connect(p); c.row_factory = sqlite3.Row; c.executescript(SCHEMA)
    try: c.execute("ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT")
    except sqlite3.OperationalError: pass
    return c, p


def raw_full(items, emb):
    if RAW_FULL.exists():
        return {int(k): v for k, v in json.load(open(RAW_FULL)).items()}
    c, p = temp_conn(); t0 = time.time()
    cl.PARALLEL_SHARD_LARGE = True
    cl.DISABLE_POOL = True          # single-process partition (no spawn deadlock)
    m = cl.cluster_partitioned(items, conn=c, min_cluster_size=2,
                               log_fn=print, emb_cache=emb)
    cl.PARALLEL_SHARD_LARGE = False
    cl.DISABLE_POOL = False
    c.close(); os.unlink(p)
    json.dump({str(k): v for k, v in m.items()}, open(RAW_FULL, "w"))
    print(f"[partition] full {len(set(m.values()))} clusters in {time.time()-t0:.0f}s", flush=True)
    return m


def apply(cfg):
    for k in KNOBS:
        setattr(cl, k, cfg.get(k, DEFAULTS[k]))


def refine(items, raw, emb, img, image=False):
    c, p = temp_conn()
    for it in items:
        it["cluster_key"] = raw.get(it["id"], "unknown")
    m = cl.refine_clusters(items, dict(raw), c, log_fn=lambda *a, **k: None, emb_cache=emb)
    if image and img:
        m = cl.image_refine(m, items, img, log_fn=lambda *a, **k: None)
    c.close(); os.unlink(p)
    return m


def show(items, m, emb, label):
    b = benchmark(items, m, emb)
    print(f"{label:16} OBJ={b['OBJ']:.4f} ABQ={b['ABQ']:.4f} cov={b['coverage']:.1f}% "
          f"uni={b['unimodal']:.3f} mix={b['mix_pct']:.1f}% spec={b['specpure']:.3f} "
          f"sep={b['sep']:.3f} nclu={b['n_clusters']}", flush=True)
    return b


def main():
    image = os.environ.get("AB_IMAGE", "0") == "1"
    items, emb, img = load_full()
    print(f"FULL: {len(items)} items, {len(img)} img, image_refine={image}", flush=True)
    raw = raw_full(items, emb)
    best = json.load(open(CACHE / "best_model.json"))["full_config"]
    apply(DEFAULTS); show(items, refine(items, raw, emb, img, image), emb, "BASELINE")
    apply(best);     show(items, refine(items, raw, emb, img, image), emb, "BEST MODEL")
    apply(DEFAULTS)
    diff = {k: best[k] for k in best if k in DEFAULTS and best[k] != DEFAULTS[k]}
    print("\nbest config (diff vs defaults):", json.dumps(diff))


if __name__ == "__main__":
    main()
