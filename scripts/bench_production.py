"""Benchmark the CURRENT production clustering — the cluster_key actually stored
in the shared DB by the last real regenerate — with the same ABQ/OBJ metric, so
it can be compared apples-to-apples against the searched best model.
"""
import sys, os, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.category_db import shared_db_path, connect
from bench import benchmark

CACHE = ROOT / "data" / "ab_cache"
E = np.load(CACHE / "emb_f16.npy", mmap_mode="r")
ids = np.load(CACHE / "ids.npy"); row = {int(i): j for j, i in enumerate(ids)}
emb = {}
items = []
with connect(shared_db_path()) as c:
    for r in c.execute("SELECT id,title,brand,price,category,cluster_key FROM items "
                       "WHERE removed=0 AND title IS NOT NULL"):
        iid = r["id"]
        if iid not in row:
            continue
        items.append({"id": iid, "title": r["title"], "brand": r["brand"],
                      "price": r["price"], "category": r["category"]})
        emb[iid] = E[row[iid]].astype(np.float32)
mapping = {}
with connect(shared_db_path()) as c:
    for r in c.execute("SELECT id,cluster_key FROM items WHERE removed=0 AND title IS NOT NULL"):
        if r["id"] in emb:
            mapping[r["id"]] = r["cluster_key"] or "unknown"
b = benchmark(items, mapping, emb)
print(f"PRODUCTION (stored cluster_key, {len(items)} items)")
print(f"  OBJ={b['OBJ']:.4f} ABQ={b['ABQ']:.4f} HARM={b['HARM']:.4f} cov={b['coverage']:.1f}% "
      f"bpure={b['bpure']:.3f} specpure={b['specpure']:.3f} unimodal={b['unimodal']:.3f} "
      f"sep={b['sep']:.3f} mix={b['mix_pct']:.1f}% nclu={b['n_clusters']}")
