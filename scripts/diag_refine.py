"""Diagnose WHICH refine stage is slow (verbose + per-stage timing) on the
cached raw partition. No re-embedding, no DB writes beyond a temp centroid table.
"""
import sys, os, json, time, tempfile, sqlite3, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import SCHEMA

d = np.load("/tmp/ab_emb.npz", allow_pickle=True)
emb = {int(i): v for i, v in zip(d["ids"], d["emb"])}
items = [it for it in pickle.load(open("/tmp/ab_items.pkl", "rb")) if it["id"] in emb]
img = pickle.load(open("/tmp/ab_img.pkl", "rb"))
raw = {int(k): v for k, v in json.load(open("/tmp/ab_raw_map.json")).items()}
print(f"items={len(items)} clusters={len(set(raw.values()))} img={len(img)}", flush=True)

fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
conn.executescript(SCHEMA)
try: conn.execute("ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT")
except sqlite3.OperationalError: pass

for it in items:
    it["cluster_key"] = raw.get(it["id"], "unknown")

t0 = time.time()
print("\n[refine_clusters] start (verbose)…", flush=True)
m = cl.refine_clusters(items, dict(raw), conn, log_fn=lambda *a, **k: print("   ", *a, flush=True), emb_cache=emb)
t_ref = time.time() - t0
print(f"[refine_clusters] DONE in {t_ref:.0f}s  → {len(set(m.values()))} clusters", flush=True)

img_of = {i: v for i, v in img.items() if i in {it["id"] for it in items}}
t1 = time.time()
print(f"\n[image_refine] start on {len(img_of)} img vecs…", flush=True)
m2 = cl.image_refine(m, items, img_of, log_fn=lambda *a, **k: print("   ", *a, flush=True))
t_img = time.time() - t1
print(f"[image_refine] DONE in {t_img:.0f}s  → {len(set(m2.values()))} clusters", flush=True)

conn.close(); os.unlink(path)
json.dump({str(k): v for k, v in m2.items()}, open("/tmp/ab_final_base.json", "w"))
print(f"[dumped baseline final mapping -> /tmp/ab_final_base.json]", flush=True)
print(f"\nTOTAL refine+image = {t_ref+t_img:.0f}s  (refine={t_ref:.0f}s image={t_img:.0f}s)", flush=True)
