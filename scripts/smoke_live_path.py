"""End-to-end smoke test of the ALWAYS-ON live path with the new defaults:
INCREMENTAL_RUNNING_MEAN + INCREMENTAL_VECTOR_MATCH + BAND_AWARE_MATCH,
plus the new cluster_centroids.price_median column.

Builds a temp DB (full schema+migrations), persists two price-band siblings
with IDENTICAL text centroids but different medians, then checks that
match_cluster_batch and cluster_titles_batch route items by price.
"""
import sys, os, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import connect

assert cl.INCREMENTAL_RUNNING_MEAN and cl.INCREMENTAL_VECTOR_MATCH and cl.BAND_AWARE_MATCH

db = Path(tempfile.mkdtemp()) / "smoke.db"
model = cl._get_model()
v_iphone = model.encode(["query: apple iphone 15"], convert_to_numpy=True)[0]
v_ps5    = model.encode(["query: sony ps5 console"], convert_to_numpy=True)[0]

rows = [
    ("apple/iphone 15 [€40-60]",   cl._emb_to_blob(v_iphone), 10, "2026-06-29", None, 50.0),
    ("apple/iphone 15 [€400-600]", cl._emb_to_blob(v_iphone), 30, "2026-06-29", None, 500.0),
    ("sony/ps5",                   cl._emb_to_blob(v_ps5),    20, "2026-06-29", None, 400.0),
]
with connect(db) as conn:
    cl._persist_centroids(conn, rows, wipe=True)
    conn.commit()

    items = [
        {"id": 1, "title": "iPhone 15",       "price": 45},    # → cheap band
        {"id": 2, "title": "iPhone 15",       "price": 520},   # → expensive band
        {"id": 3, "title": "iPhone 15",       "price": None},  # no price → cosine
        {"id": 4, "title": "Sony ps5 console", "price": 350},  # single → no-op
    ]
    got = cl.match_cluster_batch(items, conn)
    exp = {1: "apple/iphone 15 [€40-60]", 2: "apple/iphone 15 [€400-600]",
           4: "sony/ps5"}
    ok = True
    for iid, want in exp.items():
        k, s = got[iid]
        mark = "✓" if k == want else "✗"
        ok &= (k == want)
        print(f"  match id={iid} → {k} (sim={s:.3f}) {mark}")
    k3, _ = got[3]
    print(f"  match id=3 (no price) → {k3} (cosine fallback, any band ok)")

    # live insert path: routing + running-mean update persisted
    m = cl.cluster_titles_batch([{"id": 9, "title": "iPhone 15", "price": 42,
                                  "brand": "Apple"}], conn=conn)
    conn.commit()
    print(f"  titles_batch id=9 (€42) → {m[9]} "
          f"{'✓' if m[9] == 'apple/iphone 15 [€40-60]' else '✗'}")
    ok &= (m[9] == "apple/iphone 15 [€40-60]")
    cnt = conn.execute("SELECT item_count FROM cluster_centroids WHERE cluster_key=?",
                       ("apple/iphone 15 [€40-60]",)).fetchone()[0]
    print(f"  item_count after runmean update: {cnt} {'✓' if cnt == 11 else '✗'}")
    ok &= (cnt == 11)

print("SMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
