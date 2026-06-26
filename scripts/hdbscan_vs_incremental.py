"""
In-memory A/B comparison: incremental threshold clustering vs HDBSCAN.

Loads all items from data/category_tecnologia.db (read-only — no writes to
the production DB), runs both algorithms on a fresh in-memory sqlite, and
prints a side-by-side metrics report + qualitative top-cluster samples.

Use: python3 scripts/hdbscan_vs_incremental.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.clustering  import (  # noqa: E402
    cluster_titles_batch, cluster_hdbscan,
    EMBEDDINGS_AVAILABLE, HDBSCAN_AVAILABLE,
)
from core.category_db import SCHEMA  # noqa: E402

DB_PATH = ROOT / "data" / "category_tecnologia.db"


def load_items() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, brand, category FROM items WHERE title IS NOT NULL"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def metrics(cluster_map, items):
    sizes = Counter(cluster_map.values())
    n = len(sizes)
    return {
        "n_clusters":    n,
        "singleton":     sum(1 for v in sizes.values() if v == 1),
        "ge2":           sum(1 for v in sizes.values() if v >= 2),
        "ge5":           sum(1 for v in sizes.values() if v >= 5),
        "ge20":          sum(1 for v in sizes.values() if v >= 20),
        "biggest":       max(sizes.values()) if sizes else 0,
    }


def top_clusters(cluster_map, items, k=20):
    sizes = Counter(cluster_map.values())
    by_key: dict[str, list[str]] = {}
    for it in items:
        by_key.setdefault(cluster_map.get(it["id"], "?"), []).append(it["title"])
    return [(key, sz, by_key[key][:3]) for key, sz in sizes.most_common(k)]


def main():
    if not EMBEDDINGS_AVAILABLE:
        print("Need sentence-transformers", file=sys.stderr); sys.exit(1)
    if not HDBSCAN_AVAILABLE:
        print("Need hdbscan", file=sys.stderr); sys.exit(1)
    items = load_items()
    print(f"Loaded {len(items)} items from {DB_PATH.name}\n")

    # ── Run A: incremental threshold ──
    print("── A) INCREMENTAL threshold ────────────────")
    conn_a = sqlite3.connect(":memory:")
    conn_a.row_factory = sqlite3.Row
    conn_a.executescript(SCHEMA)
    t0 = time.time()
    # Use batches of 200 to mirror real-world behavior
    cluster_a: dict[int, str] = {}
    for i in range(0, len(items), 200):
        chunk = items[i:i+200]
        cluster_a.update(cluster_titles_batch(chunk, conn=conn_a))
    t_a = time.time() - t0
    m_a = metrics(cluster_a, items)
    print(f"  time: {t_a:.0f}s")
    print(f"  clusters: {m_a['n_clusters']}  singleton: {m_a['singleton']}"
          f"  ≥2: {m_a['ge2']}  ≥5: {m_a['ge5']}  ≥20: {m_a['ge20']}"
          f"  biggest: {m_a['biggest']}")

    # ── Run B: HDBSCAN ──
    print("\n── B) HDBSCAN density-based ────────────────")
    conn_b = sqlite3.connect(":memory:")
    conn_b.row_factory = sqlite3.Row
    conn_b.executescript(SCHEMA)
    t0 = time.time()
    cluster_b = cluster_hdbscan(items, conn=conn_b, min_cluster_size=2,
                                log_fn=lambda s: print(s))
    t_b = time.time() - t0
    m_b = metrics(cluster_b, items)
    print(f"  time: {t_b:.0f}s")
    print(f"  clusters: {m_b['n_clusters']}  singleton: {m_b['singleton']}"
          f"  ≥2: {m_b['ge2']}  ≥5: {m_b['ge5']}  ≥20: {m_b['ge20']}"
          f"  biggest: {m_b['biggest']}")

    # ── Comparison table ──
    print("\n── Comparison ──────────────────────────────")
    print(f"  Δ clusters:   A={m_a['n_clusters']:>5}  B={m_b['n_clusters']:>5}"
          f"  ({m_b['n_clusters']-m_a['n_clusters']:+d})")
    print(f"  Δ singleton:  A={m_a['singleton']:>5}  B={m_b['singleton']:>5}"
          f"  ({m_b['singleton']-m_a['singleton']:+d})")
    print(f"  Δ ≥2:         A={m_a['ge2']:>5}  B={m_b['ge2']:>5}"
          f"  ({m_b['ge2']-m_a['ge2']:+d})")
    print(f"  Δ ≥5:         A={m_a['ge5']:>5}  B={m_b['ge5']:>5}"
          f"  ({m_b['ge5']-m_a['ge5']:+d})")
    print(f"  Δ ≥20:        A={m_a['ge20']:>5}  B={m_b['ge20']:>5}"
          f"  ({m_b['ge20']-m_a['ge20']:+d})")
    print(f"  biggest:      A={m_a['biggest']:>5}  B={m_b['biggest']:>5}")

    # ── Top samples ──
    print("\n── Top 15 — A (incremental) ────────────────")
    for key, sz, titles in top_clusters(cluster_a, items, k=15):
        print(f"  [{sz:3d}] {key}")
        for t in titles:
            print(f"        · {t[:70]}")
    print("\n── Top 15 — B (HDBSCAN) ────────────────────")
    for key, sz, titles in top_clusters(cluster_b, items, k=15):
        print(f"  [{sz:3d}] {key}")
        for t in titles:
            print(f"        · {t[:70]}")

    # ── Specific case study: iPhone ──
    print("\n── iPhone split? — A items containing 'iphone' ──")
    iphone_ids = {it["id"] for it in items if "iphone" in it["title"].lower()}
    a_iphone = Counter(cluster_a[i] for i in iphone_ids if i in cluster_a)
    b_iphone = Counter(cluster_b[i] for i in iphone_ids if i in cluster_b)
    print(f"  A: {len(a_iphone)} cluster con item 'iphone' ({len(iphone_ids)} item totali)")
    for c, n in a_iphone.most_common(8):
        print(f"      [{n:3d}] {c}")
    print(f"  B: {len(b_iphone)} cluster con item 'iphone' ({len(iphone_ids)} item totali)")
    for c, n in b_iphone.most_common(8):
        print(f"      [{n:3d}] {c}")


if __name__ == "__main__":
    main()
