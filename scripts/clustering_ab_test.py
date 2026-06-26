"""
Historical A/B test: regex+embedding (legacy) vs embedding-only (current).

Run on 2026-05-19 on the 10572-item Tecnologia DB:
   A (regex+embedding):  5259 clusters · 80.8% singleton · 1009 ≥2 · 253 ≥5
   B (embedding only):   6286 clusters · 80.8% singleton · 1209 ≥2 · 303 ≥5
   → B won (+200 ≥2 clusters, +50 ≥5, same singleton rate, better price coherence).
   → Regex stage removed from core/clustering.py.

This script is kept as a regression baseline. The "regex+embedding" branch
now executes the same code as "embedding-only" since RULES are gone — both
branches will print identical metrics; the script is useful only to verify
the embedding pipeline still produces the same numbers as the 2026-05-19 run.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running this file directly (no -m) from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.clustering import (        # noqa: E402
    SIMILARITY_THRESHOLD,
    _get_model,
    _emb_to_blob,
    _blob_to_emb,
    _cosine,
    _make_embedding_cluster_key,
    _naive_cluster_key,
    _normalize,
    cluster_titles_batch,
    EMBEDDINGS_AVAILABLE,
)
from core.category_db import SCHEMA  # noqa: E402

import numpy as np

DB_PATH = ROOT / "data" / "category_tecnologia.db"


# ─── Embedding-only variant ─────────────────────────────────────────────────

def cluster_embedding_only(items: list[dict], conn: sqlite3.Connection,
                           threshold: float = SIMILARITY_THRESHOLD,
                           ) -> dict[int, str]:
    """Same incremental embedding logic as cluster_titles_batch, but NO regex
    fast-path. Every item goes through the embedding pipeline."""
    result: dict[int, str] = {}
    if not items:
        return result

    centroids: dict[str, "np.ndarray"] = {}
    for row in conn.execute("SELECT cluster_key, embedding FROM cluster_centroids"):
        try:
            centroids[row[0]] = _blob_to_emb(row[1])
        except Exception:
            continue

    model       = _get_model()
    titles_norm = [_normalize(it.get("title", "")) for it in items]
    embeddings  = model.encode(titles_norm, show_progress_bar=False,
                               convert_to_numpy=True)

    upd: list[tuple] = []
    ins: list[tuple] = []
    for it, emb in zip(items, embeddings):
        best_key, best_sim = None, -1.0
        for ck, c_emb in centroids.items():
            sim = _cosine(emb, c_emb)
            if sim > best_sim:
                best_sim, best_key = sim, ck
        if best_key is not None and best_sim >= threshold:
            new_centroid = (centroids[best_key] + emb) / 2.0
            centroids[best_key] = new_centroid
            result[it["id"]] = best_key
            upd.append((_emb_to_blob(new_centroid), "now", best_key))
        else:
            base = _make_embedding_cluster_key(it.get("title", ""))
            key = base
            n = 0
            while key in centroids:
                n += 1
                key = f"{base}#{n}"
            centroids[key] = emb
            result[it["id"]] = key
            ins.append((key, _emb_to_blob(emb), "now", None))
    return result


# ─── Metrics ────────────────────────────────────────────────────────────────

def metrics(cluster_map: dict[int, str], items: list[dict]) -> dict:
    sizes = Counter(cluster_map.values())
    n_clusters = len(sizes)
    n_singleton = sum(1 for v in sizes.values() if v == 1)
    n_ge2 = sum(1 for v in sizes.values() if v >= 2)
    n_ge5 = sum(1 for v in sizes.values() if v >= 5)
    n_items = len(items)
    return {
        "n_clusters":      n_clusters,
        "n_singleton":     n_singleton,
        "n_ge2":           n_ge2,
        "n_ge5":           n_ge5,
        "singleton_pct":   100.0 * n_singleton / max(1, n_clusters),
        "items_covered":   n_items,
        "sizes":           sizes,
    }


def top_clusters(cluster_map, items, top_n=15, n_titles=5):
    sizes = Counter(cluster_map.values())
    out = []
    by_key: dict[str, list[str]] = {}
    for it in items:
        k = cluster_map.get(it["id"], "?")
        by_key.setdefault(k, []).append(it["title"])
    for key, size in sizes.most_common(top_n):
        titles = by_key[key][:n_titles]
        out.append((key, size, titles))
    return out


# ─── Runner ─────────────────────────────────────────────────────────────────

def load_items() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, brand FROM items WHERE title IS NOT NULL"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_pipeline(name: str, fn, items: list[dict]) -> tuple[dict, float, dict]:
    # Fresh in-memory DB so each run starts with zero centroids
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    t0 = time.time()
    mapping = fn(items, conn)
    elapsed = time.time() - t0
    m = metrics(mapping, items)
    conn.close()
    return mapping, elapsed, m


def main():
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available — abort", file=sys.stderr)
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    items = load_items()
    print(f"Loaded {len(items)} items from {DB_PATH.name}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}\n")

    # Method A: current pipeline (regex → embedding → naive)
    print("── A) regex + embedding (current) ─────────────────────────────")
    map_a, t_a, m_a = run_pipeline("current", cluster_titles_batch, items)
    print(f"  time:        {t_a:.1f}s")
    print(f"  clusters:    {m_a['n_clusters']}")
    print(f"  singleton:   {m_a['n_singleton']} ({m_a['singleton_pct']:.1f}%)")
    print(f"  ≥2 items:    {m_a['n_ge2']}")
    print(f"  ≥5 items:    {m_a['n_ge5']}")

    # Method B: embedding-only (no regex)
    print("\n── B) embedding only ──────────────────────────────────────────")
    map_b, t_b, m_b = run_pipeline("embed_only", cluster_embedding_only, items)
    print(f"  time:        {t_b:.1f}s")
    print(f"  clusters:    {m_b['n_clusters']}")
    print(f"  singleton:   {m_b['n_singleton']} ({m_b['singleton_pct']:.1f}%)")
    print(f"  ≥2 items:    {m_b['n_ge2']}")
    print(f"  ≥5 items:    {m_b['n_ge5']}")

    # ── Comparison ─────────────────────────────────────────────────────────
    print("\n── Comparison ─────────────────────────────────────────────────")
    delta = lambda a, b: ("+" if b - a >= 0 else "") + str(b - a)
    print(f"  Δ clusters:   A={m_a['n_clusters']}  B={m_b['n_clusters']}  ({delta(m_a['n_clusters'], m_b['n_clusters'])})")
    print(f"  Δ singleton%: A={m_a['singleton_pct']:.1f}  B={m_b['singleton_pct']:.1f}  ({(m_b['singleton_pct']-m_a['singleton_pct']):+.1f} pp)")
    print(f"  Δ ≥2:         A={m_a['n_ge2']}  B={m_b['n_ge2']}  ({delta(m_a['n_ge2'], m_b['n_ge2'])})")
    print(f"  Δ ≥5:         A={m_a['n_ge5']}  B={m_b['n_ge5']}  ({delta(m_a['n_ge5'], m_b['n_ge5'])})")

    # ── Qualitative top clusters ───────────────────────────────────────────
    print("\n── Top 15 clusters — A (regex+embedding) ──────────────────────")
    for key, sz, titles in top_clusters(map_a, items, top_n=15, n_titles=4):
        print(f"  [{sz:3d}] {key}")
        for t in titles:
            print(f"        · {t[:80]}")

    print("\n── Top 15 clusters — B (embedding only) ───────────────────────")
    for key, sz, titles in top_clusters(map_b, items, top_n=15, n_titles=4):
        print(f"  [{sz:3d}] {key}")
        for t in titles:
            print(f"        · {t[:80]}")

    # ── Items where A and B disagree on cluster assignment ─────────────────
    same = sum(1 for k in map_a if map_a[k] == map_b.get(k))
    print(f"\n── Agreement ──────────────────────────────────────────────────")
    print(f"  Same cluster_key in both: {same}/{len(items)} ({100*same/len(items):.1f}%)")
    print("  NOTE: different keys can still mean *same logical cluster* — A uses")
    print("        'apple iphone 13 pro', B might generate 'iphone 13 pro nuovo'.")


if __name__ == "__main__":
    main()
