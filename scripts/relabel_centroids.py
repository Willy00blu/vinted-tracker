"""Re-compute display_label for every cluster, applying the 30% category
quorum rule (a category label is adopted only if at least 30% of the items
carry it). Fixes ~2000 clusters that were mis-labeled because a single
mis-categorized listing dictated the label of the whole cluster.

No re-clustering, no embedding — just an UPDATE on cluster_centroids.

Run: python -m scripts.relabel_centroids
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import shared_db_path, connect    # noqa: E402
from core.clustering import _display_label              # noqa: E402

CATEGORY_QUORUM = 0.30


def main():
    db = shared_db_path()
    with connect(db) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT cluster_key, category FROM items "
            "WHERE removed=0 AND cluster_key IS NOT NULL AND cluster_key != 'unknown'")]
    print(f"  read {len(rows)} items")
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster_key"]].append(r["category"] or "")
    print(f"  {len(by_cluster)} unique clusters")

    updates: list[tuple[str | None, str]] = []
    for key, cats in by_cluster.items():
        counter = Counter(c for c in cats if c)
        top_cat = None
        if counter:
            best_cat, best_n = counter.most_common(1)[0]
            if best_n / max(1, len(cats)) >= CATEGORY_QUORUM:
                top_cat = best_cat
        # Use the readable suffix (after the partition prefix) as the product name
        readable = key.split("/", 1)[-1] if "/" in key else key
        new_label = _display_label(readable, top_cat)
        updates.append((new_label, key))

    print(f"  applying {len(updates)} label updates…")
    with connect(db) as c:
        c.executemany(
            "UPDATE cluster_centroids SET display_label = ? WHERE cluster_key = ?",
            updates,
        )
        c.commit()
    print(f"  done")


if __name__ == "__main__":
    main()
