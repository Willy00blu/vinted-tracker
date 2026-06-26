"""Dry-run: compare HDBSCAN clustering of one partition (brand=X) WITH vs
WITHOUT the brand token in the embedding (Proposal #1).

For each test brand:
  - take all active items with that brand
  - embed twice: standard (brand kept) vs exclude_brand=X (brand stripped)
  - run HDBSCAN with the same params used by cluster_partitioned
  - print cluster counts, sizes, and 2-3 example titles per cluster
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import connect, shared_db_path        # noqa: E402
from core.clustering import _embed_text, _get_model, _get_hdbscan  # noqa: E402


def _cluster(items, *, exclude_brand: str | None):
    model = _get_model()
    texts = [_embed_text(it, exclude_brand=exclude_brand) for it in items]
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True,
                        batch_size=256)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    H = _get_hdbscan().HDBSCAN
    labels = H(min_cluster_size=2, metric="euclidean",
               cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(embs)
    by_label = {}
    for it, lab in zip(items, labels):
        by_label.setdefault(int(lab), []).append(it)
    return by_label


def _summary(name, by_label, max_clusters=12):
    multi = {l: v for l, v in by_label.items() if l != -1 and len(v) >= 2}
    noise = by_label.get(-1, [])
    sizes = sorted([len(v) for v in multi.values()], reverse=True)
    print(f"  [{name}]  cluster (size≥2): {len(multi)}  "
          f"item in cluster: {sum(sizes)}  noise/singleton: {len(noise)}  "
          f"top sizes: {sizes[:8]}")
    # Print up to max_clusters with examples
    by_size = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:max_clusters]
    for lab, group in by_size:
        sample_titles = [g["title"][:55] for g in group[:3]]
        print(f"    ● label={lab}  {len(group)} item")
        for t in sample_titles:
            print(f"        {t}")


def main():
    db = shared_db_path()
    BRANDS = ["Intel", "Corsair", "Rollei", "Apple", "Samsung"]
    with connect(db) as c:
        for brand in BRANDS:
            rows = [dict(r) for r in c.execute(
                "SELECT id, title, brand, category, catalog_id, price "
                "FROM items WHERE removed=0 AND brand=? AND title IS NOT NULL",
                (brand,)
            )]
            if len(rows) < 30:
                print(f"\n=== {brand}: solo {len(rows)} item — skip ===")
                continue
            print(f"\n{'='*78}")
            print(f"=== BRAND: {brand}  ({len(rows)} item totali) ===")
            print(f"{'='*78}")
            print("\nCLUSTERING senza exclude_brand (= comportamento attuale):")
            old = _cluster(rows, exclude_brand=None)
            _summary("STANDARD", old, max_clusters=8)
            print("\nCLUSTERING con exclude_brand (Proposta #1):")
            new = _cluster(rows, exclude_brand=brand)
            _summary("BRAND-STRIPPED", new, max_clusters=8)


if __name__ == "__main__":
    main()
