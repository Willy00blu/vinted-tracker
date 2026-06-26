"""Dry-run combinato: Proposta #1 (brand-token removal) + Proposta #3
(sub-partition catalog_id quando disponibile).

Vs solo #1 e baseline standard, sui brand mappazzone reali.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import connect, shared_db_path        # noqa: E402
from core.clustering import _embed_text, _get_model, _get_hdbscan  # noqa: E402


def _cluster_one_partition(items, exclude_brand):
    if len(items) < 3:
        return {-1: items}
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


def cluster_brand(items_brand, brand, *, use_brand_strip: bool, use_catalog_id: bool):
    """Ritorna list of clusters (multi only)."""
    if use_catalog_id:
        # Sub-partition by catalog_id (None as separate bucket)
        by_part = defaultdict(list)
        for it in items_brand:
            cid = it.get("catalog_id") or 0
            by_part[cid].append(it)
        partitions = list(by_part.items())
    else:
        partitions = [(0, items_brand)]
    exclude = brand if use_brand_strip else None
    out_clusters = []   # list of (key, [items])
    out_singletons = 0
    for cid, p_items in partitions:
        bl = _cluster_one_partition(p_items, exclude)
        for lab, group in bl.items():
            if lab == -1:
                out_singletons += len(group)
            elif len(group) >= 2:
                out_clusters.append((f"cid={cid}_lab={lab}", group))
            else:
                out_singletons += len(group)
    return out_clusters, out_singletons


def summary(name, clusters, singletons, max_show=6):
    sizes = sorted([len(g) for _, g in clusters], reverse=True)
    print(f"  [{name}]  cluster ≥2: {len(clusters)}  "
          f"item in cluster: {sum(sizes)}  singleton: {singletons}  "
          f"top sizes: {sizes[:8]}")
    by_size = sorted(clusters, key=lambda kv: -len(kv[1]))[:max_show]
    for k, group in by_size:
        sample = [g["title"][:55] for g in group[:3]]
        print(f"    ● {k}  {len(group)} item")
        for t in sample:
            print(f"        {t}")


def main():
    db = shared_db_path()
    BRANDS = ["Intel", "Corsair", "Apple", "Samsung"]
    with connect(db) as c:
        for brand in BRANDS:
            rows = [dict(r) for r in c.execute(
                "SELECT id, title, brand, category, catalog_id, price "
                "FROM items WHERE removed=0 AND brand=? AND title IS NOT NULL",
                (brand,)
            )]
            if len(rows) < 30:
                continue
            with_cid = sum(1 for r in rows if r["catalog_id"])
            print(f"\n{'='*78}")
            print(f"BRAND: {brand}  ({len(rows)} item, catalog_id su {with_cid} item = {100*with_cid/len(rows):.0f}%)")
            print('='*78)
            for label, brand_strip, cat_split in [
                ("BASELINE      (no brand-strip, no cat-split)", False, False),
                ("ONLY #1       (brand-strip)",                   True,  False),
                ("ONLY #3       (cat-split)",                     False, True),
                ("COMBO #1+#3   (brand-strip + cat-split)",       True,  True),
            ]:
                cl, ng = cluster_brand(rows, brand, use_brand_strip=brand_strip, use_catalog_id=cat_split)
                print()
                summary(label, cl, ng, max_show=5)


if __name__ == "__main__":
    main()
