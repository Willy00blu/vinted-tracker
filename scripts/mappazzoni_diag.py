"""Diagnostic report on residual mappazzoni after a regen.

For every cluster size>=N with price CV>X, prints why image_refine did or
did not split it: coverage of photo_urls and image_emb, Jenks2 stats,
whether the cluster already carries '#img' children (so image_refine HAS
intervened — splits visible in the table). Tells you at a glance whether
a mappazzone is stuck because:
  - it needs more photos (low image_emb coverage)
  - the Jenks2 gate excluded it (continuous price distribution → "no signal")
  - image_refine ran but couldn't find visual sub-structure (img-cos guard
    blocked, or all sub-clusters were noise)

Run: python -m scripts.mappazzoni_diag
"""
from __future__ import annotations

import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import connect, shared_db_path     # noqa: E402
from core.clustering import _jenks2_stats                 # noqa: E402

MIN_SIZE = 60       # consider clusters this big
MIN_CV   = 0.6      # consider only price-incoherent ones


def main():
    db = shared_db_path()
    with connect(db) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT id, cluster_key, price, photo_urls, image_emb "
            "FROM items WHERE removed=0 AND cluster_key NOT IN ('unknown','')"
        )]
    cl = defaultdict(list)
    for r in rows:
        cl[r["cluster_key"]].append(r)

    # Aggregate per cluster
    mappazzoni: list[tuple] = []
    for k, v in cl.items():
        if len(v) < MIN_SIZE:
            continue
        prs = [r["price"] for r in v if r["price"]]
        if len(prs) < 4:
            continue
        m = st.mean(prs); cv = (st.pstdev(prs) / m) if m else 0
        if cv < MIN_CV:
            continue
        n_url = sum(1 for r in v if r["photo_urls"])
        n_emb = sum(1 for r in v if r["image_emb"])
        stats = _jenks2_stats(prs)
        gvf = stats[1] if stats else 0
        ratio = stats[2] if stats else 0
        # Has image_refine already split this CHILDREN? — children of `k` would
        # be in the DB with `<k> #imgN` suffix.
        n_img_children = sum(1 for kk in cl if kk.startswith(f"{k} #img"))
        mappazzoni.append((k, len(v), m, cv, n_url, n_emb, gvf, ratio, n_img_children))

    mappazzoni.sort(key=lambda x: -x[1])

    print(f"{'cluster':<55} {'N':>4} {'med€':>5} {'CV':>4} "
          f"{'foto%':>5} {'emb%':>4} {'GVF':>4} {'rat':>4} {'#img':>4}  verdict")
    print("-" * 120)
    for k, n, m, cv, nu, ne, gvf, ratio, ni in mappazzoni:
        pct_url = 100 * nu / n
        pct_emb = 100 * ne / n
        # Verdict
        if ni > 0:
            verdict = f"✅ image_refine ha già spaccato in {ni} sub-cluster"
        elif pct_emb < 30:
            verdict = "❌ coverage immagine bassa — serve sold-check backfill"
        elif gvf >= 0.40 and ratio >= 1.5:
            verdict = "⚠️ gate Jenks2 OK ma img-refine ha rifiutato (img-cos guard?)"
        else:
            verdict = "⚠️ no segnale Jenks2 (CV alta ma distribuzione continua)"
        print(f"{k[:55]:<55} {n:>4} {m:>5.0f} {cv:>4.2f} "
              f"{pct_url:>4.0f}% {pct_emb:>3.0f}% {gvf:>4.2f} {ratio:>4.1f} {ni:>4}  {verdict}")
    print()
    print(f"Totale mappazzoni (size>={MIN_SIZE}, CV>{MIN_CV}): {len(mappazzoni)}")

    no_img = sum(1 for x in mappazzoni if (100 * x[5] / x[1]) < 30)
    no_sig = sum(1 for x in mappazzoni if x[6] < 0.4 or x[7] < 1.5)
    already = sum(1 for x in mappazzoni if x[8] > 0)
    print(f"  ✅ già spaccati                 : {already}")
    print(f"  ❌ coverage immagine bassa       : {no_img}")
    print(f"  ⚠️ no segnale Jenks2 (continuo) : {no_sig}")


if __name__ == "__main__":
    main()
