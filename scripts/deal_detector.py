"""Prototype the statistical DEAL detector (user's formula):

  deal = (price is anomalously LOW vs its cluster)  WEIGHTED BY
         (title is genuinely the SAME product as the cluster)

  price anomaly  : robust z = (median_logprice - logprice) / (1.4826 * MAD)
                   (positive = below the cluster's typical price)
  title fit      : cosine(item embedding, cluster centroid)
  deal_score     : price_z * title_fit   (low title_fit → discount the signal,
                   it's likely a different/cheaper product, not a deal)

Shows the top candidates on the real DB so we can eyeball + tune the threshold.
"""
import sys, collections, statistics as st, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_ab2 import base_partition, CACHE

MIN_CLUSTER = 8          # need enough items for a stable median/MAD
Z_MIN = 3.0              # robust sigma below median to consider
FIT_MIN = 0.93           # title must be at least this similar to the cluster


def main():
    d = np.load(CACHE, allow_pickle=True)
    emb = {int(i): vv for i, vv in zip(d["ids"], d["emb"])}
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items "
            "WHERE removed=0 AND title IS NOT NULL AND price > 0")]
    items = [it for it in items if it["id"] in emb]
    with connect(Path("/tmp/deal.db")) as conn:
        base = base_partition(items, emb, "catbrand")
        m = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)
    by = collections.defaultdict(list)
    for it in items:
        by[m[it["id"]]].append(it)

    deals = []
    for k, v in by.items():
        if len(v) < MIN_CLUSTER:
            continue
        lp = [math.log(it["price"]) for it in v]
        med = st.median(lp)
        mad = st.median([abs(x - med) for x in lp])
        if mad < 1e-6:
            mad = (st.pstdev(lp) or 1e-6) * 0.6745   # fallback if MAD=0
        sigma = 1.4826 * mad
        cen = np.mean([emb[it["id"]] for it in v], axis=0)
        cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
        med_price = math.exp(med)
        for it, x in zip(v, lp):
            z = (med - x) / sigma                     # >0 = below typical price
            if z < Z_MIN:
                continue
            e = emb[it["id"]]; fit = float((e / max(np.linalg.norm(e), 1e-12)) @ cen)
            if fit < FIT_MIN:
                continue
            deals.append((z * fit, z, fit, it, med_price, k, len(v)))

    deals.sort(reverse=True)
    print(f"cluster validi (>= {MIN_CLUSTER}): "
          f"{sum(1 for v in by.values() if len(v) >= MIN_CLUSTER)}  |  "
          f"occasioni candidate (z>{Z_MIN}, fit>{FIT_MIN}): {len(deals)}\n")
    print(f"{'score':>5} {'z':>4} {'fit':>5} {'€item':>7} {'€med':>7} {'sconto':>6}  prodotto / titolo")
    for score, z, fit, it, medp, k, n in deals[:30]:
        disc = 100 * (1 - it["price"] / medp)
        print(f"{score:>5.1f} {z:>4.1f} {fit:>5.2f} {it['price']:>7.0f} {medp:>7.0f} "
              f"{disc:>5.0f}%  [{k[-20:]}] {(it['title'] or '')[:36]}")


if __name__ == "__main__":
    main()
