"""Statistical DEAL detector — reference implementation (v2, 2026-06-29).

  deal_confidence = z * fit * quality
    z       robust price anomaly: (median_logprice − logprice) / (1.4826·MAD),
            computed against the item's ATTRIBUTE BUCKET (#4: storage GB +
            pro/max/plus/mini variant) when the bucket has ≥4 items, else the
            whole cluster — so a 64GB iPhone isn't a fake deal vs 256GB medians.
    fit     cosine(item embedding, cluster centroid) — same-product guard.
    quality continuous baseline-quality weight (#5): w_size·w_cv, replacing the
            old hard size≥8 gate. Small/noisy clusters contribute at reduced
            confidence instead of being dropped (A/B on 60k subset,
            scripts/test_detector.py: top-200 suspect-FP 3.0%→1.5%,
            flagged candidates +31%).

Runs on the STORED production cluster_key + cached text_emb (no re-clustering).
Usage:  python3.13 scripts/deal_detector.py [--db PATH] [--top N] [--json PATH]
        --json writes {generated_at, deals:[…]} for the web GUI (💎 Deals tab).
"""
import sys, collections, statistics as st, math, re, json, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.category_db import shared_db_path, connect
import core.clustering as cl

MIN_CLUSTER = 6          # hard floor (quality weight handles the rest)
Z_MIN = 3.0
Z_CAP = 10.0             # cap z in the confidence: in ultra-tight clusters MAD→0
                         # explodes z into the hundreds, hijacking the ranking
FIT_MIN = 0.93
# Listings that are NOT the product despite a high title fit (the #1 false
# positive observed on the first real scan: "(Solo box) Inazuma eleven" −75%).
_JUNK_TITLE = re.compile(
    r"solo\s+box|box\s+only|boite\s+vide|boîte\s+vide|scatola\s+vuota|"
    r"empty\s+box|nur\s+karton|leere?\s+verpackung|caja\s+vac[ií]a|"
    r"solo\s+scatola|seulement\s+la?\s+bo[iî]te|no\s+console|senza\s+console|"
    r"sans\s+console|lire\s+description|leggere?\s+descrizione|"
    r"pi[eè]ces?\s+d[ée]tach[ée]es|per\s+pezzi|for\s+parts|defekt|guasto|"
    r"non\s+funzionante|hs\b|ne\s+fonctionne\s+pas", re.I)


def _attrs(title):
    n = cl._normalize(title or "")
    storage = None
    m = re.findall(r"(\d+)\s*(gb|tb)", n)
    if m:
        vals = [int(num) * (1000 if unit == "tb" else 1) for num, unit in m]
        vals = [g for g in vals if 8 <= g <= 4000]
        if vals:
            storage = max(vals)
    toks = set(n.split())
    if "max" in toks and "pro" in toks:
        var = "pro max"
    elif "ultra" in toks:
        var = "ultra"
    elif "max" in toks:
        var = "max"
    elif "pro" in toks:
        var = "pro"
    elif "plus" in toks:
        var = "plus"
    elif "mini" in toks:
        var = "mini"
    else:
        var = ""
    return storage, var


def _robust(logps):
    med = st.median(logps)
    mad = st.median([abs(x - med) for x in logps])
    if mad < 1e-6:
        mad = (st.pstdev(logps) or 1e-6) * 0.6745
    return med, 1.4826 * mad


def detect(items, emb_of, mapping, top_n=30):
    by = collections.defaultdict(list)
    for it in items:
        if it.get("price") and it["price"] > 0 and it["id"] in emb_of:
            by[mapping.get(it["id"], "unknown")].append(it)
    deals = []
    for k, v in by.items():
        if len(v) < MIN_CLUSTER:
            continue
        logps = [math.log(it["price"]) for it in v]
        # continuous quality weight (#5)
        logcv = st.pstdev(logps) / abs(st.mean(logps)) if st.mean(logps) else 1.0
        w_size = min(1.0, max(0.0, (len(v) - 5) / 15.0))
        w_cv = min(1.0, max(0.0, 1 - (logcv - 0.20) / 0.30))
        quality = w_size * w_cv
        if quality <= 0:
            continue
        cen = np.mean([emb_of[it["id"]] for it in v], axis=0)
        cen /= max(float(np.linalg.norm(cen)), 1e-12)
        # attribute buckets (#4)
        buckets = collections.defaultdict(list)
        for it in v:
            buckets[_attrs(it["title"])].append(math.log(it["price"]))
        bstats = {b: _robust(lps) for b, lps in buckets.items() if len(lps) >= 4}
        gmed, gsigma = _robust(logps)
        for it in v:
            if _JUNK_TITLE.search(it.get("title") or ""):
                continue          # empty box / broken / for-parts → not a deal
            e = emb_of[it["id"]]
            fit = float((e / max(np.linalg.norm(e), 1e-12)) @ cen)
            if fit < FIT_MIN:
                continue
            med, sigma = bstats.get(_attrs(it["title"]), (gmed, gsigma))
            z = (med - math.log(it["price"])) / sigma
            if z < Z_MIN:
                continue
            deals.append((min(z, Z_CAP) * fit * quality, z, fit, quality, it,
                          math.exp(med), k, len(v)))
    deals.sort(key=lambda d: -d[0])
    return deals[:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", default=None,
                    help="write results as JSON (for the web GUI)")
    # legacy positional compat: deal_detector.py [db] [top]
    ap.add_argument("pos", nargs="*", default=[])
    args = ap.parse_args()
    db = Path(args.db or (args.pos[0] if args.pos else shared_db_path()))
    top_n = args.top if not args.pos[1:] else int(args.pos[1])
    with connect(db) as c:
        rows = c.execute(
            "SELECT id,title,brand,price,cluster_key,text_emb,photo_urls "
            "FROM items WHERE removed=0 AND title IS NOT NULL AND price > 0 "
            "AND text_emb IS NOT NULL").fetchall()
    items, emb_of, mapping = [], {}, {}
    for r in rows:
        try:
            e = cl._blob_to_emb(r["text_emb"])
        except Exception:
            continue
        photo = (r["photo_urls"] or "").split(",")[0] or None
        items.append({"id": r["id"], "title": r["title"], "price": r["price"],
                      "photo": photo})
        emb_of[r["id"]] = e
        mapping[r["id"]] = r["cluster_key"] or "unknown"
    print(f"{len(items)} priced items, {len(set(mapping.values()))} clusters",
          flush=True)
    deals = detect(items, emb_of, mapping, top_n)
    if args.json:
        out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_items": len(items),
               "deals": [{
                   "conf": round(conf, 2), "z": round(z, 2),
                   "fit": round(fit, 3), "quality": round(q, 2),
                   "id": it["id"], "title": it["title"],
                   "price": it["price"], "median": round(medp, 2),
                   "discount_pct": round(100 * (1 - it["price"] / medp)),
                   "saving": round(medp - it["price"]),
                   "cluster_key": k, "cluster_size": n,
                   "photo": it.get("photo"),
                   "url": f"https://www.vinted.it/items/{it['id']}",
               } for conf, z, fit, q, it, medp, k, n in deals]}
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"✓ {len(out['deals'])} occasioni scritte in {args.json}", flush=True)
    print(f"{'conf':>5} {'z':>4} {'fit':>5} {'q':>4} {'€item':>7} {'€med':>7} "
          f"{'sconto':>6}  prodotto / titolo")
    for conf, z, fit, q, it, medp, k, n in deals[:40]:
        disc = 100 * (1 - it["price"] / medp)
        print(f"{conf:>5.1f} {z:>4.1f} {fit:>5.2f} {q:>4.2f} {it['price']:>7.0f} "
              f"{medp:>7.0f} {disc:>5.0f}%  [{k[-24:]}] {(it['title'] or '')[:34]}")


if __name__ == "__main__":
    main()
