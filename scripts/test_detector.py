"""#4 (attribute-conditioned deal baseline) + #5 (continuous baseline-quality
weight) — tested on the FINAL pipeline mapping.

Current detector (scripts/deal_detector.py): per cluster, robust z on log-price
  z = (median_logprice - logprice) / (1.4826*MAD);  deal = z>=Z_MIN AND fit>=FIT_MIN.

#4: condition the baseline on lexical attributes (storage GB, pro/max/plus/
    ultra/mini variant) before computing z, so a cheaper LOWER-spec variant is
    not flagged as a deal.
#5: replace the binary size>=8 gate with a continuous baseline-quality weight
    q(size, logCV); rank by z*fit*q and report expanded usable coverage.

No human labels → FP proxy: a flagged "deal" is SUSPECT (likely a smaller-spec
variant, not a real discount) if its storage is strictly below the cluster's
dominant storage, OR its variant rank is below the dominant variant rank. #4
should cut the suspect rate at equal/again recall.
"""
import sys, os, json, math, collections, statistics as st, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl

CACHE = ROOT / "data" / "ab_cache"
MAP_JSON = str(CACHE / "final_best_map.json")
Z_MIN = 3.0
FIT_MIN = 0.93
MIN_CLUSTER = 8

_VARIANT_RANK = {"mini": 0, "": 1, "plus": 2, "pro": 2, "max": 3,
                 "pro max": 4, "ultra": 4}


def _attrs(title):
    n = cl._normalize(title)
    storage = None
    m = re.findall(r"(\d+)\s*(gb|tb)", n)
    if m:
        vals = []
        for num, unit in m:
            g = int(num) * (1000 if unit == "tb" else 1)
            if 8 <= g <= 4000:           # plausible storage range
                vals.append(g)
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


def load():
    import pickle
    E = np.load(CACHE / "emb_f16.npy", mmap_mode="r")
    ids = np.load(CACHE / "ids.npy"); row = {int(i): j for j, i in enumerate(ids)}
    m = {int(k): v for k, v in json.load(open(MAP_JSON)).items()}
    emb = {iid: E[row[iid]].astype(np.float32) for iid in m if iid in row}
    items = {it["id"]: it for it in pickle.load(open(CACHE / "items.pkl", "rb"))
             if it["id"] in emb}
    return emb, items, m


def norm(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _robust(logps):
    med = st.median(logps)
    mad = st.median([abs(x - med) for x in logps])
    if mad < 1e-6:
        mad = (st.pstdev(logps) or 1e-6) * 0.6745
    return med, 1.4826 * mad


def detect(emb, items, mapping, mode):
    """mode in {'current','attr','weighted'}. Returns list of deals as dicts."""
    min_cluster = 6 if mode == "weighted" else MIN_CLUSTER   # #5 relaxes the floor
    by = collections.defaultdict(list)
    for iid, k in mapping.items():
        it = items.get(iid)
        if it and it.get("price") and it["price"] > 0 and iid in emb:
            by[k].append(it)
    deals = []
    for k, v in by.items():
        if len(v) < min_cluster:
            continue
        logps = [math.log(it["price"]) for it in v]
        cen = norm(np.mean([emb[it["id"]] for it in v], axis=0))
        # cluster dominant attributes (for FP proxy + #4 bucketing)
        storages = [s for s, _ in (_attrs(it["title"]) for it in v) if s]
        dom_storage = collections.Counter(storages).most_common(1)[0][0] if storages else None
        varr = [_VARIANT_RANK.get(_attrs(it["title"])[1], 1) for it in v]
        dom_var = collections.Counter(varr).most_common(1)[0][0]
        # baseline-quality weight (#5)
        logcv = st.pstdev(logps) / abs(st.mean(logps)) if st.mean(logps) else 1.0
        w_size = min(1.0, max(0.0, (len(v) - 5) / 15.0))
        w_cv = min(1.0, max(0.0, 1 - (logcv - 0.20) / 0.30))
        bq = w_size * w_cv
        # #4 bucket medians
        bucket_stats = {}
        if mode == "attr":
            buck = collections.defaultdict(list)
            for it in v:
                s, var = _attrs(it["title"])
                buck[(s, var)].append(math.log(it["price"]))
            for b, lps in buck.items():
                if len(lps) >= 4:
                    bucket_stats[b] = _robust(lps)
        gmed, gsigma = _robust(logps)
        for it in v:
            x = norm(emb[it["id"]])
            fit = float(x @ cen)
            if fit < FIT_MIN:
                continue
            lp = math.log(it["price"])
            if mode == "attr":
                s, var = _attrs(it["title"])
                bs = bucket_stats.get((s, var))
                med, sigma = bs if bs else (gmed, gsigma)
            else:
                med, sigma = gmed, gsigma
            z = (med - lp) / sigma
            score = z * fit
            if mode == "weighted":
                if z < Z_MIN or fit < FIT_MIN:
                    continue
                score = z * fit * bq
            else:
                if z < Z_MIN:
                    continue
            s_it, var_it = _attrs(it["title"])
            suspect = False
            if dom_storage and s_it and s_it < dom_storage:
                suspect = True
            if _VARIANT_RANK.get(var_it, 1) < dom_var:
                suspect = True
            deals.append({"score": score, "z": z, "fit": fit, "k": k,
                          "title": it["title"], "price": it["price"],
                          "suspect": suspect, "bq": bq, "n": len(v)})
    deals.sort(key=lambda d: -d["score"])
    return deals


def report(deals, name, topk=200):
    top = deals[:topk]
    susp = sum(1 for d in top if d["suspect"])
    print(f"  {name:10} total_deals={len(deals):5d}  top{topk}_suspectFP={100*susp/max(1,len(top)):5.1f}%  "
          f"(suspect {susp}/{len(top)})")


def main():
    emb, items, mapping = load()
    print(f"loaded {len(emb)} emb, {len(set(mapping.values()))} clusters")
    cur = detect(emb, items, mapping, "current")
    att = detect(emb, items, mapping, "attr")
    wei = detect(emb, items, mapping, "weighted")
    print("FP proxy = flagged deal whose storage/variant is BELOW the cluster mode")
    report(cur, "current")
    report(att, "attr(#4)")
    report(wei, "weighted(#5)")
    # coverage: usable clusters
    print(f"  weighted(#5) flags {len(wei)} vs current {len(cur)} "
          f"(extra come from size 6-9 / moderate-CV clusters at reduced weight)")


if __name__ == "__main__":
    main()
