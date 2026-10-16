"""#1 (running-mean centroid) + #7 (vectorized match) — test the ALWAYS-ON
incremental path, which the regen benchmarks never see. Simulates the monitor
running BETWEEN two regenerations:

  - seed each cluster's centroid from half its items (= state just after regen),
  - stream the other half in random order, assign to nearest centroid (cos,
    threshold 0.82), update the assigned centroid per the rule under test.

Rules: static (no update, no-drift reference) | ema (production `(old+emb)/2`)
| runmean (fix #1: `(old*n+emb)/(n+1)`).

Metrics:
  acc_exact : streamed item routed to its exact cluster
  acc_fam   : routed to the right base FAMILY (price-band tags [€..]/#high/#low
              stripped) — text centroids of sibling bands are ~identical, so
              exact-band routing is structurally hard for a text-only matcher
              (that gap is what task #7 band-aware routing must close)
  drift     : cos(final centroid, true full-cluster mean) — 1.0 is perfect

Uses data/ab_cache fp16 embeddings + final_best_map.json (best-model mapping).
"""
import sys, os, json, time, collections, random, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
CACHE = ROOT / "data" / "ab_cache"
MAP_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE / "final_best_map.json"
THRESH = 0.82
MIN_CLUSTER = 6
SEED = 0
_BAND = re.compile(r" \[€[^\]]*\]| #high\d+| #low\d+| \[\?\]| #out\d+")


def load():
    import pickle
    E = np.load(CACHE / "emb_f16.npy", mmap_mode="r")
    ids = np.load(CACHE / "ids.npy"); row = {int(i): j for j, i in enumerate(ids)}
    m = {int(k): v for k, v in json.load(open(MAP_JSON)).items()}
    emb = {iid: E[row[iid]].astype(np.float32) for iid in m if iid in row}
    prices = {it["id"]: it.get("price")
              for it in pickle.load(open(CACHE / "items.pkl", "rb"))
              if it["id"] in emb}
    return emb, m, prices


def norm(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def run(emb, mapping, rule, prices=None):
    import math
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for iid, k in mapping.items():
        if iid in emb:
            by[k].append(iid)
    by = {k: v for k, v in by.items() if len(v) >= MIN_CLUSTER}
    cen, cnt, truemean, stream = {}, {}, {}, []
    med = {}
    for k, ids in by.items():
        ids = ids[:]; rng.shuffle(ids)
        h = len(ids) // 2
        seed_ids, stream_ids = ids[:h], ids[h:]
        if not seed_ids or not stream_ids:
            continue
        V = np.array([norm(emb[i]) for i in ids])
        truemean[k] = norm(V.mean(0))
        cen[k] = norm(np.array([norm(emb[i]) for i in seed_ids]).mean(0))
        cnt[k] = len(seed_ids)
        if prices is not None:
            pr = sorted(prices[i] for i in seed_ids
                        if prices.get(i) and prices[i] > 0)
            med[k] = pr[len(pr) // 2] if pr else None
        stream.extend((i, k) for i in stream_ids)
    rng.shuffle(stream)
    keys = list(cen.keys())
    kidx = {k: j for j, k in enumerate(keys)}
    fam_of = {k: _BAND.sub("", k) for k in keys}
    # band sibling index (same as production _band_base_index)
    band_idx = collections.defaultdict(list)
    for k in keys:
        band_idx[fam_of[k]].append(k)
    band_idx = {b: ks for b, ks in band_idx.items() if len(ks) >= 2}
    C = np.array([cen[k] for k in keys], dtype=np.float32)
    exact = fam_ok = total = 0
    for iid, truek in stream:
        x = norm(emb[iid])
        sims = C @ x
        j = int(sims.argmax())
        total += 1
        if float(sims[j]) < THRESH:
            continue                      # would create a new cluster → miss
        assigned = keys[j]
        # band-aware price re-route (production _band_route logic)
        if rule.endswith("+band") and prices is not None:
            p = prices.get(iid)
            sibs = band_idx.get(fam_of[assigned])
            if p and p > 0 and sibs:
                lp = math.log(p); bd = None
                for k2 in sibs:
                    m2 = med.get(k2)
                    if m2 and m2 > 0:
                        d = abs(lp - math.log(m2))
                        if bd is None or d < bd:
                            bd, assigned = d, k2
        j = kidx[assigned]
        if assigned == truek:
            exact += 1
        if fam_of[assigned] == fam_of[truek]:
            fam_ok += 1
        if rule.startswith("ema"):
            C[j] = norm((C[j] + x) / 2.0)
        elif rule.startswith("runmean"):
            n = cnt[assigned]
            C[j] = norm((C[j] * n + x) / (n + 1))
            cnt[assigned] = n + 1
    drift = float(np.mean([C[kidx[k]] @ truemean[k] for k in keys]))
    return exact / max(1, total), fam_ok / max(1, total), drift


def main():
    emb, mapping, prices = load()
    print(f"incremental-path test: {len(emb)} items, "
          f"{len(set(mapping.values()))} clusters  [{MAP_JSON.name}]", flush=True)
    for rule in ("static", "ema", "runmean", "runmean+band"):
        t0 = time.time()
        acc, fam, drift = run(emb, mapping, rule, prices)
        print(f"  {rule:13} acc_exact={100*acc:5.2f}%  acc_family={100*fam:5.2f}%  "
              f"centroid_drift_cos={drift:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
