"""#1 (running-mean centroid) + #7 (vectorized match) — test the ALWAYS-ON
incremental path, which SCORE/ANOM_READY never see (they're computed on the
regen mapping). We simulate the monitor running BETWEEN two regenerations:

  - seed each cluster's centroid from half its items (= state just after regen),
  - stream the other half one-by-one in random global order,
  - assign each to the nearest centroid (cos argmax, threshold 0.82),
  - update the assigned centroid with the current rule.

We compare the EMA(α=0.5) update `(old+emb)/2` (current) vs the true running
mean `(old*n+emb)/(n+1)` (#1) on:
  - assignment ACCURACY (streamed item routed to its true cluster),
  - centroid DRIFT (cos(final_centroid, full-cluster mean)).
A static-centroid (no update) run is the no-drift reference.
"""
import sys, os, json, time, collections, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")

EMB_NPZ = "/tmp/ab_emb.npz"
MAP_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab_final_base.json"
THRESH = 0.82
MIN_CLUSTER = 6
SEED = 0


def load():
    d = np.load(EMB_NPZ, allow_pickle=True)
    emb = {int(i): v.astype(np.float32) for i, v in zip(d["ids"], d["emb"])}
    m = json.load(open(MAP_JSON))
    m = {int(k): v for k, v in m.items()}
    return emb, m


def norm(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def run(emb, mapping, rule):
    """rule in {'static','ema','runmean'}. Returns (accuracy, mean_drift)."""
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for iid, k in mapping.items():
        if iid in emb:
            by[k].append(iid)
    by = {k: v for k, v in by.items() if len(v) >= MIN_CLUSTER}
    # split seed / stream
    cen = {}; cnt = {}; truemean = {}; stream = []
    for k, ids in by.items():
        ids = ids[:]; rng.shuffle(ids)
        h = len(ids) // 2
        seed_ids, stream_ids = ids[:h], ids[h:]
        if not seed_ids or not stream_ids:
            continue
        V = np.array([norm(emb[i]) for i in ids])
        truemean[k] = norm(V.mean(0))
        Vs = np.array([norm(emb[i]) for i in seed_ids])
        cen[k] = norm(Vs.mean(0)); cnt[k] = len(seed_ids)
        for i in stream_ids:
            stream.append((i, k))
    rng.shuffle(stream)
    keys = list(cen.keys())
    kidx = {k: j for j, k in enumerate(keys)}
    C = np.array([cen[k] for k in keys], dtype=np.float32)
    correct = 0; total = 0
    for iid, truek in stream:
        x = norm(emb[iid])
        sims = C @ x
        j = int(sims.argmax())
        if float(sims[j]) < THRESH:
            # below threshold → would create a new cluster; counts as a miss
            total += 1
            continue
        assigned = keys[j]
        total += 1
        if assigned == truek:
            correct += 1
        if rule == "ema":
            C[j] = norm((C[j] + x) / 2.0)
            cnt[assigned] += 1
        elif rule == "runmean":
            n = cnt[assigned]
            C[j] = norm((C[j] * n + x) / (n + 1))
            cnt[assigned] = n + 1
        # static: no update
    # drift vs true full-cluster mean
    drifts = [float(C[kidx[k]] @ truemean[k]) for k in keys]
    return correct / max(1, total), float(np.mean(drifts))


def main():
    emb, mapping = load()
    print(f"loaded {len(emb)} emb, {len(set(mapping.values()))} clusters in mapping")
    for rule in ("static", "ema", "runmean"):
        t0 = time.time()
        acc, drift = run(emb, mapping, rule)
        print(f"  {rule:8} assign_acc={100*acc:5.2f}%  centroid_drift_cos={drift:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
