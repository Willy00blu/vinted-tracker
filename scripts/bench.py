"""Cluster-quality benchmark for a FUTURE price-anomaly detector.

This does NOT run a detector. It measures the INTRINSIC quality of the clusters
that predicts how good they will be as price baselines for any future anomaly
detector. A cluster is a good baseline iff it is ONE product at ONE spec tier,
so its price distribution is a single population whose deviations are real
deals — not artefacts of product/spec mixing or of artificial price splitting.

Per cluster (size >= MIN, priced):
  bpure   : dominant-brand share (canonicalized, junk-brand filtered)
  specpure: dominant numeric model/spec-token share (catches iPhone 12 vs 13,
            i5-6600 vs i5-6500, 64GB vs 256GB mixed in one cluster) — a mixed
            cluster makes lower-spec items look like fake deals
  unimodal: 0 if a Jenks2 price split is BOTH strong (GVF/ratio) AND semantic
            (centroid cos < 0.97) → a hidden product mixture (e.g. console+game)
            that would generate false deals; else 1
  sep_w   : separation margin of this cluster's centroid vs the nearest other
            centroid in the SAME category|brand partition, clipped to [0,1];
            ~0 for artificial price-tier fragments and over-merge-confusable
            clusters → penalizes both over-split and over-merge
  q       = bpure * specpure * unimodal * sep_w      (the per-cluster quality)

Headline:
  ABQ      = Σ_clusters size·q / N_priced   ← fraction of the catalog living in
             clean, single-product, well-separated price baselines (0..1)
Reported axes (means over qualifying clusters) + coverage + SEP + legacy SCORE.
Detector-agnostic: a config that scores higher ABQ will give any sane future
detector cleaner baselines and fewer false positives.
"""
import sys, os, math, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl

MIN = 8
SEP_SCALE = 0.05           # margin at which sep_w saturates to 1
_JUNK = {"", "altro", "altri", "senza marca", "senzamarca", "nomark", "no brand",
         "none", "sans marque", "ohne marke", "sin marca", "sem marca", "geen merk"}


def _canon_brand(b):
    if not b:
        return ""
    bl = b.strip().lower()
    if bl in _JUNK:
        return ""
    return (cl._normalize_brand(b) or b).strip().lower()


def _part_key(it):
    cat = (it.get("category") or "").strip()
    brand, _ = cl._lex_sig(it.get("title", ""), it.get("brand"))
    brand = (cl._normalize_brand(brand) or brand or "").lower()
    if cat and brand:
        return f"{cat}|{brand}"
    if cat:
        return cat
    return brand or "__nobrand__"


def benchmark(items, mapping, emb, min_size=MIN):
    by = collections.defaultdict(list)
    priced_total = 0
    for it in items:
        if it.get("price") and it["price"] > 0 and it["id"] in emb:
            priced_total += 1
        by[mapping.get(it["id"], "unknown")].append(it)

    # centroids (normalized) + cluster→partition, for separation
    cen, cpart = {}, {}
    itpart = {it["id"]: _part_key(it) for it in items}
    for k, v in by.items():
        ids = [it["id"] for it in v if it["id"] in emb]
        if not ids:
            continue
        V = np.array([emb[i] for i in ids], dtype=np.float32)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        c = V.mean(0); c = c / max(float(np.linalg.norm(c)), 1e-12)
        cen[k] = c
        cpart[k] = collections.Counter(itpart[i] for i in ids).most_common(1)[0][0]
    part_cens = collections.defaultdict(list)
    for k, c in cen.items():
        part_cens[cpart[k]].append((k, c))

    def nearest_margin(k):
        lst = part_cens[cpart[k]]
        if len(lst) < 2:
            return 1.0                    # alone in its partition → no rival
        c = cen[k]
        best = max(float(c @ c2) for k2, c2 in lst if k2 != k)
        # cohesion≈own≈1; margin = own_self(1) - best_other ≈ 1-best
        return 1.0 - best

    tot_q = 0.0
    n_used = items_used = 0
    sum_b = sum_s = sum_u = sum_sep = sum_coh = 0.0
    mix_items = 0
    for k, v in by.items():
        if len(v) < min_size or k not in cen:
            continue
        ids = [it["id"] for it in v if it["id"] in emb]
        # brand purity
        brs = [b for b in (_canon_brand(it.get("brand")) for it in v) if b]
        bpure = (collections.Counter(brs).most_common(1)[0][1] / len(brs)) if brs else 1.0
        # spec purity (numeric model/spec tokens)
        nums_per = []
        for it in v:
            _, nums = cl._lex_sig(it.get("title", ""), it.get("brand"))
            nums_per.append(nums)
        flat = [n for nums in nums_per for n in nums]
        if flat:
            dom = collections.Counter(flat).most_common(1)[0][0]
            have = sum(1 for nums in nums_per if dom in nums)
            specpure = have / len(v)
        else:
            specpure = 1.0
        # unimodality: hidden semantic price-mixture?
        unimodal = 1.0
        pr = [(it["price"], it["id"]) for it in v if it.get("price") and it["price"] > 0
              and it["id"] in emb]
        if len(pr) >= 2 * 3:
            stats = cl._jenks2_stats([p for p, _ in pr], min_band=3)
            if stats:
                idx, gvf, ratio = stats
                if gvf >= 0.60 and ratio >= 2.0:
                    sp = sorted(pr)
                    A = [i for _, i in sp[:idx]]; B = [i for _, i in sp[idx:]]
                    ca = np.mean([emb[i] for i in A], axis=0)
                    cb = np.mean([emb[i] for i in B], axis=0)
                    ca = ca / max(np.linalg.norm(ca), 1e-12)
                    cb = cb / max(np.linalg.norm(cb), 1e-12)
                    if float(ca @ cb) < 0.97:
                        unimodal = 0.0
                        mix_items += len(v)
        margin = nearest_margin(k)
        sep_w = min(1.0, max(0.0, margin / SEP_SCALE))
        coh = float(np.mean([cen[k] @ (emb[i] / max(np.linalg.norm(emb[i]), 1e-12))
                             for i in ids]))
        q = bpure * specpure * unimodal * sep_w
        tot_q += q * len(v)
        items_used += len(v)
        n_used += 1
        sum_b += bpure; sum_s += specpure; sum_u += unimodal
        sum_sep += margin; sum_coh += coh
    abq = tot_q / max(1, priced_total)
    d = lambda x: x / max(1, n_used)
    return {
        "ABQ": abq,
        "coverage": 100 * items_used / max(1, priced_total),
        "bpure": d(sum_b), "specpure": d(sum_s), "unimodal": d(sum_u),
        "sep": d(sum_sep), "coh": d(sum_coh),
        "mix_pct": 100 * mix_items / max(1, items_used),
        "n_clusters": n_used, "n_items": items_used,
    }


def report(items, mapping, emb, label, with_legacy=True):
    b = benchmark(items, mapping, emb)
    print(f"=== {label} ===")
    print(f"  ABQ={b['ABQ']:.3f}  cov={b['coverage']:.1f}%  bpure={b['bpure']:.3f} "
          f"specpure={b['specpure']:.3f} unimodal={b['unimodal']:.3f} sep={b['sep']:.3f} "
          f"coh={b['coh']:.3f}  mix={b['mix_pct']:.1f}%  "
          f"(clusters={b['n_clusters']})")
    if with_legacy:
        m = cl.eval_metrics(items, mapping, emb)
        print(f"  [legacy] SCORE={m['score']:.1f} ANOM_READY={m['anom_ready']:.1f}% "
              f"N_IDEAL={m['n_ideal']}")
    return b


if __name__ == "__main__":
    import json, pickle
    cache = ROOT / "data" / "ab_cache"
    if (cache / "emb_f16.npy").exists():
        E = np.load(cache / "emb_f16.npy", mmap_mode="r")
        ids = np.load(cache / "ids.npy"); row = {int(i): j for j, i in enumerate(ids)}
        items = pickle.load(open(cache / "items.pkl", "rb"))
        emb = {it["id"]: E[row[it["id"]]].astype(np.float32) for it in items
               if it["id"] in row}
    else:
        d = np.load("/tmp/ab_emb.npz", allow_pickle=True)
        emb = {int(i): v for i, v in zip(d["ids"], d["emb"])}
        items = pickle.load(open("/tmp/ab_items.pkl", "rb"))
    items = [it for it in items if it["id"] in emb]
    mpath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab_final_base.json"
    mapping = {int(k): v for k, v in json.load(open(mpath)).items()}
    report(items, mapping, emb, f"BENCHMARK [{Path(mpath).name}]")
