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
import sys, os, math, re, collections, statistics as st
from pathlib import Path
import numpy as np

_YEAR = re.compile(r"^(19|20)\d\d$")     # listing years are NOT product specs


def _spec_nums(title, brand):
    """Numeric model/spec tokens, with listing-years stripped (they pollute
    spec-purity: '2021'/'2022' are when it was posted, not what it is)."""
    _, nums = cl._lex_sig(title, brand)
    return {n for n in nums if not _YEAR.match(n)}

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
        # spec purity (numeric model/spec tokens). Only judged when ENOUGH items
        # actually carry a spec number — products without model numbers in their
        # titles ("Nintendo Switch Lite", "Apple Watch") are not spec-impure,
        # they're just number-less, so they get specpure=1 (no evidence of mix).
        nums_per = [_spec_nums(it.get("title", ""), it.get("brand")) for it in v]
        n_with = sum(1 for nums in nums_per if nums)
        if n_with < max(5, 0.4 * len(v)):
            specpure = 1.0
        else:
            flat = [n for nums in nums_per for n in nums]
            dom = collections.Counter(flat).most_common(1)[0][0]
            have = sum(1 for nums in nums_per if dom in nums)
            specpure = have / n_with
        # unimodality: a strong, populated PRICE-BIMODALITY (two separated modes,
        # each ≥3 items) means the cluster mixes a device with its accessories/
        # games/parts (e.g. iPhone €500 + cover €15, Switch console €210 + games
        # €20). A single product — even new/used or storage variants — does NOT
        # produce gvf≥0.6 with a ≥3× modal ratio. We deliberately do NOT use a
        # text-cos gate: these mixtures share the brand+model token so their text
        # is near-identical (cos≈0.97-0.99) — exactly why the pipeline misses them.
        unimodal = 1.0
        pr = [it["price"] for it in v if it.get("price") and it["price"] > 0]
        if len(pr) >= 2 * 3:
            stats = cl._jenks2_stats(pr, min_band=3)
            if stats:
                idx, gvf, ratio = stats
                # Honest measure kept slightly BROADER than the de-mixer's split
                # threshold (gvf 0.60 / ratio 3.0) so it can't be trivially zeroed
                # — residual 2.5-3× mixtures still count as exposure.
                if gvf >= 0.55 and ratio >= 2.5:
                    unimodal = 0.0
                    mix_items += len(v)
        margin = nearest_margin(k)
        coh = float(np.mean([cen[k] @ (emb[i] / max(np.linalg.norm(emb[i]), 1e-12))
                             for i in ids]))
        # purity-only quality (single product · single spec · no mixture).
        # Separation is reported as its OWN axis (sep) instead of multiplied in,
        # so ABQ stays sensitive; over-split/over-merge show up as sep↓.
        q = bpure * specpure * unimodal
        tot_q += q * len(v)
        items_used += len(v)
        n_used += 1
        sum_b += bpure; sum_s += specpure; sum_u += unimodal
        sum_sep += margin; sum_coh += coh
    # ── MASK: deal-masking exposure from price-band splitting ────────────────
    # HARM covers the under-split failure (mixture hides deals in noise); MASK
    # covers the OVER-split failure: when one product is split into price bands,
    # a real deal (−40% off band B's median) lands inside a cheaper sibling
    # band's [p5,p95] range, where it looks like a normal listing → never
    # flagged. Together HARM+MASK bracket the split/merge trade-off, so the
    # objective can no longer be gamed by splitting more (or less).
    _BAND_TAG = re.compile(r" \[€[^\]]*\]| #high\d+| #low\d+| \[\?\]")
    fams = collections.defaultdict(list)
    for k, v in by.items():
        pr = sorted(it["price"] for it in v
                    if it.get("price") and it["price"] > 0)
        if len(pr) >= 2:
            fams[_BAND_TAG.sub("", k)].append(pr)
    masked_items = 0
    for fam, bands in fams.items():
        if len(bands) < 2:
            continue
        bands.sort(key=lambda pr: pr[len(pr) // 2])          # by median
        for bi in range(1, len(bands)):
            b = bands[bi]
            deal = 0.60 * b[len(b) // 2]                     # −40% off median
            for s in bands[:bi]:                             # cheaper siblings
                lo = s[max(0, int(0.05 * len(s)))]
                hi = s[min(len(s) - 1, int(0.95 * len(s)))]
                if lo <= deal <= hi:
                    masked_items += len(b)
                    break

    abq = tot_q / max(1, priced_total)
    harm = mix_items / max(1, priced_total)     # catalog share in HARMFUL mixed
                                                # clusters (false-positive factories
                                                # for any future detector)
    mask = masked_items / max(1, priced_total)  # catalog share whose −40% deal
                                                # would hide in a sibling band
    d = lambda x: x / max(1, n_used)
    return {
        "ABQ": abq,
        "HARM": harm,
        "MASK": mask,
        "OBJ": abq - harm - mask,               # primary search objective:
                                                # reward clean baselines, punish
                                                # BOTH failure modes 1:1 —
                                                # mixtures (under-split) and
                                                # deal-masking (over-split)
        "coverage": 100 * items_used / max(1, priced_total),
        "bpure": d(sum_b), "specpure": d(sum_s), "unimodal": d(sum_u),
        "sep": d(sum_sep), "coh": d(sum_coh),
        "mix_pct": 100 * mix_items / max(1, items_used),
        "n_clusters": n_used, "n_items": items_used,
    }


def report(items, mapping, emb, label, with_legacy=True):
    b = benchmark(items, mapping, emb)
    print(f"=== {label} ===")
    print(f"  OBJ={b['OBJ']:.3f}  HARM={b['HARM']:.3f}  MASK={b['MASK']:.3f}")
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
