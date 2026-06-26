"""Benchmark pipeline: ablation study to pick the best clustering variant.

Builds a stratified 25-30k sample (top brands + top categories + ALL items of
the historical mappazzoni so we can see if they get fixed), runs 7 pipeline
variants on the SAME sample with SHARED embedding cache, and reports the
canonical SCORE + diagnostic axes side by side. The winning config is
declared at the end.

Variants:
    A. baseline                  — current production pipeline
    B. +brand-strip              — Proposal #1
    C. +img-refine OR gate       — fix: image_refine on (CV>0.5 OR Jenks2 near-miss)
    D. B+C                       — combo
    E. D + text-refine recursive — Proposal #2
    F. D + Jenks2 aggressive     — high_conf_gvf 0.75→0.65
    G. D + min_cluster_size=3    — slightly stricter HDBSCAN
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import connect, shared_db_path                  # noqa: E402
from core.clustering import (                                          # noqa: E402
    _embed_text, _get_model, _get_hdbscan, _blob_to_emb, _lex_sig,
    _jenks2_stats, _price_split_jenks2,
    RESCUE_GATE_HI, RESCUE_GATE_LO,
)

# ─── Sample building ─────────────────────────────────────────────────────────

MAPPAZZONI = [
    "nintendo/nintendo switch", "corsair/ram corsair vengeance ddr4",
    "de 100 chargeurs câble", "kingston/2x ddr4 8gb ram",
    "sony/console ps4 slim jeux", "intel/i5 6500",
    "apple/iphone 11", "apple/iphone 12", "apple/iphone 13",
    "rollei/mini appareil photo rollei",
]

def build_sample(target=30000, force_mappazzoni=False):
    """`force_mappazzoni=True` adds ALL items of historical mappazzoni clusters
    (biased toward broken clusters → singleton rate high). `False` = pure random
    (representative of the corpus, SCORE comparable to full-regen eval)."""
    sample_cache = CACHE_DIR / f"sample_{target}_{'mp' if force_mappazzoni else 'rnd'}.pkl"
    if sample_cache.exists():
        import pickle
        with open(sample_cache, "rb") as f:
            rows = pickle.load(f)
        img_of = {}
        for r in rows:
            if r["image_emb"] is not None:
                try:
                    img_of[r["id"]] = _blob_to_emb(r["image_emb"])
                except Exception:
                    pass
        return rows, img_of
    with connect(shared_db_path()) as c:
        if force_mappazzoni:
            keys = ",".join(["?"] * len(MAPPAZZONI))
            rows_mp = [dict(r) for r in c.execute(
                f"SELECT id,title,brand,category,catalog_id,price,image_emb "
                f"FROM items WHERE removed=0 AND cluster_key IN ({keys}) AND title IS NOT NULL",
                MAPPAZZONI
            )]
            already = {r["id"] for r in rows_mp}
            remaining = target - len(rows_mp)
            rows_fill = [dict(r) for r in c.execute(
                "SELECT id,title,brand,category,catalog_id,price,image_emb "
                "FROM items WHERE removed=0 AND title IS NOT NULL "
                "ORDER BY RANDOM() LIMIT ?",
                (remaining * 2,)
            )]
            rows_fill = [r for r in rows_fill if r["id"] not in already][:remaining]
            rows = rows_mp + rows_fill
        else:
            rows = [dict(r) for r in c.execute(
                "SELECT id,title,brand,category,catalog_id,price,image_emb "
                "FROM items WHERE removed=0 AND title IS NOT NULL "
                "ORDER BY RANDOM() LIMIT ?", (target,)
            )]
    import pickle
    with open(sample_cache, "wb") as f:
        pickle.dump(rows, f)
    img_of = {}
    for r in rows:
        if r["image_emb"] is not None:
            try:
                img_of[r["id"]] = _blob_to_emb(r["image_emb"])
            except Exception:
                pass
    return rows, img_of


# ─── Pipeline components ─────────────────────────────────────────────────────

CACHE_DIR = Path("/tmp/vinted_bench")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(tag, ids):
    """Cache key = tag + hash(sorted ids). Different sample → different file."""
    import hashlib
    h = hashlib.md5(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()[:10]
    return CACHE_DIR / f"emb_{tag}_{h}.npz"


def embed_corpus(items, *, exclude_brand_by_partition: dict[str, str] | None = None,
                 tag: str = "std"):
    """Embed with disk cache. Re-uses identical sample+config across runs."""
    ids = [it["id"] for it in items]
    cache = _cache_path(tag, ids)
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {int(i): v for i, v in zip(d["ids"], d["embs"])}
    model = _get_model()
    if exclude_brand_by_partition is None:
        texts = [_embed_text(it) for it in items]
    else:
        texts = [_embed_text(it, exclude_brand=exclude_brand_by_partition.get(_part_key(it)))
                 for it in items]
    embs = model.encode(texts, show_progress_bar=False,
                        convert_to_numpy=True, batch_size=256)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    np.savez(cache, ids=np.array(ids, dtype=np.int64), embs=embs.astype(np.float32))
    return {it["id"]: e for it, e in zip(items, embs)}


def _part_key(it):
    cat = (it.get("category") or "").strip()
    brand, _ = _lex_sig(it.get("title", ""), it.get("brand"))
    if cat and brand:
        return f"{cat}|{brand}"
    elif cat:
        return cat
    else:
        return brand or "__nobrand__"


def partition_and_cluster(items, emb_of, *, min_cluster_size=2):
    """HDBSCAN within each partition."""
    by_part = defaultdict(list)
    for it in items:
        by_part[_part_key(it)].append(it)
    H = _get_hdbscan().HDBSCAN
    mapping = {}
    used = set()
    for part, p_items in by_part.items():
        ns = "" if part == "__nobrand__" else f"{part}/"
        if len(p_items) <= min_cluster_size:
            for it in p_items:
                mapping[it["id"]] = ns + (it.get("title", "?")[:30])
            continue
        E = np.array([emb_of[it["id"]] for it in p_items])
        labels = H(min_cluster_size=min_cluster_size, metric="euclidean",
                   cluster_selection_method="eom",
                   core_dist_n_jobs=-1).fit_predict(E)
        by_label = defaultdict(list)
        for it, lab in zip(p_items, labels):
            by_label[int(lab)].append(it)
        for lab, group in by_label.items():
            if lab == -1:
                for it in group:
                    mapping[it["id"]] = ns + (it.get("title", "?")[:30] + f"_n{it['id'] % 1000}")
            else:
                # readable key = title of medoid
                vecs = np.array([emb_of[it["id"]] for it in group])
                centroid = vecs.mean(0); centroid /= max(np.linalg.norm(centroid), 1e-12)
                dists = np.linalg.norm(vecs - centroid, axis=1)
                medoid = group[int(np.argmin(dists))]
                key = ns + (medoid.get("title", "?")[:50].lower())
                # de-dup
                k, c = key, 0
                while k in used:
                    c += 1; k = f"{key}~{c}"
                used.add(k)
                for it in group:
                    mapping[it["id"]] = k
    return mapping


def rescue_singletons(items, mapping, emb_of, *, gate_hi=RESCUE_GATE_HI, gate_lo=RESCUE_GATE_LO):
    groups = defaultdict(list)
    for it in items:
        groups[mapping[it["id"]]].append(it)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    singletons = [v[0] for v in groups.values() if len(v) == 1]
    if not multi or not singletons:
        return mapping
    keys = list(multi.keys())
    cmat = np.array([np.mean([emb_of[it["id"]] for it in multi[k]], axis=0) for k in keys])
    cmat = cmat / np.maximum(np.linalg.norm(cmat, axis=1, keepdims=True), 1e-12)
    cbrand = {k: Counter(it.get("brand", "") for it in multi[k] if it.get("brand")).most_common(1) for k in keys}
    for it in singletons:
        e = emb_of[it["id"]]
        sims = cmat @ e
        j = int(sims.argmax())
        s = float(sims[j])
        if s >= gate_hi:
            mapping[it["id"]] = keys[j]
        elif s >= gate_lo:
            if it.get("brand") and cbrand[keys[j]] and it["brand"] == cbrand[keys[j]][0][0]:
                mapping[it["id"]] = keys[j]
    return mapping


def jenks2_split(items, mapping, emb_of, *, high_conf_gvf=0.75, high_conf_ratio=4.0):
    groups = defaultdict(list)
    for it in items:
        groups[mapping[it["id"]]].append(it)
    new_groups = _price_split_jenks2(
        groups, emb_of, lambda *a: None,
        high_conf_gvf=high_conf_gvf, high_conf_ratio=high_conf_ratio,
    )
    out = {}
    for k, v in new_groups.items():
        for it in v:
            out[it["id"]] = k
    return out


def image_refine_v2(items, mapping, img_of, *, gate_mode="near_miss"):
    """gate_mode: 'near_miss' = current (Jenks2 GVF>=0.4, ratio>=1.5)
                  'cv'        = old (CV>0.5)
                  'or'        = NEW: CV>0.5 OR near-miss
                  'and'       = strict (both)"""
    if not img_of:
        return mapping
    H = _get_hdbscan().HDBSCAN
    norm_img = {i: v / max(float(np.linalg.norm(v)), 1e-12) for i, v in img_of.items()}
    groups = defaultdict(list)
    for it in items:
        groups[mapping[it["id"]]].append(it)
    out = dict(mapping)
    n_split = 0
    for k, v in groups.items():
        if len(v) < 6:
            continue
        pr = [it["price"] for it in v if it.get("price")]
        if len(pr) < 6:
            continue
        m_pr = st.mean(pr); cv = (st.pstdev(pr) / m_pr) if m_pr else 0.0
        stats = _jenks2_stats(pr, min_band=3)
        nm = stats is not None and stats[1] >= 0.40 and stats[2] >= 1.5
        if gate_mode == "cv":
            allow = cv > 0.5
        elif gate_mode == "near_miss":
            allow = nm
        elif gate_mode == "or":
            allow = cv > 0.5 or nm
        else:   # 'and'
            allow = cv > 0.5 and nm
        if not allow:
            continue
        members = [it for it in v if it["id"] in norm_img]
        if len(members) < 6:
            continue
        X = np.array([norm_img[it["id"]] for it in members])
        labels = H(min_cluster_size=2, metric="euclidean",
                   cluster_selection_method="leaf",
                   core_dist_n_jobs=-1).fit_predict(X)
        uniq = sorted({int(l) for l in labels if l != -1})
        if len(uniq) < 2:
            continue
        # img-cos guard
        centroids = {}
        for l in uniq:
            sub = np.array([norm_img[it["id"]] for it, lab in zip(members, labels) if int(lab) == l])
            c = sub.mean(0); c = c / max(float(np.linalg.norm(c)), 1e-12)
            centroids[l] = c
        max_cos = 0.0
        ls = list(uniq)
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                cos = float(centroids[ls[i]] @ centroids[ls[j]])
                if cos > max_cos:
                    max_cos = cos
        if max_cos >= 0.92:
            continue
        n_split += 1
        for it, l in zip(members, labels):
            if l != -1:
                out[it["id"]] = f"{k} #img{int(l)}"
    return out


def text_refine_recursive(items, mapping, emb_of, *, min_size=20, max_cohesion=0.85):
    """Proposal #2: HDBSCAN secondary pass inside clusters with low cohesion."""
    H = _get_hdbscan().HDBSCAN
    groups = defaultdict(list)
    for it in items:
        groups[mapping[it["id"]]].append(it)
    out = dict(mapping)
    n_split = 0
    for k, v in groups.items():
        if len(v) < min_size:
            continue
        vecs = np.array([emb_of[it["id"]] for it in v])
        centroid = vecs.mean(0); centroid /= max(np.linalg.norm(centroid), 1e-12)
        cohesion = float((vecs @ centroid).mean())
        if cohesion >= max_cohesion:
            continue
        labels = H(min_cluster_size=3, metric="euclidean",
                   cluster_selection_method="leaf",
                   core_dist_n_jobs=-1).fit_predict(vecs)
        uniq = sorted({int(l) for l in labels if l != -1})
        if len(uniq) < 2:
            continue
        n_split += 1
        for it, l in zip(v, labels):
            if l != -1:
                out[it["id"]] = f"{k} #txt{int(l)}"
    return out


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(items, mapping, emb_of):
    n = len(items)
    by = defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    cov = sum(len(v) for v in multi.values())
    cn = cd = 0; pok = 0; bp_num = bp_den = 0
    mastodonti = 0
    for k, v in multi.items():
        vecs = np.array([emb_of[it["id"]] for it in v])
        cen = vecs.mean(0); cen /= max(np.linalg.norm(cen), 1e-12)
        cn += float((vecs @ cen).sum()); cd += len(v)
        pr = [it["price"] for it in v if it.get("price")]
        if len(pr) > 1 and st.mean(pr) and st.pstdev(pr) / st.mean(pr) <= 0.9:
            pok += len(v)
        brs = [it.get("brand") for it in v if it.get("brand")]
        if brs:
            bp_num += Counter(brs).most_common(1)[0][1]; bp_den += len(brs)
        if len(v) >= 100:
            ps = sorted(pr); pcv = (st.pstdev(ps) / st.mean(ps)) if st.mean(ps) else 0
            if pcv > 0.8:
                mastodonti += 1
    coh = cn / max(1, cd); price = 100 * pok / n; bp = 100 * bp_num / max(1, bp_den)
    sizes = [len(v) for v in by.values()]
    sing = sum(1 for s in sizes if s == 1)
    return {
        "items": n, "cluster": len(sizes), "multi": len(multi),
        "singleton_pct": 100*sing/len(sizes),
        "coverage": 100*cov/n,
        "cohesion": coh,
        "price_coh": price,
        "brand_pure": bp,
        "score": coh * price,
        "mappazzoni_residui": mastodonti,
    }


# ─── Run ─────────────────────────────────────────────────────────────────────

def run_variant(name, items, img_of, emb_std, *, brand_strip=False,
                image_gate="near_miss", text_refine=False,
                jenks2_high_conf_gvf=0.75, min_cluster_size=2):
    print(f"\n─── {name} ───", flush=True)
    t0 = time.time()
    if brand_strip:
        # Identify partition brands (only re-embed items in big partitions size>=20)
        by_part = defaultdict(list)
        for it in items:
            by_part[_part_key(it)].append(it)
        excl = {}
        for part, p_items in by_part.items():
            if len(p_items) >= 20:
                _, brand_token = _lex_sig(p_items[0].get("title", ""), p_items[0].get("brand"))
                if "|" in part:
                    bk = part.split("|")[-1]
                else:
                    bk = part
                if bk and bk != "__nobrand__":
                    excl[part] = bk
        # Override emb only for items in those partitions
        emb_brandstrip = embed_corpus(
            [it for it in items if _part_key(it) in excl],
            exclude_brand_by_partition=excl,
            tag="brandstrip",
        )
        emb_use = {it["id"]: emb_brandstrip.get(it["id"], emb_std[it["id"]]) for it in items}
    else:
        emb_use = emb_std

    mapping = partition_and_cluster(items, emb_use, min_cluster_size=min_cluster_size)
    mapping = rescue_singletons(items, mapping, emb_use)
    mapping = jenks2_split(items, mapping, emb_use, high_conf_gvf=jenks2_high_conf_gvf)
    mapping = image_refine_v2(items, mapping, img_of, gate_mode=image_gate)
    if text_refine:
        mapping = text_refine_recursive(items, mapping, emb_use)
    m = metrics(items, mapping, emb_use)
    m["elapsed_s"] = int(time.time() - t0)
    print(f"  cluster {m['cluster']}  singleton {m['singleton_pct']:.0f}%  "
          f"coverage {m['coverage']:.0f}%  cohesion {m['cohesion']:.3f}  "
          f"price-coh {m['price_coh']:.0f}%  brand-pure {m['brand_pure']:.0f}%  "
          f"mastodonti_residui {m['mappazzoni_residui']}  "
          f"SCORE {m['score']:.1f}  [{m['elapsed_s']}s]")
    return m


def main():
    print("Loading sample (30k random, no mappazzoni forced)…", flush=True)
    items, img_of = build_sample(target=30000, force_mappazzoni=False)
    print(f"  {len(items)} items, {len(img_of)} con image_emb")

    print("\nEmbedding standard (e5-large, no brand-strip)…", flush=True)
    emb_std = embed_corpus(items, exclude_brand_by_partition=None)

    results = {}
    results["A. baseline"]         = run_variant(
        "A. baseline (current production)", items, img_of, emb_std)
    results["B. +brand-strip"]     = run_variant(
        "B. +brand-strip (Proposal #1)", items, img_of, emb_std, brand_strip=True)
    results["C. +img OR gate"]     = run_variant(
        "C. baseline + img-refine OR gate", items, img_of, emb_std, image_gate="or")
    results["D. B+C"]              = run_variant(
        "D. brand-strip + img OR gate", items, img_of, emb_std,
        brand_strip=True, image_gate="or")
    results["E. D+text-refine"]    = run_variant(
        "E. D + text-refine recursive (Proposal #2)", items, img_of, emb_std,
        brand_strip=True, image_gate="or", text_refine=True)
    results["F. D+Jenks2 agg"]     = run_variant(
        "F. D + Jenks2 aggressive (high_conf_gvf 0.65)", items, img_of, emb_std,
        brand_strip=True, image_gate="or", jenks2_high_conf_gvf=0.65)
    results["G. D+min_clust=3"]    = run_variant(
        "G. D + min_cluster_size=3", items, img_of, emb_std,
        brand_strip=True, image_gate="or", min_cluster_size=3)
    results["H. baseline+Jenks2 agg"] = run_variant(
        "H. baseline + Jenks2 aggressive ONLY (no brand-strip, no img OR)",
        items, img_of, emb_std,
        brand_strip=False, image_gate="near_miss",
        jenks2_high_conf_gvf=0.65)
    results["I. baseline+brand-strip ONLY"] = run_variant(
        "I. baseline + brand-strip ONLY (no Jenks2 agg, no img OR)",
        items, img_of, emb_std,
        brand_strip=True, image_gate="near_miss")

    print("\n" + "="*78)
    print("RIEPILOGO  (SCORE = cohesion × price_coh, MAX vince)")
    print("="*78)
    header = f"{'variant':28} {'SCORE':>6}  {'cohes':>5} {'price':>5} {'brand':>5} {'cov':>4} {'sing%':>5} {'mast':>4} {'cl':>5}"
    print(header)
    print("─" * len(header))
    sorted_res = sorted(results.items(), key=lambda kv: -kv[1]["score"])
    for name, m in sorted_res:
        print(f"{name:28} {m['score']:6.1f}  {m['cohesion']:5.3f} {m['price_coh']:5.0f} "
              f"{m['brand_pure']:5.0f} {m['coverage']:4.0f} {m['singleton_pct']:5.0f} "
              f"{m['mappazzoni_residui']:4} {m['cluster']:5}")
    print()
    print(f"VINCITORE: {sorted_res[0][0]}  (SCORE {sorted_res[0][1]['score']:.1f})")


if __name__ == "__main__":
    main()
