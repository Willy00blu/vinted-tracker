"""Multi-photo test: average the CLIP embeddings of ALL photos of each Vinted
item (mean-pooling, like FashionID) vs using only the first photo. Does it give
tighter console/game separation? Scrapes PS4/Switch (rate-limited), embeds with
B-32. Cache /tmp/multiphoto.npz.

  python3 scripts/multiphoto_test.py          # build + compare
  python3 scripts/multiphoto_test.py sweep
"""
import sys, io, urllib.request, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from scripts.cluster_ab2 import base_partition
from scripts.fashion_clip_test import norm, report

CACHE = Path("/tmp/multiphoto.npz")
QUERIES = ["ps4", "playstation 4", "ps4 jeu", "nintendo switch",
           "nintendo switch jeu", "nintendo switch console"]
CAP = 500
MAX_PHOTOS = 4


def build():
    from core.vinted_client import safe_search, try_domains
    from core.discovery import DOMAINS_DEFAULT
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    items = {}
    for q in QUERIES:
        for p in (1, 2):
            params = {"search_text": q, "order": "newest_first", "per_page": 96, "page": p}
            res, used, errs = try_domains(
                DOMAINS_DEFAULT, lambda d, pr=params: safe_search(d, pr), key="mphoto")
            if not res:
                print(f"  {q} p{p}: {errs}"); continue
            for it in res:
                urls = [ph.url for ph in (it.photos or []) if ph.url][:MAX_PHOTOS]
                if urls and it.title and (it.price or 0) > 0:
                    items[it.id] = {"id": it.id, "title": it.title, "brand": it.brand_title or "",
                                    "price": float(it.price), "urls": urls}
    items = list(items.values())[:CAP]
    nph = sum(len(it["urls"]) for it in items)
    print(f"scrapati: {len(items)}  foto totali: {nph} (media {nph/max(1,len(items)):.1f}/item) — scarico…", flush=True)

    def dl(u):
        try:
            r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            return Image.open(io.BytesIO(urllib.request.urlopen(r, timeout=10).read())).convert("RGB")
        except Exception:
            return None
    # flat list of (item_idx, image)
    flat_imgs, flat_idx = [], []
    keep = []
    for it in items:
        ims = [dl(u) for u in it["urls"]]
        ims = [im for im in ims if im is not None]
        if not ims:
            continue
        idx = len(keep); keep.append(it)
        for im in ims:
            flat_imgs.append(im); flat_idx.append(idx)
        if len(keep) % 100 == 0:
            print(f"   {len(keep)} item", flush=True)
    print(f"item con foto: {len(keep)}  immagini scaricate: {len(flat_imgs)} — embeddo B-32 + L-14…", flush=True)
    tv = cl._get_model().encode([cl._embed_text({"title": it["title"]}) for it in keep],
                                show_progress_bar=False, convert_to_numpy=True)

    def per_item(ev):
        dim = ev.shape[1]
        single = np.zeros((len(keep), dim)); multi = np.zeros((len(keep), dim))
        seen = [False]*len(keep); acc = collections.defaultdict(list)
        for e, idx in zip(ev, flat_idx):
            acc[idx].append(e)
            if not seen[idx]:
                single[idx] = e; seen[idx] = True
        for idx, es in acc.items():
            multi[idx] = np.mean(es, axis=0)
        return single, multi

    ev_b32 = SentenceTransformer("clip-ViT-B-32").encode(
        flat_imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=64)
    b32_single, b32_multi = per_item(ev_b32)
    ev_l14 = SentenceTransformer("clip-ViT-L-14").encode(
        flat_imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
    l14_single, l14_multi = per_item(ev_l14)
    np.savez(CACHE, ids=np.array([it["id"] for it in keep]),
             titles=np.array([it["title"] for it in keep], dtype=object),
             brands=np.array([it["brand"] for it in keep], dtype=object),
             prices=np.array([it["price"] for it in keep]),
             tv=tv, b32_single=b32_single, b32_multi=b32_multi,
             l14_single=l14_single, l14_multi=l14_multi)
    print("cache salvata (B-32 + L-14, single + multi)")


def sweep():
    from core.category_db import connect
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv = norm(d["tv"])
    b32_single = norm(d["b32_single"]); b32_multi = norm(d["b32_multi"])
    l14_single = norm(d["l14_single"]); l14_multi = norm(d["l14_multi"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    print(f"campione: {len(items)}\n")

    def run(vecs, name, eyeball=False):
        cache = {it["id"]: vecs[emap[it["id"]]] for it in items}
        with connect(Path("/tmp/mph.db")) as conn:
            base = base_partition(items, cache, "catbrand")
            m = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=cache)
        report(m, items, name)
        if eyeball:
            H = cl._get_hdbscan().HDBSCAN
            lab = H(min_cluster_size=4, metric="euclidean", cluster_selection_method="leaf",
                    core_dist_n_jobs=-1).fit_predict(vecs)
            g = collections.defaultdict(list)
            for it, l in zip(items, lab): g[int(l)].append(it)
            print(f"   --- gruppi-immagine {name} (primi 8) ---")
            for l, v in sorted(((l, v) for l, v in g.items() if l != -1), key=lambda kv: -len(kv[1]))[:8]:
                pr = [it["price"] for it in v]
                print(f"     n={len(v):3} med€{st.median(pr):>4.0f}: " +
                      " | ".join(it["title"][:20] for it in v[:3]))

    run(tv, "solo TESTO")
    run(b32_single, "B-32 foto-singola")
    run(b32_multi, "B-32 multi-foto", eyeball=True)
    run(l14_single, "L-14 foto-singola")
    run(l14_multi, "L-14 multi-foto", eyeball=True)
    run(np.concatenate([0.4*tv, 0.6*b32_multi], axis=1), "COMB testo0.4/B-32-multi")
    run(np.concatenate([0.4*tv, 0.6*l14_multi], axis=1), "COMB testo0.4/L-14-multi")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        if not CACHE.exists():
            build()
        sweep()
