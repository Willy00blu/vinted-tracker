"""Same test as fashion_clip_test but on ELECTRONICS dirty clusters, with the
bigger CLIP ViT-L-14. Queries chosen to create messy multi-product clusters
(ps4 = console+games+controllers, iphone = many models, ...). Cache /tmp/elec_l14.npz.

  python3 scripts/elec_clip_l14.py          # build + compare
  python3 scripts/elec_clip_l14.py sweep
"""
import sys, io, time, urllib.request, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
# All models (e5, clip-ViT-B-32, clip-ViT-L-14) are pre-cached → run HF OFFLINE
# so no httpx download happens in the same process as the curl_cffi scraper
# (that conflict caused a "client has been closed" crash).
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from scripts.cluster_ab2 import base_partition
from scripts.fashion_clip_test import norm, report  # reuse helpers

CACHE = Path("/tmp/elec_l14.npz")
CLIP_MODEL = "clip-ViT-L-14"
QUERIES = ["ps4", "playstation 4", "ps4 console", "ps4 jeu",
           "nintendo switch", "nintendo switch console", "nintendo switch jeu",
           "nintendo switch oled"]
CAP = 700


def build():
    from core.vinted_client import safe_search, try_domains
    from core.discovery import DOMAINS_DEFAULT
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    items = {}
    for q in QUERIES:
        for p in (1, 2):
            params = {"search_text": q, "order": "newest_first", "per_page": 96, "page": p}
            # rate-limited path (shared limiter, like the monitor) → no 429
            res, used, errs = try_domains(
                DOMAINS_DEFAULT, lambda d, pr=params: safe_search(d, pr), key="imgtest")
            if not res:
                print(f"  {q} p{p}: {errs}"); continue
            for it in res:
                ph = it.photos[0].url if it.photos else None
                if ph and it.title and (it.price or 0) > 0:
                    items[it.id] = {"id": it.id, "title": it.title,
                                    "brand": it.brand_title or "", "price": float(it.price), "photo": ph}
    items = list(items.values())[:CAP]
    print(f"scrapati: {len(items)} — scarico immagini…", flush=True)

    def dl(u):
        try:
            r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            return Image.open(io.BytesIO(urllib.request.urlopen(r, timeout=10).read())).convert("RGB")
        except Exception:
            return None
    imgs, keep = [], []
    for i, it in enumerate(items):
        im = dl(it["photo"])
        if im is not None:
            imgs.append(im); keep.append(it)
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(items)}", flush=True)
    print(f"immagini ok: {len(keep)} — embeddo e5 + B-32 + L-14…", flush=True)
    tv = cl._get_model().encode([cl._embed_text({"title": it["title"]}) for it in keep],
                                show_progress_bar=False, convert_to_numpy=True)
    iv_b32 = SentenceTransformer("clip-ViT-B-32").encode(
        imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
    iv_l14 = SentenceTransformer("clip-ViT-L-14").encode(
        imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=16)
    np.savez(CACHE, ids=np.array([it["id"] for it in keep]),
             titles=np.array([it["title"] for it in keep], dtype=object),
             brands=np.array([it["brand"] for it in keep], dtype=object),
             prices=np.array([it["price"] for it in keep]),
             tv=tv, iv_b32=iv_b32, iv_l14=iv_l14)
    print("cache salvata (B-32 + L-14)")


def sweep():
    from core.category_db import connect
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv = norm(d["tv"]); b32 = norm(d["iv_b32"]); l14 = norm(d["iv_l14"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    print(f"campione ELETTRONICA: {len(items)}\n")

    def run(vecs, name):
        cache = {it["id"]: vecs[emap[it["id"]]] for it in items}
        with connect(Path("/tmp/elec.db")) as conn:
            base = base_partition(items, cache, "catbrand")
            m = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=cache)
        report(m, items, name); return m

    text_map = run(tv, "solo TESTO")
    run(b32, "solo IMMAGINE B-32")
    run(l14, "solo IMMAGINE L-14")
    run(np.concatenate([0.3*tv, 0.7*b32], axis=1), "COMB testo0.3/B-32")
    run(np.concatenate([0.3*tv, 0.7*l14], axis=1), "COMB testo0.3/L-14")
    run(np.concatenate([0.5*tv, 0.5*l14], axis=1), "COMB testo0.5/L-14")

    # targeted: do B-32 vs L-14 fix the MESSY text clusters? (n>=6, CV>0.7)
    by = collections.defaultdict(list)
    for it in items: by[text_map[it["id"]]].append(it)
    def cv(v):
        p = [it["price"] for it in v if it["price"]]
        return (st.pstdev(p)/st.mean(p)) if len(p) > 1 and st.mean(p) else 0
    messy = [(k, v) for k, v in by.items() if len(v) >= 6 and cv(v) > 0.7]
    H = cl._get_hdbscan().HDBSCAN
    print(f"\n=== MESSY (n>=6,CV>0.7): {len(messy)} — quanti li sistema l'immagine? ===")
    for mname, ivec in (("B-32", b32), ("L-14", l14)):
        improved = 0
        for k, v in messy:
            X = norm(np.array([ivec[emap[it["id"]]] for it in v]))
            lab = H(min_cluster_size=3, metric="euclidean", cluster_selection_method="leaf",
                    core_dist_n_jobs=-1).fit_predict(X)
            sub = collections.defaultdict(list)
            for it, l in zip(v, lab):
                sub[int(l)].append(it)
            real = {l: g for l, g in sub.items() if l != -1 and len(g) >= 2}
            if not real: continue
            after = sum(cv(g)*len(g) for g in real.values())/sum(len(g) for g in real.values())
            if after < cv(v) - 0.1:
                improved += 1
        print(f"  {mname}: {improved}/{len(messy)} cluster sporchi migliorati")

    # EYEBALL: image-cluster ALL items by L-14 → vedere se separa console/giochi/accessori
    print("\n=== EYEBALL: tutti gli item raggruppati per IMMAGINE L-14 (primi 12) ===")
    X = l14
    lab = H(min_cluster_size=4, metric="euclidean", cluster_selection_method="leaf",
            core_dist_n_jobs=-1).fit_predict(X)
    g = collections.defaultdict(list)
    for it, l in zip(items, lab):
        g[int(l)].append(it)
    for l, v in sorted(((l, v) for l, v in g.items() if l != -1), key=lambda kv: -len(kv[1]))[:12]:
        pr = [it["price"] for it in v]
        print(f"  n={len(v):3} med€{st.median(pr):>4.0f}: " +
              " | ".join(it["title"][:24] for it in v[:4]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        if not CACHE.exists():
            build()
        sweep()
