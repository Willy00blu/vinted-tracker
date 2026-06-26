"""Broad A/B: text-only vs text+image on the REAL pipeline (cat/brand partition
+ refine), over a wide sample of popular products. Phase 1 scrapes + downloads
images + embeds (cached to /tmp/mm_sample.npz). Phase 2 sweeps the blend weight
and measures coverage / brand-purity / price-coherence.

  python3 scripts/mm_broad_test.py          # phase1 (if needed) + phase2
  python3 scripts/mm_broad_test.py sweep     # phase2 only (reuse cache)
"""
import sys, io, urllib.request, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from scripts.cluster_ab2 import base_partition
from scripts.cluster_experiment import _tokenize

CACHE = Path("/tmp/mm_sample.npz")
QUERIES = ["iphone 11", "iphone 13", "airpods", "nintendo switch", "nintendo switch jeu",
           "ps4", "ps4 jeu", "ps5", "xbox series", "samsung galaxy", "macbook air",
           "apple watch", "jbl", "gopro", "dyson", "kindle"]
CAP = 900


def build_cache():
    from core.vinted_client import safe_search
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    items = {}
    for q in QUERIES:
        for p in (1, 2):
            try:
                its = safe_search("www.vinted.it", {"search_text": q, "order": "newest_first",
                                  "per_page": 96, "page": p})
            except Exception as e:
                print(f"  {q} p{p}: {e}"); continue
            for it in its:
                ph = it.photos[0].url if it.photos else None
                if ph and it.title and (it.price or 0) > 0:
                    items[it.id] = {"id": it.id, "title": it.title,
                                    "brand": it.brand_title or "", "price": float(it.price),
                                    "photo": ph}
    items = list(items.values())[:CAP]
    print(f"scrapati (cap {CAP}): {len(items)}  — scarico immagini…", flush=True)

    def dl(url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=10).read())).convert("RGB")
        except Exception:
            return None
    imgs, keep = [], []
    for i, it in enumerate(items):
        im = dl(it["photo"])
        if im is not None:
            imgs.append(im); keep.append(it)
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(items)}", flush=True)
    print(f"immagini ok: {len(keep)}  — embeddo testo (e5) + immagine (CLIP)…", flush=True)
    tv = cl._get_model().encode([cl._embed_text({"title": it["title"]}) for it in keep],
                                show_progress_bar=False, convert_to_numpy=True)
    clip = SentenceTransformer("clip-ViT-B-32")
    iv = clip.encode(imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
    np.savez(CACHE,
             ids=np.array([it["id"] for it in keep]),
             titles=np.array([it["title"] for it in keep], dtype=object),
             brands=np.array([it["brand"] for it in keep], dtype=object),
             prices=np.array([it["price"] for it in keep]),
             tv=tv, iv=iv)
    print(f"cache salvata: {CACHE}")


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def metrics(mapping, items, name):
    n = len(items)
    by = collections.defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    sizes = [len(v) for v in by.values()]
    cov = sum(s for s in sizes if s >= 2)
    pok = num = den = 0
    for v in by.values():
        if len(v) < 2:
            continue
        pr = [it["price"] for it in v if it["price"]]
        if len(pr) > 1 and st.mean(pr) and st.pstdev(pr) / st.mean(pr) <= 0.9:
            pok += len(v)
        brs = [_tokenize(it["title"], it["brand"])[1] for it in v]
        brs = [b for b in brs if b]
        if brs:
            num += collections.Counter(brs).most_common(1)[0][1]; den += len(brs)
    print(f"{name:28} cluster={len(sizes):4} copertura={100*cov/n:3.0f}% "
          f"brand-purity={100*num/max(1,den):3.0f}% price-coher={100*pok/n:3.0f}%")


def sweep():
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv, iv = norm(d["tv"]), norm(d["iv"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    print(f"campione: {len(items)}\n=== pipeline reale (cat/brand + refine) ===")

    def run(vecs, name):
        cache = {it["id"]: vecs[emap[it["id"]]] for it in items}
        with __import__("core.category_db", fromlist=["connect"]).connect(Path("/tmp/mm.db")) as conn:
            base = base_partition(items, cache, "catbrand")
            m = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=cache)
        metrics(m, items, name)

    run(tv, "solo TESTO (attuale)")
    run(iv, "solo IMMAGINE")
    for a in (0.2, 0.3, 0.4, 0.5):
        comb = np.concatenate([a * tv, (1 - a) * iv], axis=1)
        run(comb, f"COMBINATO testo={a} img={1-a:.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        if not CACHE.exists():
            build_cache()
        sweep()
