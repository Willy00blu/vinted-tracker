"""Test a BIGGER CLIP (ViT-L-14) on the FASHION domain (where literature says
images dominate). Scrapes fashion products that create 'dirty' text clusters
(model variants lumped by generic titles), embeds text (e5) + image (CLIP-L-14),
and compares text/image/combined on the real pipeline. Cache /tmp/fashion.npz.

  python3 scripts/fashion_clip_test.py          # build cache + compare
  python3 scripts/fashion_clip_test.py sweep     # compare only
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

CACHE = Path("/tmp/fashion.npz")
CLIP_MODEL = "clip-ViT-L-14"
QUERIES = ["nike air max", "nike air force 1", "adidas samba", "jordan 1",
           "new balance 550", "zara dress", "levis 501", "north face jacket",
           "carhartt", "ralph lauren polo", "stone island", "moncler",
           "nike hoodie", "adidas tracksuit"]
CAP = 700


def build():
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
    print(f"immagini ok: {len(keep)} — embeddo e5 + {CLIP_MODEL}…", flush=True)
    tv = cl._get_model().encode([cl._embed_text({"title": it["title"]}) for it in keep],
                                show_progress_bar=False, convert_to_numpy=True)
    clip = SentenceTransformer(CLIP_MODEL)
    iv = clip.encode(imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=16)
    np.savez(CACHE, ids=np.array([it["id"] for it in keep]),
             titles=np.array([it["title"] for it in keep], dtype=object),
             brands=np.array([it["brand"] for it in keep], dtype=object),
             prices=np.array([it["price"] for it in keep]), tv=tv, iv=iv)
    print("cache salvata")


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def report(mapping, items, name):
    n = len(items); by = collections.defaultdict(list)
    for it in items: by[mapping[it["id"]]].append(it)
    cov = sum(len(v) for v in by.values() if len(v) >= 2)
    pok = num = den = 0
    for v in by.values():
        if len(v) < 2: continue
        pr = [it["price"] for it in v if it["price"]]
        if len(pr) > 1 and st.mean(pr) and st.pstdev(pr)/st.mean(pr) <= 0.9: pok += len(v)
        brs = [_tokenize(it["title"], it["brand"])[1] for it in v]; brs = [b for b in brs if b]
        if brs: num += collections.Counter(brs).most_common(1)[0][1]; den += len(brs)
    print(f"{name:26} cluster={len([1 for v in by.values() if len(v)>=2]):3} "
          f"cov={100*cov/n:3.0f}% brand-pur={100*num/max(1,den):3.0f}% price-coh={100*pok/n:3.0f}%")


def sweep():
    from core.category_db import connect
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv, iv = norm(d["tv"]), norm(d["iv"]); emap = {it["id"]: k for k, it in enumerate(items)}
    print(f"campione MODA: {len(items)}  (img model {CLIP_MODEL})\n")

    def run(vecs, name):
        cache = {it["id"]: vecs[emap[it["id"]]] for it in items}
        with connect(Path("/tmp/fash.db")) as conn:
            base = base_partition(items, cache, "catbrand")
            m = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=cache)
        report(m, items, name); return m

    run(tv, "solo TESTO")
    run(iv, "solo IMMAGINE (L-14)")
    best = None
    for a in (0.2, 0.3, 0.4, 0.5):
        run(np.concatenate([a*tv, (1-a)*iv], axis=1), f"COMB testo={a} img={1-a:.1f}")
    # eyeball dirty: image-only groups for a brand
    m = run(iv, "(img per eyeball)")
    by = collections.defaultdict(list)
    for it in items: by[m[it["id"]]].append(it)
    print("\n=== gruppi IMMAGINE (L-14), primi 8 per dimensione ===")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]:
        if len(v) < 3: continue
        pr = [it["price"] for it in v]
        print(f"  n={len(v)} med€{st.median(pr):.0f}: " + " | ".join(it["title"][:22] for it in v[:4]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        if not CACHE.exists():
            build()
        sweep()
