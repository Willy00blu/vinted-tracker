"""Validate the gated two-stage on a RANDOM population sample (Tecnologia catalog
2994, newest), not the messy-biased PS4/Switch queries. Multi-photo + L-14.
Checks it FIXES messy clusters AND leaves clean ones intact. Cache
/tmp/random_tech.npz.

  python3 scripts/random_tech_test.py          # build + eval
  python3 scripts/random_tech_test.py sweep
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
from core.category_db import connect
from scripts.cluster_ab2 import base_partition
from scripts.refine_sweep import norm, cvp, metrics, two_stage

CACHE = Path("/tmp/random_tech.npz")
CATALOG = "2994"
PAGES = 10
CAP = 800
MAX_PHOTOS = 4


def build():
    from core.vinted_client import safe_search, try_domains
    from core.discovery import DOMAINS_DEFAULT
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    items = {}
    for p in range(1, PAGES + 1):
        params = {"catalog_ids": CATALOG, "order": "newest_first", "per_page": 96, "page": p}
        res, used, errs = try_domains(
            DOMAINS_DEFAULT, lambda d, pr=params: safe_search(d, pr), key="randtech")
        if not res:
            print(f"  p{p}: {errs}"); continue
        for it in res:
            urls = [ph.url for ph in (it.photos or []) if ph.url][:MAX_PHOTOS]
            if urls and it.title and (it.price or 0) > 0:
                items[it.id] = {"id": it.id, "title": it.title, "brand": it.brand_title or "",
                                "price": float(it.price), "urls": urls}
    items = list(items.values())[:CAP]
    print(f"scrapati: {len(items)} — scarico foto…", flush=True)

    def dl(u):
        try:
            r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            return Image.open(io.BytesIO(urllib.request.urlopen(r, timeout=10).read())).convert("RGB")
        except Exception:
            return None
    flat, fidx, keep = [], [], []
    for it in items:
        ims = [im for im in (dl(u) for u in it["urls"]) if im is not None]
        if not ims:
            continue
        idx = len(keep); keep.append(it)
        for im in ims:
            flat.append(im); fidx.append(idx)
        if len(keep) % 100 == 0:
            print(f"   {len(keep)}", flush=True)
    print(f"item: {len(keep)}  foto: {len(flat)} — embeddo e5 + L-14…", flush=True)
    tv = cl._get_model().encode([cl._embed_text({"title": it["title"]}) for it in keep],
                                show_progress_bar=False, convert_to_numpy=True)
    ev = SentenceTransformer("clip-ViT-L-14").encode(flat, show_progress_bar=False,
                                                     convert_to_numpy=True, batch_size=32)
    dim = ev.shape[1]; multi = np.zeros((len(keep), dim)); acc = collections.defaultdict(list)
    for e, idx in zip(ev, fidx):
        acc[idx].append(e)
    for idx, es in acc.items():
        multi[idx] = np.mean(es, axis=0)
    np.savez(CACHE, ids=np.array([it["id"] for it in keep]),
             titles=np.array([it["title"] for it in keep], dtype=object),
             brands=np.array([it["brand"] for it in keep], dtype=object),
             prices=np.array([it["price"] for it in keep]), tv=tv, l14_multi=multi)
    print("cache salvata")


def sweep():
    d = np.load(CACHE, allow_pickle=True)
    items = [{"id": int(i), "title": t, "brand": b, "price": float(p)}
             for i, t, b, p in zip(d["ids"], d["titles"], d["brands"], d["prices"])]
    tv = norm(d["tv"]); img = norm(d["l14_multi"])
    emap = {it["id"]: k for k, it in enumerate(items)}
    tcache = {it["id"]: tv[emap[it["id"]]] for it in items}
    print(f"campione CASUALE Tecnologia: {len(items)}\n")

    with connect(Path("/tmp/rt.db")) as conn:
        base = base_partition(items, tcache, "catbrand")
        text_map = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=tcache)
    # population breakdown
    by = collections.defaultdict(list)
    for it in items: by[text_map[it["id"]]].append(it)
    multi_cl = [v for v in by.values() if len(v) >= 2]
    clean = [v for v in multi_cl if cvp(v) <= 0.7]
    messy = [v for v in multi_cl if cvp(v) > 0.7]
    print(f"cluster (≥2): {len(multi_cl)}  di cui PULITI(CV≤0.7): {len(clean)}  SPORCHI(CV>0.7): {len(messy)}")
    cov, pc = metrics(text_map, items)
    print(f"\n{'TESTO-only':36} cov={cov:3.0f}% price-coh={pc:3.0f}%")

    m, mf, cs = two_stage(text_map, items, img, emap, 2, cv_gate=0.7)
    cov, pc = metrics(m, items)
    print(f"{'TESTO + img-refine gated (L-14)':36} cov={cov:3.0f}% price-coh={pc:3.0f}% "
          f"messy_fix={mf}/{len(messy)} clean_split(BAD)={cs}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        if not CACHE.exists():
            build()
        sweep()
