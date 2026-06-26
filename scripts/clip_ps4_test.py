"""Decisive test: cluster ambiguous 'PS4' items BY IMAGE (CLIP) and show whether
the image-clusters separate console / games / controllers (by what they ARE),
with their price medians — the exact case text+price couldn't split.
"""
import sys, io, urllib.request, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.vinted_client import safe_search
import core.clustering as cl
from PIL import Image
from sentence_transformers import SentenceTransformer


def scrape(query, pages=2):
    out = []
    for p in range(1, pages + 1):
        try:
            items = safe_search("www.vinted.it", {"search_text": query,
                                "order": "newest_first", "per_page": 96, "page": p})
        except Exception as e:
            print(f"  scrape {query} p{p}: {e}"); break
        for it in items:
            ph = it.photos[0].url if it.photos else None
            if ph and it.title and (it.price or 0) > 0:
                out.append({"id": it.id, "title": it.title, "price": it.price, "photo": ph})
    return out


def dl(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=10).read())).convert("RGB")
    except Exception:
        return None


def main():
    items = {}
    for q in ("ps4", "playstation 4", "jeu ps4"):
        for it in scrape(q, 2):
            items[it["id"]] = it
    items = list(items.values())
    print(f"item PS4 scrapati: {len(items)}")

    imgs, keep = [], []
    for it in items:
        im = dl(it["photo"])
        if im is not None:
            imgs.append(im); keep.append(it)
    print(f"immagini ok: {len(keep)}\n")

    clip = SentenceTransformer("clip-ViT-B-32")
    iv = clip.encode(imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
    iv = iv / np.maximum(np.linalg.norm(iv, axis=1, keepdims=True), 1e-12)

    H = cl._get_hdbscan().HDBSCAN
    labels = H(min_cluster_size=4, metric="euclidean",
               cluster_selection_method="leaf", core_dist_n_jobs=-1).fit_predict(iv)

    groups = collections.defaultdict(list)
    for it, l in zip(keep, labels):
        groups[int(l)].append(it)
    print("=== GRUPPI PER IMMAGINE (CLIP) ===")
    for l, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        pr = [it["price"] for it in v]
        name = "RUMORE/misti" if l == -1 else f"img-cluster {l}"
        print(f"\n[{name}] n={len(v)} med€{st.median(pr):.0f} (range €{min(pr):.0f}-{max(pr):.0f})")
        for it in v[:6]:
            print(f"    €{it['price']:>4.0f}  {it['title'][:46]}")


if __name__ == "__main__":
    main()
