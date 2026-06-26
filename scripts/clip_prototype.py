"""Prototype: do IMAGE embeddings (CLIP) separate same-text-different-product
items where text fails? Scrapes ambiguous gaming items (console+games+accessory
all titled 'nintendo switch'/'ps4'), embeds titles (e5) and photos (CLIP), and
compares within-type vs across-type similarity for each modality.

If image contrast (within-type sim minus across-type sim) >> text contrast,
adding images would meaningfully improve clustering.
"""
import sys, io, urllib.request, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
# Force HF online BEFORE importing core.clustering (which would set offline mode
# because e5 is cached) so CLIP can be downloaded the first time.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.vinted_client import safe_search
import core.clustering as cl
from scripts.cluster_typesplit import _detect_type
from PIL import Image
from sentence_transformers import SentenceTransformer


def scrape(query, pages=2):
    out = []
    for p in range(1, pages + 1):
        try:
            items = safe_search("www.vinted.it",
                                {"search_text": query, "order": "newest_first",
                                 "per_page": 96, "page": p})
        except Exception as e:
            print(f"  scrape {query} p{p} fallito: {e}"); break
        for it in items:
            ph = it.photos[0].url if it.photos else None
            if ph and it.title:
                out.append({"id": it.id, "title": it.title,
                            "price": (it.price or 0), "photo": ph})
    return out


def dl_image(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=10).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def cos_matrix(V):
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return V @ V.T


def contrast(sim, types):
    within, across = [], []
    n = len(types)
    for i in range(n):
        for j in range(i + 1, n):
            (within if types[i] == types[j] else across).append(sim[i, j])
    return (st.mean(within) if within else 0,
            st.mean(across) if across else 0, len(within), len(across))


def main():
    items = []
    for q in ("nintendo switch", "ps4", "playstation 4"):
        items += scrape(q, pages=2)
    # de-dup by id
    seen = {}
    for it in items:
        seen[it["id"]] = it
    items = list(seen.values())
    # keep only items with a detectable type
    for it in items:
        it["type"] = _detect_type(it["title"])
    typed = [it for it in items if it["type"] in ("console", "game", "controller")]
    by_t = collections.Counter(it["type"] for it in typed)
    print(f"scrapati: {len(items)}  con tipo chiaro: {len(typed)}  {dict(by_t)}")
    # balance: cap per type
    cap = 35
    buckets = collections.defaultdict(list)
    for it in typed:
        if len(buckets[it["type"]]) < cap:
            buckets[it["type"]].append(it)
    sample = [it for v in buckets.values() for it in v]
    print(f"campione bilanciato: {len(sample)} ({ {t: len(v) for t, v in buckets.items()} })")

    # download images
    print("scarico immagini…", flush=True)
    imgs, keep = [], []
    for it in sample:
        im = dl_image(it["photo"])
        if im is not None:
            imgs.append(im); keep.append(it)
    print(f"immagini ok: {len(keep)}")
    types = [it["type"] for it in keep]

    # TEXT embeddings (e5)
    txt_model = cl._get_model()
    tv = txt_model.encode([cl._embed_text({"title": it["title"]}) for it in keep],
                          show_progress_bar=False, convert_to_numpy=True)
    # IMAGE embeddings (CLIP)
    print("carico CLIP + embeddo immagini…", flush=True)
    clip = SentenceTransformer("clip-ViT-B-32")
    iv = clip.encode(imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=32)

    ts, ta, nw, na = contrast(cos_matrix(tv), types)
    ims, ima, _, _ = contrast(cos_matrix(iv), types)
    print(f"\n{'modalità':10} {'within':>7} {'across':>7} {'CONTRASTO':>10}")
    print(f"{'TESTO e5':10} {ts:>7.3f} {ta:>7.3f} {ts-ta:>10.3f}")
    print(f"{'IMMAGINE':10} {ims:>7.3f} {ima:>7.3f} {ims-ima:>10.3f}")
    print(f"\ncoppie within={nw} across={na}")
    print("→ contrasto più alto = separa meglio console/giochi/controller")


if __name__ == "__main__":
    main()
