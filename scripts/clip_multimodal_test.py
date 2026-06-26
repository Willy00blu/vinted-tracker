"""Compare clustering of the same PS4 items by: image-only, text-only, and
text+image combined (concat with weight alpha). Metrics: noise%, coverage
(items in clusters>=2), and median within-cluster price-CV (lower = groups are
more 'same product').
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


def scrape(q, pages=2):
    out = []
    for p in range(1, pages + 1):
        try:
            its = safe_search("www.vinted.it", {"search_text": q, "order": "newest_first",
                              "per_page": 96, "page": p})
        except Exception as e:
            print(f"  {q} p{p}: {e}"); break
        for it in its:
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


def norm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def evaluate(feat, keep, name):
    H = cl._get_hdbscan().HDBSCAN
    lab = H(min_cluster_size=4, metric="euclidean",
            cluster_selection_method="leaf", core_dist_n_jobs=-1).fit_predict(norm(feat))
    groups = collections.defaultdict(list)
    for it, l in zip(keep, lab):
        groups[int(l)].append(it)
    n = len(keep)
    noise = len(groups.get(-1, []))
    multi = {l: v for l, v in groups.items() if l != -1 and len(v) >= 2}
    cov = sum(len(v) for v in multi.values())
    cvs = []
    for v in multi.values():
        pr = [it["price"] for it in v]
        if len(pr) > 1 and st.mean(pr):
            cvs.append(st.pstdev(pr) / st.mean(pr))
    print(f"{name:24} cluster={len(multi):3} copertura={100*cov/n:3.0f}% rumore={100*noise/n:3.0f}% "
          f"CV-prezzo-mediano={st.median(cvs) if cvs else 0:.2f}")
    return groups


def main():
    items = {}
    for q in ("ps4", "playstation 4", "jeu ps4"):
        for it in scrape(q, 2):
            items[it["id"]] = it
    items = list(items.values())
    imgs, keep = [], []
    for it in items:
        im = dl(it["photo"])
        if im is not None:
            imgs.append(im); keep.append(it)
    print(f"item: {len(keep)}\n")

    txt = cl._get_model()
    tv = norm(txt.encode([cl._embed_text({"title": it["title"]}) for it in keep],
                         show_progress_bar=False, convert_to_numpy=True))
    clip = SentenceTransformer("clip-ViT-B-32")
    iv = norm(clip.encode(imgs, show_progress_bar=False, convert_to_numpy=True, batch_size=32))

    print("=== CONFRONTO ===")
    evaluate(iv, keep, "solo IMMAGINE")
    evaluate(tv, keep, "solo TESTO")
    for a in (0.3, 0.5, 0.7):
        comb = np.concatenate([a * tv, (1 - a) * iv], axis=1)
        evaluate(comb, keep, f"COMBINATO testo={a} img={1-a:.1f}")

    # eyeball the combined 0.5
    print("\n=== gruppi COMBINATO (testo0.5/img0.5), primi 8 ===")
    comb = np.concatenate([0.5 * tv, 0.5 * iv], axis=1)
    g = evaluate(comb, keep, "(combinato 0.5)")
    for l, v in sorted(((l, v) for l, v in g.items() if l != -1 and len(v) >= 2),
                       key=lambda kv: -len(kv[1]))[:8]:
        pr = [it["price"] for it in v]
        print(f"  n={len(v)} med€{st.median(pr):.0f}: " + " | ".join(it["title"][:20] for it in v[:4]))


if __name__ == "__main__":
    main()
