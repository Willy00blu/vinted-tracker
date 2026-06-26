"""Build the persistent testing cache (stable location, memory-frugal).

Writes to data/ab_cache/ (gitignored):
  emb_f16.npy   (N,1024) float16 text embeddings   ~414MB
  ids.npy       (N,)     int64 item ids
  items.pkl     [{id,title,brand,price,category}]
  img_f16.npy   (M,D)    float16 CLIP image vectors
  img_ids.npy   (M,)     int64 ids aligned to img_f16
  subset.json   {"ids":[...]}  ~60k ids = whole partitions, for FAST iteration

float16 halves RAM/disk; cosine is unaffected at this precision. The subset
preserves WHOLE category|brand partitions so cluster structure is realistic.
"""
import sys, os, json, pickle, collections, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
from core.category_db import shared_db_path, connect

OUT = ROOT / "data" / "ab_cache"
OUT.mkdir(parents=True, exist_ok=True)
SUBSET_TARGET = 60000
TECH_BRANDS = {"intel", "amd", "nvidia"}


def part_key(it):
    cat = (it.get("category") or "").strip()
    brand, _ = cl._lex_sig(it.get("title", ""), it.get("brand"))
    brand = (cl._normalize_brand(brand) or brand or "").lower()
    if cat and brand:
        return f"{cat}|{brand}"
    if cat:
        return cat
    return brand or "__nobrand__"


def main():
    t0 = time.time()
    ids, embs, items = [], [], []
    img_ids, img_vecs = [], []
    n_seen = 0
    with connect(shared_db_path()) as c:
        cur = c.execute(
            "SELECT id,title,brand,price,category,text_emb,image_emb FROM items "
            "WHERE removed=0 AND title IS NOT NULL AND text_emb IS NOT NULL"
        )
        for r in cur:                       # stream: don't hold all blobs at once
            n_seen += 1
            try:
                e = cl._blob_to_emb(r["text_emb"])
                if e.shape[0] != 1024:
                    continue
            except Exception:
                continue
            ids.append(r["id"]); embs.append(e.astype(np.float16))
            items.append({"id": r["id"], "title": r["title"], "brand": r["brand"],
                          "price": r["price"], "category": r["category"]})
            if r["image_emb"] is not None:
                try:
                    iv = cl._blob_to_emb(r["image_emb"])
                    img_ids.append(r["id"]); img_vecs.append(iv.astype(np.float16))
                except Exception:
                    pass
            if n_seen % 50000 == 0:
                print(f"  {n_seen} rows… ({time.time()-t0:.0f}s)", flush=True)
    print(f"streamed {n_seen} rows, kept {len(ids)} in {time.time()-t0:.0f}s", flush=True)

    E = np.vstack(embs); del embs
    np.save(OUT / "emb_f16.npy", E)
    np.save(OUT / "ids.npy", np.array(ids, dtype=np.int64))
    pickle.dump(items, open(OUT / "items.pkl", "wb"))
    IMG = np.vstack(img_vecs); del img_vecs
    np.save(OUT / "img_f16.npy", IMG)
    np.save(OUT / "img_ids.npy", np.array(img_ids, dtype=np.int64))
    print(f"text {E.shape} ({E.nbytes/1e6:.0f}MB) | img {IMG.shape} "
          f"({IMG.nbytes/1e6:.0f}MB)", flush=True)

    # ── subset: whole partitions until ~SUBSET_TARGET, biggest first, but ALWAYS
    #    include tech (intel/amd/nvidia) partitions for the CPU model-token test.
    by = collections.defaultdict(list)
    for it in items:
        by[part_key(it)].append(it["id"])
    parts = sorted(by.items(), key=lambda kv: -len(kv[1]))
    chosen, n = [], 0
    tech_parts = [p for p, v in by.items()
                  if any(b in p for b in TECH_BRANDS) or "process" in p or "cpu" in p]
    for p in tech_parts:
        chosen.append(p); n += len(by[p])
    for p, v in parts:
        if p in chosen:
            continue
        if n >= SUBSET_TARGET:
            break
        # skip the giant __nobrand__ residual (would dominate + slow); keep shards
        if p == "__nobrand__":
            continue
        chosen.append(p); n += len(v)
    sub_ids = [iid for p in chosen for iid in by[p]]
    json.dump({"ids": sub_ids, "parts": chosen}, open(OUT / "subset.json", "w"))
    print(f"subset: {len(sub_ids)} items in {len(chosen)} partitions "
          f"(tech parts: {len(tech_parts)})", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
