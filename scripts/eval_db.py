"""Print the 4-axis cluster quality of a DB's stored cluster_key.

Reads `text_emb` from the DB cache when available (= TEXT_EMB_VERSION matches);
falls back to live encoding only for items still missing a fresh vector. With
the text_worker keeping the cache full, a full-DB eval runs in seconds.

Usage: python3 scripts/eval_db.py <db_path> "<label>"
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
from core.category_db import connect
from core.clustering import (
    _get_model, _embed_text, _blob_to_emb,
    TEXT_EMB_VERSION, eval_metrics,
)


def metrics(path, name):
    with connect(Path(path)) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,cluster_key,text_emb,text_emb_version "
            "FROM items WHERE removed=0 AND title IS NOT NULL")]
    # 1) load cached embeddings where the version matches the current model
    emb_of: dict = {}
    todo: list[dict] = []
    for it in items:
        blob = it.get("text_emb")
        if blob is not None and it.get("text_emb_version") == TEXT_EMB_VERSION:
            try:
                emb_of[it["id"]] = _blob_to_emb(blob)
                continue
            except Exception:
                pass
        todo.append(it)
    print(f"  embeddings: {len(emb_of)}/{len(items)} from DB cache "
          f"(text_emb_version={TEXT_EMB_VERSION})")
    # 2) only re-encode the stragglers
    if todo:
        print(f"  re-encoding {len(todo)} fresh…")
        e = _get_model().encode([_embed_text(it) for it in todo],
                                show_progress_bar=False,
                                convert_to_numpy=True, batch_size=256)
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        for it, v in zip(todo, e):
            emb_of[it["id"]] = v
    # 3) standard 4-axis metrics
    mapping = {it["id"]: it["cluster_key"] for it in items}
    m = eval_metrics(items, mapping, emb_of)
    print(f"=== {name} ===")
    print(f"  items:{m['items']} cluster:{m['cluster']} "
          f"singleton:{m['singleton_pct']:.0f}%cl")
    print(f"  coverage(>=2):{m['coverage']:.0f}%  "
          f"cohesion:{m['cohesion']:.3f}  "
          f"price-coherent:{m['price_coh']:.0f}%  "
          f"brand-purity:{m['brand_pure']:.0f}%  "
          f"SCORE:{m['score']:.1f}")


if __name__ == "__main__":
    metrics(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
