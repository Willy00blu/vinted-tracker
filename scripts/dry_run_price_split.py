"""Dry-run of _price_split_jenks2 on the current DB.

Embeds ONLY the items in candidate clusters (size>=8 + CV>0.5) — much
cheaper than embedding the whole 88k corpus — runs the full split logic
including the embedding-cosine guard, and reports:
  - structural candidates (passed size+CV gates)
  - jenks2 candidates (passed GVF+ratio gates)
  - actually split (passed embedding-dissimilarity gate)
  - rejected by embedding (same-product = protected as deal)
  - examples of accepted / rejected splits
"""
from __future__ import annotations

import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import connect, shared_db_path                # noqa: E402
from core.clustering import _price_split_jenks2, embed_items        # noqa: E402


def main():
    db = shared_db_path()
    print(f"DB: {db.name}")
    with connect(db) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT id, cluster_key, title, brand, price FROM items "
            "WHERE removed=0 AND cluster_key NOT IN ('unknown','') "
            "AND title IS NOT NULL AND price IS NOT NULL AND price>0"
        )]

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster_key"]].append(r)

    # Structural pre-filter: only candidates we'd ever consider.
    min_size, min_cv = 8, 0.5
    candidates: dict[str, list[dict]] = {}
    for k, v in by_cluster.items():
        if len(v) < min_size:
            continue
        prs = [r["price"] for r in v]
        m = st.mean(prs)
        if not m:
            continue
        if st.pstdev(prs) / m <= min_cv:
            continue
        candidates[k] = v
    print(f"\nstructural candidates (size>=8, CV>0.5): {len(candidates)}  "
          f"({sum(len(v) for v in candidates.values())} item)")

    # Embed only the candidate items — much cheaper than the whole corpus.
    items_to_embed = [it for v in candidates.values() for it in v]
    print(f"embedding {len(items_to_embed)} titoli con e5-large (CPU)…")
    emb_of = embed_items(items_to_embed)
    print(f"  done — {len(emb_of)} vettori")

    # Apply Jenks2 with default guards (from the function signature).
    logs: list[str] = []
    out = _price_split_jenks2(candidates, emb_of, lambda s: logs.append(s))
    for line in logs:
        print(line.strip())

    # Re-derive accepted/rejected by inspecting `out`.
    accepted: list[tuple] = []     # (k, n, A_range, B_range, gvf?, ratio?)
    rejected_emb: list[tuple] = [] # we have to re-compute manually for examples
    rejected_struct: list[tuple] = []
    # `out` keys that are NEW (have " [€...]") came from a split.
    for new_k, v in out.items():
        if "[€" in new_k:
            base = new_k.split(" [€")[0]
            prs = sorted(r["price"] for r in v)
            accepted.append((base, len(v), prs[0], prs[-1], new_k))

    # Map accepted by base cluster
    accepted_by_base: dict[str, list[tuple]] = defaultdict(list)
    for base, n, lo, hi, full_k in accepted:
        accepted_by_base[base].append((n, lo, hi, full_k))

    print(f"\nVERDETTO:")
    print(f"  structural candidates                  : {len(candidates)}")
    print(f"  splittati (passa anche emb-check)      : {len(accepted_by_base)}")
    print(f"  bloccati da gates (gvf/ratio/emb)      : {len(candidates) - len(accepted_by_base)}")

    print("\n=== 15 SPLIT ACCETTATI (titoli diversi → prodotti diversi) ===")
    # Show original cluster + the two split bands
    for base, parts in list(accepted_by_base.items())[:15]:
        parts.sort(key=lambda x: x[1])    # by lo price
        rng = " + ".join(f"{p[0]} a €{p[1]:.0f}-{p[2]:.0f}" for p in parts)
        print(f"  {base[:50]:50}  →  {rng}")

    # For rejected examples (where Jenks2 would have liked to split but emb blocked),
    # we run the function with max_emb_cos=1.0 (no emb block) on a small subset and
    # compare. Skipping for now — Jenks2 already logs the count.

    print("\n=== il Wii câbles e l'Apple mappazzone ===")
    for needle in ("wii câbles", "iphone 11 64go"):
        for new_k, v in out.items():
            if needle in new_k:
                prs = sorted(r["price"] for r in v)
                kind = "SPLIT" if "[€" in new_k else "non spaccato"
                print(f"  [{kind}] {new_k[:65]}  size {len(v)}  €{prs[0]:.0f}-{prs[-1]:.0f}")


if __name__ == "__main__":
    main()
