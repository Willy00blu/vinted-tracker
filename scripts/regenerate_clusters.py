"""
Regenerate cluster_key + display_label for every item in a category DB.

Wipes the cluster_centroids table, re-clusters every item from scratch using
the current embedding pipeline (no regex), and writes back items.cluster_key.
Newly-created clusters get an English display_label via translation
(Helsinki-NLP/opus-mt-mul-en, ~300MB model loaded on first use).

Use:
    python3 scripts/regenerate_clusters.py data/category_tecnologia.db
    python3 scripts/regenerate_clusters.py --all                  # all category_*.db

Safety:
    - Always makes a .bak copy of the DB before touching it.
    - All writes inside one transaction; abort = no changes.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.clustering  import (   # noqa: E402
    cluster_partitioned, cluster_titles_batch, refine_clusters, embed_items,
    image_refine, _blob_to_emb,
    EMBEDDINGS_AVAILABLE, HDBSCAN_AVAILABLE,
)
from core.category_db import SCHEMA  # noqa: E402


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, bak)
    return bak


def _flush_print(*a, **kw):
    """Always-flushing print. Background subprocesses run with block-buffered
    stdout — using `print` without `flush=True` left the regen log silent
    for the whole run, making it impossible to monitor progress."""
    kw.setdefault("flush", True)
    print(*a, **kw)


def regenerate(db_path: Path, *,
               algorithm: str = "hdbscan",   # "hdbscan" | "incremental"
               min_cluster_size: int = 2,
               log_fn=_flush_print) -> dict:
    """Wipe centroids and re-cluster every item in `db_path`.

    `algorithm`:
      - "hdbscan"     (default): density-based clustering that splits dense
                                  sub-groups (iPhone 13 / Pro / Pro Max) when
                                  there's enough data. Cluster_keys NOT stable
                                  across runs.
      - "incremental":            old threshold-based pass — stable keys but
                                  no smart splits.

    `log_fn` is a single-arg callback (default `print`) used for every line
    of progress output — pass `gui._log` when called from a Tk worker.
    """
    log_fn(f"=== {db_path.name} ===")
    bak = backup(db_path)
    log_fn(f"  backup: {bak.name}")
    log_fn(f"  algorithm: {algorithm}")

    # Pause background workers (sold_check / text / image) to avoid WAL
    # contention during the heavy regen. The GUI's sync_*_worker checks this
    # flag and refuses to (re)spawn them. SIGTERM first to let them flush, then
    # SIGKILL fallback after 2s — avoids SQLite WAL corruption from mid-commit kill.
    pause_flag = db_path.parent / ".workers_paused"
    pause_flag.touch()
    # Guarantee removal even on crash/Ctrl-C (otherwise workers stay paused
    # forever and the GUI silently does nothing).
    import atexit as _atexit
    def _cleanup_pause_flag():
        try:
            if pause_flag.exists():
                pause_flag.unlink()
        except Exception:
            pass
    _atexit.register(_cleanup_pause_flag)
    import subprocess as _sp
    for mod in ("sold_check_worker", "text_worker", "image_worker"):
        _sp.run(["pkill", "-TERM", "-f", f"core.{mod}"], check=False)
    time.sleep(2)
    for mod in ("sold_check_worker", "text_worker", "image_worker"):
        _sp.run(["pkill", "-9", "-f", f"core.{mod}"], check=False)
    log_fn("  ⏸ background workers in pausa (flag .workers_paused)")

    if algorithm == "hdbscan" and not HDBSCAN_AVAILABLE:
        log_fn("  ⚠ hdbscan package not installed → falling back to incremental")
        algorithm = "incremental"

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # Longer busy_timeout: a running monitor may briefly hold the write lock
    # during its per-item sold-check commits. 60s gives us ample room to slip
    # in between those.
    conn.execute("PRAGMA busy_timeout=60000")
    # Tighter auto-checkpoint (default 1000 pages ≈ 4 MB). 10k pages ≈ 40 MB:
    # with workers always-on, passive checkpoint is rarely triggered and the
    # WAL bloats to GB. This forces a flush more aggressively.
    conn.execute("PRAGMA wal_autocheckpoint=10000")
    conn.executescript(SCHEMA)
    try:
        conn.execute("ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()   # release any lock taken by schema/migrate before the heavy work

    n_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    log_fn(f"  items in DB: {n_items}")
    if n_items == 0:
        conn.close()
        return {"items": 0, "clusters": 0, "translated": 0}

    # ── Pulizia preliminare: cancella dal DB gli item il cui titolo è 100%
    #    noise (dopo _embed_text ritorna ""). Non hanno informazione di
    #    prodotto utile e contaminano i cluster.
    from core.clustering import _embed_text as _et
    log_fn(f"  pulizia: identifico item con titolo 100% noise…")
    purge_ids = []
    rows_titles = conn.execute(
        "SELECT id, title, brand FROM items WHERE title IS NOT NULL"
    ).fetchall()
    for r in rows_titles:
        out = _et({"title": r["title"], "brand": r["brand"]})
        # _embed_text ritorna "" (o solo "query: ") quando la pulizia svuota il titolo
        bare = out.replace("query: ", "").strip() if out else ""
        if not bare:
            purge_ids.append(r["id"])
    if purge_ids:
        log_fn(f"  pulizia: cancello {len(purge_ids)} item con titolo 100% noise")
        # cancello a chunk per evitare query troppo grande
        for i in range(0, len(purge_ids), 500):
            chunk = purge_ids[i:i+500]
            ph = ",".join("?"*len(chunk))
            conn.execute(f"DELETE FROM items WHERE id IN ({ph})", chunk)
        conn.commit()
        # ri-conto
        n_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        log_fn(f"  items dopo pulizia: {n_items}")

    rows = conn.execute(
        "SELECT id, title, brand, category, price, photo_urls, removed, "
        "       image_emb, text_emb, text_emb_version "
        "FROM items WHERE title IS NOT NULL"
    ).fetchall()
    items = [dict(r) for r in rows]

    # NOTE: we deliberately do NOT wipe cluster_centroids here. The slow work
    # (embedding + translation, ~2 min) must happen with NO write lock held,
    # otherwise we'd block the running monitor for the whole duration. Each
    # algorithm acquires the lock only at its final persist step:
    #   - hdbscan:     cluster_hdbscan() does its own DELETE+INSERT at the end
    #   - incremental: we DELETE + let cluster_titles_batch upsert per batch
    t0 = time.time()
    emb_cache = None
    if algorithm == "hdbscan":
        # Embed the whole corpus ONCE; reuse for the density pass + refinement.
        log_fn(f"  [stage 1/4] embedding {len(items)} item con e5-large …")
        t_emb = time.time()
        emb_cache = embed_items(items, log_fn=log_fn, conn=conn)
        log_fn(f"  [stage 1/4] embedded {len(emb_cache)} items in "
               f"{time.time()-t_emb:.0f}s  (tot {time.time()-t0:.0f}s)")
        # Partition by Vinted category (when known) then HDBSCAN within each.
        log_fn(f"  [stage 2/4] cluster_partitioned + HDBSCAN …")
        t_cl = time.time()
        cluster_map = cluster_partitioned(
            items, conn=conn,
            min_cluster_size=min_cluster_size,
            log_fn=log_fn, emb_cache=emb_cache,
        )
        log_fn(f"  [stage 2/4] partitioning done in {time.time()-t_cl:.0f}s  "
               f"(tot {time.time()-t0:.0f}s)")
    else:
        # Incremental pass (legacy threshold-based) — wipe first, then upsert
        conn.execute("DELETE FROM cluster_centroids")
        cluster_map: dict[int, str] = {}
        batch_size = 200
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            m = cluster_titles_batch(chunk, conn=conn)
            cluster_map.update(m)
            conn.commit()   # release lock between batches
            elapsed = time.time() - t0
            log_fn(f"  {i + len(chunk)}/{len(items)} ({elapsed:.0f}s)")

    # Refinement: lexical-gated singleton rescue + price-band split, then
    # recomputed centroids. Big quality jump on top of the raw density pass.
    log_fn(f"  [stage 3/4] refine (rescue singleton + price-split Jenks2) …")
    t_rf = time.time()
    for it in items:
        it["cluster_key"] = cluster_map.get(it["id"], "unknown")
    cluster_map = refine_clusters(items, cluster_map, conn, log_fn=log_fn,
                                  emb_cache=emb_cache)
    log_fn(f"  [stage 3/4] refine done in {time.time()-t_rf:.0f}s  "
           f"(tot {time.time()-t0:.0f}s)")
    # Commit now so the write lock is RELEASED before stage 2 — keeps the
    # monitor / sold-check / image worker from hitting "database is locked".
    conn.commit()

    # STAGE 2 — image refine: split price-incoherent clusters by image into
    # visual sub-products. The regen NO LONGER downloads photos or loads CLIP:
    # it consumes ONLY the image_emb already computed by the continuous image
    # worker (which keeps images "in pari" on its own and is paused while we
    # regenerate). Big win: the regen is faster and never loads CLIP → far less
    # RAM. Gated → never touches clean clusters; works fine with partial image
    # coverage (only items that already have an embedding take part).
    if algorithm == "hdbscan":
        log_fn(f"  [stage 4/4] image refine …")
        t_im = time.time()
        img_of: dict = {}
        for it in items:
            blob = it.get("image_emb")
            if blob is not None:
                try:
                    img_of[it["id"]] = _blob_to_emb(blob)
                except Exception:
                    pass
        log_fn(f"  immagini pronte dal worker: {len(img_of)}/{len(items)} item "
               f"(il regen non scarica foto)")
        if img_of:
            for it in items:
                it["cluster_key"] = cluster_map.get(it["id"], "unknown")
            cluster_map = image_refine(cluster_map, items, img_of, log_fn=log_fn)
        log_fn(f"  [stage 4/4] image refine done in {time.time()-t_im:.0f}s  "
               f"(tot {time.time()-t0:.0f}s)")

    log_fn(f"  scrivo cluster_key sul DB …")
    conn.executemany(
        "UPDATE items SET cluster_key = ? WHERE id = ?",
        [(cluster_map.get(it["id"], "unknown"), it["id"]) for it in items],
    )

    # INLINE EVAL — same metrics as scripts/eval_db.py, but reusing emb_cache.
    # Saves the ~8-min re-embedding pass; result printed right after regen so
    # the operator sees SCORE without running a separate command.
    if algorithm == "hdbscan" and emb_cache:
        from core.clustering import eval_metrics
        # Build the active-only mapping that eval_db.py uses (removed=0).
        # `items` already comes from "WHERE title IS NOT NULL" but includes
        # removed=1 → filter for parity with the canonical eval.
        active = [it for it in items if not it.get("removed")]
        m = eval_metrics(active, cluster_map, emb_cache)
        log_fn("")
        log_fn(f"  📊 EVAL inline (no re-embedding):")
        log_fn(f"     items: {m['items']}  cluster: {m['cluster']}  "
               f"singleton: {m['singleton_pct']:.0f}%  coverage: {m['coverage']:.0f}%")
        log_fn(f"     cohesion: {m['cohesion']:.3f}  price-coh: {m['price_coh']:.0f}%  "
               f"brand-pure: {m['brand_pure']:.0f}%  → SCORE: {m['score']:.1f}")
        # Extended metrics (anomaly-ready KPI for downstream deal detector)
        log_fn(f"     N_IDEAL: {m['n_ideal']}  ({m['ideal_items']} items)   "
               f"N_BLOB: {m['n_blob']}  ({m['blob_items']} items)   "
               f"N_GIANT: {m['n_giant']}")
        log_fn(f"     ANOM_READY: {m['anom_ready']:.1f}%   QSCORE: {m['qscore']:.2f}")
        log_fn("")

    n_clusters = conn.execute(
        "SELECT COUNT(*) FROM cluster_centroids"
    ).fetchone()[0]
    n_translated = conn.execute(
        "SELECT COUNT(*) FROM cluster_centroids WHERE display_label IS NOT NULL"
    ).fetchone()[0]
    conn.commit()
    # Explicit checkpoint(TRUNCATE) collapses the WAL back to 0 bytes — keeps
    # disk usage in check and speeds up subsequent readers.
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    conn.close()

    # Resume background workers — remove the pause flag we set at the start.
    try:
        if pause_flag.exists():
            pause_flag.unlink()
            log_fn("  ▶ background workers ripristinati (flag rimosso)")
    except Exception:
        pass

    log_fn(f"  done: {n_clusters} clusters, {n_translated} with English label "
           f"({100*n_translated/max(1,n_clusters):.0f}%)  "
           f"[{time.time()-t0:.0f}s]")
    return {"items": len(items), "clusters": n_clusters,
            "translated": n_translated}


def main():
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers required", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("db", nargs="?", help="Path to a category_*.db")
    parser.add_argument("--all", action="store_true",
                        help="Process every data/category_*.db file")
    args = parser.parse_args()

    if args.all:
        targets = sorted((ROOT / "data").glob("category_*.db"))
    elif args.db:
        targets = [Path(args.db)]
    else:
        parser.error("specify a DB path or --all")

    if not targets:
        print("no DBs found", file=sys.stderr)
        sys.exit(1)

    for db in targets:
        if not db.exists():
            print(f"skip (not found): {db}", file=sys.stderr)
            continue
        regenerate(db)


if __name__ == "__main__":
    main()
