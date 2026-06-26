"""Background text-embedding worker.

A standalone process (separate from the monitor and the regenerate) that
continuously turns item titles into vectors:

  loop:
    pick a batch of items whose `text_emb` is missing OR cached with a stale
        model version (= TEXT_EMB_VERSION constant in clustering.py, bumped
        whenever the model or `_embed_text` logic changes)
    encode them with the SAME `_embed_text(item)` pipeline used elsewhere
        (so vectors line up with centroids and cluster_partitioned)
    store `text_emb` blob + `text_emb_version` in items

Symmetric to `image_worker` — both decouple the heavy embedding from the
regenerate (which now reads cache from the DB and only re-encodes the few
items that arrived since the last worker pass).

Run:  python -m core.text_worker            # continuous
      python -m core.text_worker --once     # one pass then exit
"""
from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np

from .category_db import shared_db_path, connect
from .clustering import (
    _embed_text, _get_model, _emb_to_blob,
    EMBEDDINGS_AVAILABLE, TEXT_EMB_VERSION,
)

BATCH        = 256      # items per pass (e5-large encodes ~67 item/s on CPU)
IDLE_SLEEP   = 60       # seconds when nothing to do
PASS_SLEEP   = 1        # tiny pace between busy passes


def _todo(conn, batch: int) -> list[dict]:
    """Items needing a text embedding: never embedded, or stale version."""
    rows = conn.execute(
        """
        SELECT id, title, brand FROM items
         WHERE title IS NOT NULL AND removed = 0
           AND (text_emb IS NULL OR text_emb_version IS NULL
                OR text_emb_version != ?)
           AND (text_tried_at IS NULL
                OR text_tried_at < datetime('now', '-1 hours'))
         ORDER BY last_seen_at DESC
         LIMIT ?
        """,
        (TEXT_EMB_VERSION, batch),
    ).fetchall()
    return [dict(r) for r in rows]


def run(once: bool = False, db_path=None) -> None:
    if not EMBEDDINGS_AVAILABLE:
        print("text_worker: embeddings non disponibili — esco", flush=True)
        return
    from pathlib import Path
    db = Path(db_path) if db_path else shared_db_path()
    print(f"text_worker — started · DB: {db.name} · version: {TEXT_EMB_VERSION}", flush=True)
    model = _get_model()
    while True:
        with connect(db) as conn:
            items = _todo(conn, BATCH)
        if not items:
            if once:
                print("text_worker: niente da fare", flush=True)
                return
            time.sleep(IDLE_SLEEP)
            continue
        t0 = time.time()
        texts = [_embed_text(it) for it in items]
        embs = model.encode(texts, show_progress_bar=False,
                            convert_to_numpy=True, batch_size=256)
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        try:
            with connect(db) as conn:
                conn.executemany(
                    "UPDATE items SET text_emb = ?, text_emb_version = ?, "
                    "text_tried_at = datetime('now') WHERE id = ?",
                    [(_emb_to_blob(e.astype("float32")), TEXT_EMB_VERSION, it["id"])
                     for it, e in zip(items, embs)],
                )
                conn.commit()
            dt = time.time() - t0
            print(f"text_worker: {len(items)} vettorizzati ({dt:.1f}s, "
                  f"{len(items)/max(dt,1e-6):.0f} item/s)", flush=True)
        except sqlite3.OperationalError as e:
            print(f"text_worker: DB occupato ({e}); riprovo dopo", flush=True)
            time.sleep(IDLE_SLEEP)
            continue
        if once:
            return
        time.sleep(PASS_SLEEP)


def main():
    ap = argparse.ArgumentParser(description="Vinted text-embedding worker")
    ap.add_argument("--once", action="store_true", help="one pass then exit")
    ap.add_argument("--db", default=None, help="override DB path (testing)")
    args = ap.parse_args()
    try:
        run(once=args.once, db_path=args.db)
    except KeyboardInterrupt:
        print("\ntext_worker fermato.")


if __name__ == "__main__":
    main()
