"""Background image-embedding worker.

A standalone process (separate from the monitor and the regenerate) that
continuously turns item photos into vectors:

  loop:
    pick a batch of items that have photo_urls but no image_emb yet
        (freshest first → their CDN URLs are still valid; skip ones tried
         recently so an expired URL isn't retried in a tight loop)
    download each item's photos IN RAM → CLIP multi-photo mean → store image_emb
        (the photos are never written to disk; only the ~vector is kept)
    mark image_tried_at so failures aren't hammered

Keeping this in its OWN process means the always-on monitor never loads the
heavy CLIP model, and the regenerate finds image_emb already computed (fast).

Run:  python -m core.image_worker            # continuous
      python -m core.image_worker --once     # one pass then exit
"""
from __future__ import annotations

import argparse
import sqlite3
import time

from .category_db import shared_db_path, connect
from .clustering import (
    embed_item_images, _emb_to_blob, EMBEDDINGS_AVAILABLE, IMAGE_EMB_VERSION,
)

BATCH        = 40       # items per pass
IDLE_SLEEP   = 60       # seconds to wait when there's nothing to do
PASS_SLEEP   = 3        # small pace between busy passes
RETRY_HOURS  = 12       # don't re-attempt a failed (stale-URL) item for this long


def _todo(conn, batch: int) -> list[dict]:
    """Pick items needing an image embedding: either never tried, OR cached
    with a stale model version (= constant bumped on model change)."""
    rows = conn.execute(
        """
        SELECT id, photo_urls, image_emb FROM items
         WHERE photo_urls IS NOT NULL AND removed = 0
           AND (image_emb IS NULL OR image_emb_version IS NULL
                OR image_emb_version != ?)
           AND (image_tried_at IS NULL
                OR image_tried_at < datetime('now', ?))
         ORDER BY last_seen_at DESC
         LIMIT ?
        """,
        (IMAGE_EMB_VERSION, f"-{RETRY_HOURS} hours", batch),
    ).fetchall()
    return [dict(r) for r in rows]


def run(once: bool = False, db_path=None) -> None:
    if not EMBEDDINGS_AVAILABLE:
        print("image_worker: embeddings non disponibili — esco", flush=True)
        return
    from pathlib import Path
    db = Path(db_path) if db_path else shared_db_path()
    print(f"image_worker — started · DB: {db.name}", flush=True)
    while True:
        with connect(db) as conn:
            items = _todo(conn, BATCH)
        if not items:
            if once:
                print("image_worker: niente da fare", flush=True)
                return
            time.sleep(IDLE_SLEEP)
            continue
        t0 = time.time()
        vecs = embed_item_images(items, log_fn=lambda *a: None)   # download+CLIP, in RAM
        now_ids = [it["id"] for it in items]
        try:
            with connect(db) as conn:
                if vecs:
                    conn.executemany(
                        "UPDATE items SET image_emb = ?, image_emb_version = ?, "
                        "image_tried_at = datetime('now') WHERE id = ?",
                        [(_emb_to_blob(v), IMAGE_EMB_VERSION, iid)
                         for iid, v in vecs.items()],
                    )
                # mark every attempted item (incl. failures) so we don't tight-loop
                conn.executemany(
                    "UPDATE items SET image_tried_at = datetime('now') WHERE id = ? AND image_emb IS NULL",
                    [(iid,) for iid in now_ids if iid not in vecs],
                )
                conn.commit()
            print(f"image_worker: {len(vecs)}/{len(items)} vettorizzati ({time.time()-t0:.0f}s)", flush=True)
        except sqlite3.OperationalError as e:
            # transient lock (e.g. a regenerate writing) → survive, retry next pass
            print(f"image_worker: DB occupato ({e}); riprovo dopo", flush=True)
            time.sleep(IDLE_SLEEP)
            continue
        if once:
            return
        time.sleep(PASS_SLEEP)


def main():
    ap = argparse.ArgumentParser(description="Vinted image-embedding worker")
    ap.add_argument("--once", action="store_true", help="one pass then exit")
    ap.add_argument("--db", default=None, help="override DB path (testing)")
    args = ap.parse_args()
    try:
        run(once=args.once, db_path=args.db)
    except KeyboardInterrupt:
        print("\nimage_worker fermato.")


if __name__ == "__main__":
    main()
