"""
SQLite persistence layer for category monitors.

Category monitors scrape entire Vinted catalog IDs and accumulate item-level
data over time (weeks/months) — JSON files don't scale to that volume, so we
use SQLite. Schema:

  items
    id                    INTEGER PRIMARY KEY     # Vinted item ID
    title                 TEXT NOT NULL
    brand                 TEXT
    price                 REAL                    # current/last observed listing price
    last_price            REAL                    # previous price (for drop comparison)
    favourite_count       INTEGER DEFAULT 0
    cluster_key           TEXT
    first_seen_at         TEXT NOT NULL           # ISO-like "YYYY-MM-DD HH:MM:SS"
    last_seen_at          TEXT NOT NULL
    seen_count            INTEGER DEFAULT 1       # how many scrape cycles observed it
    presumed_sold         INTEGER DEFAULT 0       # 0/1 flag
    presumed_sold_at      TEXT
    presumed_sold_price   REAL                    # last price observed before disappearing

  price_drops
    id                    INTEGER PRIMARY KEY AUTOINCREMENT
    item_id               INTEGER NOT NULL REFERENCES items(id)
    old_price             REAL NOT NULL
    new_price             REAL NOT NULL
    drop_pct              INTEGER NOT NULL
    detected_at           TEXT NOT NULL
"""

from __future__ import annotations

import datetime
import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id                  INTEGER PRIMARY KEY,
    title               TEXT NOT NULL,
    brand               TEXT,
    price               REAL,
    last_price          REAL,
    favourite_count     INTEGER DEFAULT 0,
    cluster_key         TEXT,
    first_seen_at       TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    seen_count          INTEGER DEFAULT 1,
    presumed_sold       INTEGER DEFAULT 0,
    presumed_sold_at    TEXT,
    presumed_sold_price REAL,
    last_checked_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_cluster      ON items(cluster_key);
CREATE INDEX IF NOT EXISTS idx_items_sold         ON items(presumed_sold);
CREATE INDEX IF NOT EXISTS idx_items_last_seen    ON items(last_seen_at);
-- idx_items_last_checked is created AFTER _migrate() so it works on legacy
-- DBs where the column is added via ALTER TABLE.

CREATE TABLE IF NOT EXISTS price_drops (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER NOT NULL,
    old_price   REAL NOT NULL,
    new_price   REAL NOT NULL,
    drop_pct    INTEGER NOT NULL,
    detected_at TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
CREATE INDEX IF NOT EXISTS idx_drops_item ON price_drops(item_id);
CREATE INDEX IF NOT EXISTS idx_drops_at   ON price_drops(detected_at);

-- One row per day: lets us plot 14-day medians / counts without recomputing.
-- Written by record_snapshot() at the end of each monitor cycle; we keep one
-- row per (date) so multiple cycles within the same day just overwrite the
-- snapshot with the latest numbers.
CREATE TABLE IF NOT EXISTS daily_snapshots (
    date          TEXT PRIMARY KEY,
    total         INTEGER NOT NULL,
    active        INTEGER NOT NULL,
    sold          INTEGER NOT NULL,
    median_listed REAL,
    median_sold   REAL,
    drops_today   INTEGER DEFAULT 0
);

-- Persistent centroids for incremental embedding clustering. Each row is one
-- cluster: the embedding is a numpy float32 array serialized to BLOB. The
-- moving-average update means cluster_id ("cluster_key") stays STABLE across
-- monitor cycles — without this, every batch DBSCAN run would regenerate
-- arbitrary cluster IDs and break the items.cluster_key foreign relation.
CREATE TABLE IF NOT EXISTS cluster_centroids (
    cluster_key       TEXT PRIMARY KEY,
    embedding         BLOB NOT NULL,
    item_count        INTEGER DEFAULT 1,
    last_updated_at   TEXT,
    display_label     TEXT
);
"""


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply ALTER TABLE statements for columns that may be missing on older
    DBs. CREATE TABLE IF NOT EXISTS doesn't add columns to existing tables,
    so any new column added to the SCHEMA needs a matching line here.
    Indexes that depend on migrated columns are created here too — they
    can't live in SCHEMA because they'd fire BEFORE the ALTER TABLE on
    legacy DBs and crash with "no such column".
    Every statement is wrapped in try/except so re-runs are idempotent."""
    migrations = [
        "ALTER TABLE items ADD COLUMN last_checked_at TEXT",
        "CREATE INDEX IF NOT EXISTS idx_items_last_checked ON items(last_checked_at)",
        "ALTER TABLE cluster_centroids ADD COLUMN display_label TEXT",
        # Snapshot of favourite_count at the moment we marked the item sold —
        # the live items.favourite_count keeps drifting as the feed re-scrapes
        # active items, so we need a frozen value for sold-time analytics.
        "ALTER TABLE items ADD COLUMN presumed_sold_favourites INTEGER",
        # Vinted leaf category, extracted from the item HTML during sold-check
        # (the catalog browse doesn't return it). Used to enrich the embedding
        # text. catalog_id = Vinted's taxonomy id, category = English slug name.
        "ALTER TABLE items ADD COLUMN catalog_id INTEGER",
        "ALTER TABLE items ADD COLUMN category TEXT",
        # "removed" = listing 404'd / disappeared (seller deleted/withdrew it).
        # Distinct from presumed_sold: a SOLD item keeps a 200 page with a
        # "Venduto" banner; only a 404 means removed. Removed items are excluded
        # from active/sold counts and medians (verified 2026-05-22).
        "ALTER TABLE items ADD COLUMN removed INTEGER DEFAULT 0",
        "ALTER TABLE items ADD COLUMN removed_at TEXT",
        "CREATE INDEX IF NOT EXISTS idx_items_removed ON items(removed)",
        # Image clustering (multimodal two-stage): photo_urls are captured cheaply
        # from the feed at scrape time (comma-joined, up to a few); image_emb is
        # the L-14 multi-photo mean vector, computed once in the regenerate
        # subprocess and cached here (so the always-on monitor never loads CLIP).
        "ALTER TABLE items ADD COLUMN photo_urls TEXT",
        "ALTER TABLE items ADD COLUMN image_emb BLOB",
        # When the image worker last attempted to embed this item (success OR
        # fail) — so a stale/expired URL isn't retried in a tight loop.
        "ALTER TABLE items ADD COLUMN image_tried_at TEXT",
        "CREATE INDEX IF NOT EXISTS idx_items_image_todo ON items(image_emb, removed)",
        # Composite indexes for the GUI's median queries — without them, the
        # median LISTED + median SOLD lookups in stats_summary do a full table
        # sort (TEMP B-TREE FOR ORDER BY) → ~17s on 100k rows → WebSocket
        # keepalive times out → browser shows "connection lost" even with
        # working wi-fi. With these the planner can range-scan an already-
        # sorted index → drops to <10ms.
        "CREATE INDEX IF NOT EXISTS idx_items_listed_price "
        "ON items(presumed_sold, removed, price)",
        "CREATE INDEX IF NOT EXISTS idx_items_sold_price "
        "ON items(presumed_sold, removed, presumed_sold_price)",
        # Covers the GROUP BY cluster_key in the dashboard: lets SQLite walk
        # the index already grouped by cluster_key (no TEMP B-TREE for GROUP).
        # Includes presumed_sold so the SUM(CASE WHEN presumed_sold=0...)
        # columns are computed from the index alone (covering index).
        "CREATE INDEX IF NOT EXISTS idx_items_cluster_grp "
        "ON items(removed, cluster_key, presumed_sold, price, presumed_sold_price)",
        # Title embedding cache (pre-computed by text_worker continuously).
        # Symmetric to image_emb. *_version lets us bump a constant to
        # invalidate the cache when the model or _embed_text logic changes —
        # the workers then re-embed gradually in background. Without versioning,
        # mixing vectors from different models silently breaks clustering.
        "ALTER TABLE items ADD COLUMN text_emb BLOB",
        "ALTER TABLE items ADD COLUMN text_emb_version TEXT",
        "ALTER TABLE items ADD COLUMN text_tried_at TEXT",
        "ALTER TABLE items ADD COLUMN image_emb_version TEXT",
        "CREATE INDEX IF NOT EXISTS idx_items_text_todo "
        "ON items(text_emb_version, removed)",
    ]
    for stmt in migrations:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            # "duplicate column name" — already migrated
            pass


@contextmanager
def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL mode lets readers and one writer coexist without blocking each
    # other — needed because the monitor cycles and the recheck_sold script
    # may run concurrently. busy_timeout makes brief contention transparent.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Shared single DB ─────────────────────────────────────────────────────────
# All category monitors read/write ONE shared DB. Each category is a FILTERED
# VIEW of it (by its leaf catalog_ids): the running collector sold-checks only
# its category's items, and the dashboard shows only its category's items.

_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "data"
SHARED_DB = _DATA_DIR / "category_shared.db"


def shared_db_path() -> Path:
    """Return the shared DB path. One-time: if it doesn't exist yet, seed it from
    the largest existing category_*.db (a proper sqlite backup, WAL-safe) so the
    accumulated history — items, sold/removed, clusters — carries over."""
    if not SHARED_DB.exists():
        candidates = [p for p in _DATA_DIR.glob("category_*.db") if p != SHARED_DB]
        if candidates:
            seed = max(candidates, key=lambda p: p.stat().st_size)
            try:
                _DATA_DIR.mkdir(parents=True, exist_ok=True)
                src = sqlite3.connect(seed)
                dst = sqlite3.connect(SHARED_DB)
                with dst:
                    src.backup(dst)
                src.close()
                dst.close()
            except Exception:
                pass
    return SHARED_DB


def _cat_filter(catalog_ids, alias: str = "") -> tuple[str, list]:
    """Build an optional ' AND <alias>catalog_id IN (?, ...)' fragment + args.
    Empty/None catalog_ids → no filter (the broad 'everything' view)."""
    if not catalog_ids:
        return "", []
    col = f"{alias}catalog_id" if alias else "catalog_id"
    ph = ",".join("?" * len(catalog_ids))
    return f" AND {col} IN ({ph})", list(catalog_ids)


def upsert_items(
    conn: sqlite3.Connection,
    items: list[dict],
    drop_threshold_pct: float,
    drop_min_abs: float,
) -> tuple[int, int, list[dict]]:
    """
    Insert new items and update existing ones with the latest observation.

    For each item that already exists with a known price, compare current
    vs. stored price; emit a price-drop event when both thresholds are met.

    Returns (n_new, n_updated, drops).
    """
    now    = _now()
    n_new  = 0
    n_upd  = 0
    drops: list[dict] = []

    for it in items:
        row = conn.execute(
            "SELECT id, price, last_price, seen_count FROM items WHERE id = ?",
            (it["id"],),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO items (id, title, brand, price, last_price,
                                   favourite_count, cluster_key, catalog_id,
                                   photo_urls,
                                   first_seen_at, last_seen_at, seen_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (it["id"], it["title"], it.get("brand"), it["price"],
                 it["price"], it.get("favourite_count", 0),
                 it.get("cluster_key"), it.get("catalog_id"),
                 it.get("photo_urls"), now, now),
            )
            n_new += 1
            continue

        prev_price = row["price"] or 0
        new_price  = it["price"] or 0

        # Possible drop event
        if (prev_price > 0 and new_price > 0
                and drop_threshold_pct > 0):
            drop_abs = prev_price - new_price
            drop_pct = (drop_abs / prev_price) * 100
            if drop_pct >= drop_threshold_pct and drop_abs >= drop_min_abs:
                drop_evt = {
                    "item_id":   it["id"],
                    "title":     it["title"],
                    "old_price": prev_price,
                    "new_price": new_price,
                    "drop_pct":  round(drop_pct),
                    "url":       it.get("url", ""),
                }
                drops.append(drop_evt)
                conn.execute(
                    """
                    INSERT INTO price_drops (item_id, old_price, new_price,
                                             drop_pct, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (it["id"], prev_price, new_price, drop_evt["drop_pct"], now),
                )

        conn.execute(
            """
            UPDATE items
               SET title = ?, brand = ?, price = ?, last_price = ?,
                   favourite_count = ?, cluster_key = ?,
                   catalog_id = COALESCE(?, catalog_id),
                   photo_urls = COALESCE(?, photo_urls),
                   last_seen_at = ?, seen_count = seen_count + 1,
                   presumed_sold = 0, presumed_sold_at = NULL,
                   presumed_sold_price = NULL,
                   removed = 0, removed_at = NULL
             WHERE id = ?
            """,
            (it["title"], it.get("brand"), new_price, prev_price,
             it.get("favourite_count", 0), it.get("cluster_key"),
             it.get("catalog_id"), it.get("photo_urls"), now, it["id"]),
        )
        n_upd += 1

    return n_new, n_upd, drops


def mark_presumed_sold(
    conn: sqlite3.Connection,
    not_seen_for_seconds: int,
    min_seen_count: int,
) -> int:
    """
    Mark items that haven't been observed for `not_seen_for_seconds` AND were
    "stable" (seen at least `min_seen_count` times) as presumed_sold.
    Their last observed price becomes presumed_sold_price.
    Returns how many items were newly flagged.
    """
    now    = _now()
    cutoff = (datetime.datetime.now()
              - datetime.timedelta(seconds=not_seen_for_seconds)
              ).strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        """
        UPDATE items
           SET presumed_sold = 1,
               presumed_sold_at = ?,
               presumed_sold_price = price
         WHERE presumed_sold = 0
           AND seen_count >= ?
           AND last_seen_at < ?
        """,
        (now, min_seen_count, cutoff),
    )
    return cur.rowcount


def record_snapshot(conn: sqlite3.Connection) -> None:
    """Upsert today's row in daily_snapshots with current aggregate stats."""
    today = datetime.date.today().isoformat()
    s = stats_summary(conn)
    drops_today = conn.execute(
        "SELECT COUNT(*) FROM price_drops WHERE substr(detected_at,1,10) = ?",
        (today,),
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO daily_snapshots (date, total, active, sold,
                                     median_listed, median_sold, drops_today)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            total         = excluded.total,
            active        = excluded.active,
            sold          = excluded.sold,
            median_listed = excluded.median_listed,
            median_sold   = excluded.median_sold,
            drops_today   = excluded.drops_today
        """,
        (today, s["total"], s["active"], s["sold"],
         s["median_listed"], s["median_sold"], drops_today),
    )


def load_snapshots(conn: sqlite3.Connection, days: int = 14) -> list[dict]:
    """Return the last `days` snapshots (most recent last)."""
    rows = conn.execute(
        """
        SELECT date, total, active, sold, median_listed, median_sold, drops_today
          FROM daily_snapshots
         ORDER BY date DESC
         LIMIT ?
        """,
        (days,),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def recent_sold(conn: sqlite3.Connection, limit: int = 30,
                catalog_ids=None) -> list[dict]:
    """Return items recently flagged as presumed_sold (optionally scoped to a
    category's leaf catalog_ids)."""
    cf, ca = _cat_filter(catalog_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, brand, presumed_sold_price AS price,
               presumed_sold_at AS sold_at, cluster_key, favourite_count
          FROM items
         WHERE presumed_sold = 1 AND removed = 0 AND presumed_sold_price > 0{cf}
         ORDER BY presumed_sold_at DESC
         LIMIT ?
        """,
        ca + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def price_distribution(conn: sqlite3.Connection, bins: int = 12,
                       max_price_cap: float = 2000,
                       cluster_key: str | None = None,
                       catalog_ids=None
                       ) -> list[tuple[float, float, int]]:
    """Return (lo, hi, count) histogram bins of active listing prices.
    If `cluster_key` is given, restrict to that cluster; `catalog_ids` scopes to
    a category's leaf sub-categories."""
    sql = "SELECT price FROM items WHERE presumed_sold = 0 AND removed = 0 AND price > 0 AND price <= ?"
    args: list = [max_price_cap]
    if cluster_key:
        sql += " AND cluster_key = ?"
        args.append(cluster_key)
    cf, ca = _cat_filter(catalog_ids)
    sql += cf
    args += ca
    prices = [r[0] for r in conn.execute(sql, args)]
    if not prices:
        return []
    lo_p = min(prices)
    hi_p = max(prices)
    if hi_p <= lo_p:
        return [(lo_p, hi_p, len(prices))]
    width = (hi_p - lo_p) / bins
    edges = [lo_p + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for p in prices:
        idx = min(int((p - lo_p) / width), bins - 1)
        counts[idx] += 1
    return [(edges[i], edges[i + 1], counts[i]) for i in range(bins)]


def stats_for_cluster(conn: sqlite3.Connection, cluster_key: str,
                      catalog_ids=None) -> dict:
    """Per-cluster equivalent of stats_summary. Excludes removed; optionally
    scoped to a category's leaf catalog_ids."""
    cf, ca = _cat_filter(catalog_ids)
    total = conn.execute(
        f"SELECT COUNT(*) FROM items WHERE cluster_key = ? AND removed = 0{cf}",
        [cluster_key] + ca,
    ).fetchone()[0]
    sold = conn.execute(
        f"SELECT COUNT(*) FROM items WHERE cluster_key = ? AND presumed_sold = 1 AND removed = 0{cf}",
        [cluster_key] + ca,
    ).fetchone()[0]
    median_listed = conn.execute(
        f"""
        SELECT price FROM items
         WHERE cluster_key = ? AND presumed_sold = 0 AND removed = 0 AND price > 0{cf}
         ORDER BY price
         LIMIT 1 OFFSET (
            SELECT COUNT(*) / 2 FROM items
             WHERE cluster_key = ? AND presumed_sold = 0 AND removed = 0 AND price > 0{cf}
         )
        """,
        [cluster_key] + ca + [cluster_key] + ca,
    ).fetchone()
    median_sold = conn.execute(
        f"""
        SELECT presumed_sold_price FROM items
         WHERE cluster_key = ? AND presumed_sold = 1 AND removed = 0 AND presumed_sold_price > 0{cf}
         ORDER BY presumed_sold_price
         LIMIT 1 OFFSET (
            SELECT COUNT(*) / 2 FROM items
             WHERE cluster_key = ? AND presumed_sold = 1 AND removed = 0 AND presumed_sold_price > 0{cf}
         )
        """,
        [cluster_key] + ca + [cluster_key] + ca,
    ).fetchone()
    return {
        "total":          total,
        "active":         total - sold,
        "sold":           sold,
        "median_listed":  median_listed[0] if median_listed else None,
        "median_sold":    median_sold[0]   if median_sold   else None,
    }


def sold_history_for_cluster(conn: sqlite3.Connection, cluster_key: str,
                             limit: int = 60, catalog_ids=None) -> list[dict]:
    """Return chronological list of sold items in a cluster:
       [{date, price}, ...] ordered by sold_at ascending."""
    cf, ca = _cat_filter(catalog_ids)
    rows = conn.execute(
        f"""
        SELECT substr(presumed_sold_at, 1, 10) AS date,
               presumed_sold_price             AS price,
               title, id
          FROM items
         WHERE cluster_key = ? AND presumed_sold = 1 AND removed = 0
               AND presumed_sold_price > 0{cf}
         ORDER BY presumed_sold_at ASC
         LIMIT ?
        """,
        [cluster_key] + ca + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def cluster_drops(conn: sqlite3.Connection, cluster_key: str,
                  limit: int = 30, catalog_ids=None) -> list[dict]:
    """Recent price drops restricted to one cluster."""
    cf, ca = _cat_filter(catalog_ids, alias="i.")
    rows = conn.execute(
        f"""
        SELECT d.old_price, d.new_price, d.drop_pct, d.detected_at,
               i.title, i.id AS item_id
          FROM price_drops d
          JOIN items i ON i.id = d.item_id
         WHERE i.cluster_key = ?{cf}
         ORDER BY d.detected_at DESC
         LIMIT ?
        """,
        [cluster_key] + ca + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def all_items_for_cluster(conn: sqlite3.Connection, cluster_key: str,
                          catalog_ids=None) -> list[dict]:
    """Return every item in a cluster, both active and sold, with the fields
    a UI needs to render a clickable list."""
    cf, ca = _cat_filter(catalog_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, brand,
               price, last_price, favourite_count,
               first_seen_at, last_seen_at, seen_count,
               presumed_sold, presumed_sold_at, presumed_sold_price
          FROM items
         WHERE cluster_key = ? AND removed = 0{cf}
         ORDER BY presumed_sold ASC, price DESC
        """,
        [cluster_key] + ca,
    ).fetchall()
    return [dict(r) for r in rows]


def cluster_sold(conn: sqlite3.Connection, cluster_key: str,
                 limit: int = 30, catalog_ids=None) -> list[dict]:
    """Recent presumed-sold items restricted to one cluster."""
    cf, ca = _cat_filter(catalog_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, brand, presumed_sold_price AS price,
               presumed_sold_at AS sold_at, cluster_key, favourite_count
          FROM items
         WHERE cluster_key = ? AND presumed_sold = 1 AND removed = 0 AND presumed_sold_price > 0{cf}
         ORDER BY presumed_sold_at DESC
         LIMIT ?
        """,
        [cluster_key] + ca + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def search_items_by_title(conn: sqlite3.Connection, query: str,
                          catalog_ids=None, limit: int = 100) -> list[dict]:
    """Item-level full-text search by title substring (LIKE %q%). Returns
    active (removed=0) items only, capped at `limit` to keep the popup
    responsive. Used by the GUI search bar alongside the cluster filter.
    Splits the query on whitespace and AND-joins LIKE clauses so multi-word
    searches narrow down (e.g. "iphone 13 pro 256" matches title containing
    all four tokens in any order)."""
    q = (query or "").strip()
    if not q or len(q) < 2:
        return []
    cf, ca = _cat_filter(catalog_ids)
    tokens = [t for t in q.split() if t]
    if not tokens:
        return []
    likes_sql = " AND ".join(["LOWER(title) LIKE ?"] * len(tokens))
    likes_params = [f"%{t.lower()}%" for t in tokens]
    rows = conn.execute(
        f"""
        SELECT id, title, brand, price, last_price, presumed_sold,
               presumed_sold_price, favourite_count, cluster_key,
               last_seen_at
          FROM items
         WHERE removed = 0 AND title IS NOT NULL
           AND {likes_sql}{cf}
         ORDER BY presumed_sold ASC, favourite_count DESC, last_seen_at DESC
         LIMIT ?
        """,
        likes_params + ca + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


_ITEMS_TO_CHECK_ORDER = {
    # Default: oldest-checked first (NULL = never checked = oldest) → guarantees
    # eventual full coverage of the scoped set.
    "oldest":     "(last_checked_at IS NULL) DESC, last_checked_at ASC, id ASC",
    # Most-watched first (then oldest among equals) → focuses the limited budget
    # on items likely to sell / be deals, re-checking them often.
    "favourites": "favourite_count DESC, (last_checked_at IS NULL) DESC, last_checked_at ASC, id ASC",
    # Highest price first.
    "price":      "price DESC, (last_checked_at IS NULL) DESC, last_checked_at ASC, id ASC",
    # Photo backfill: pick items WITHOUT photo_urls first (legacy items scraped
    # before the photo-capture feature). Self-disabling: once every item has
    # photos the first sort key is uniform → falls back to oldest-checked.
    "no_photo_first":
        "(photo_urls IS NULL) DESC, (last_checked_at IS NULL) DESC, last_checked_at ASC, id ASC",
}


def items_to_check(conn: sqlite3.Connection, limit: int,
                   min_seen_count: int = 3,
                   catalog_ids: list[int] | None = None,
                   order: str = "oldest") -> list[dict]:
    """Pick up to `limit` active items to re-validate via direct HTTP.

    `catalog_ids`: restrict to items belonging to these leaf catalog ids (the
    collector scopes the sold-check to the selected category only). None = all.
    `order`: 'oldest' (default) | 'favourites' | 'price' — see _ITEMS_TO_CHECK_ORDER.

    Only items seen `min_seen_count` times are eligible (skip one-off sightings)."""
    where = "presumed_sold = 0 AND removed = 0 AND seen_count >= ?"
    args: list = [min_seen_count]
    if catalog_ids:
        where += " AND catalog_id IN (%s)" % ",".join("?" * len(catalog_ids))
        args.extend(catalog_ids)
    order_sql = _ITEMS_TO_CHECK_ORDER.get(order, _ITEMS_TO_CHECK_ORDER["oldest"])
    args.append(limit)
    rows = conn.execute(
        f"SELECT id FROM items WHERE {where} ORDER BY {order_sql} LIMIT ?",
        args,
    ).fetchall()
    return [dict(r) for r in rows]


def record_http_check(
    conn: sqlite3.Connection,
    item_id: int,
    status: str,              # "active" | "sold" | "removed" | "unknown"
    price: float | None = None,
    drop_threshold_pct: float = 15,
    drop_min_abs: float = 5,
    catalog_id: int | None = None,
    category: str | None = None,
    favourites: int | None = None,
    photo_urls: str | None = None,
    brand: str | None = None,
) -> dict | None:
    """
    Persist the outcome of a direct HTTP check on an item, mirroring the
    drop-detection logic that runs inside `upsert_items` for the feed path.

    `catalog_id` / `category` (extracted from the item HTML) are stored
    whenever present, regardless of status — they enrich the embedding text
    and don't depend on sold/active state.

    Behaviour by status:
      "sold"    → flag presumed_sold, freeze favourite_count snapshot.
      "active"  → unflag presumed_sold, refresh last_seen_at. If `price` is
                  provided AND differs from the stored price, update both
                  items.price (new observation) and items.last_price (old).
                  When the change is a drop large enough to clear both
                  thresholds, also insert a price_drops row and return the
                  drop-event dict for caller logging.
      "unknown" → just touch last_checked_at, nothing else changes.

    Returns the drop-event dict when a price drop is recorded, otherwise None.
    """
    now = _now()

    # Store category info + fresh photo URLs whenever we have them (independent
    # of sold/active). Refreshing photo_urls here backfills OLD items (scraped
    # before photo capture) and keeps URLs fresh so the image worker can fetch
    # them — FREE, since the sold-check already downloaded this page. Reset
    # image_tried_at so the worker re-attempts with the fresh URLs.
    if catalog_id is not None or category is not None:
        conn.execute(
            "UPDATE items SET catalog_id = COALESCE(?, catalog_id), "
            "category = COALESCE(?, category) WHERE id = ?",
            (catalog_id, category, item_id),
        )
    if photo_urls:
        conn.execute(
            "UPDATE items SET photo_urls = ?, image_tried_at = NULL WHERE id = ?",
            (photo_urls, item_id),
        )
    # Brand backfill: if the sold-check pulls a NEW brand (currently the page
    # parser doesn't extract it, but this is the defensive hook for when it
    # will), update it AND invalidate the cached text embedding — the brand is
    # part of _embed_text input, so the old vector is stale. The text_worker
    # will re-encode the affected items on its next pass.
    if brand:
        cur = conn.execute("SELECT brand FROM items WHERE id = ?", (item_id,)).fetchone()
        if cur and (cur[0] or "").strip().lower() != brand.strip().lower():
            conn.execute(
                "UPDATE items SET brand = ?, text_emb_version = NULL WHERE id = ?",
                (brand, item_id),
            )

    if status == "sold":
        # SOLD = page still up (200 + "Venduto" banner). Clear `removed` in case
        # it had been flagged before.
        conn.execute(
            """
            UPDATE items
               SET presumed_sold              = 1,
                   removed                    = 0,
                   removed_at                 = NULL,
                   presumed_sold_at           = COALESCE(presumed_sold_at, ?),
                   presumed_sold_price        = COALESCE(presumed_sold_price, price),
                   presumed_sold_favourites   = COALESCE(presumed_sold_favourites, favourite_count),
                   last_checked_at            = ?
             WHERE id = ?
            """,
            (now, now, item_id),
        )
        return None

    if status == "removed":
        # REMOVED = 404 / gone (seller deleted or withdrew). NOT a sale — clear
        # presumed_sold so it leaves the sold stats and medians.
        conn.execute(
            """
            UPDATE items
               SET removed                    = 1,
                   removed_at                 = COALESCE(removed_at, ?),
                   presumed_sold              = 0,
                   presumed_sold_at           = NULL,
                   presumed_sold_price        = NULL,
                   presumed_sold_favourites   = NULL,
                   last_checked_at            = ?
             WHERE id = ?
            """,
            (now, now, item_id),
        )
        return None

    if status == "unknown":
        conn.execute(
            "UPDATE items SET last_checked_at = ? WHERE id = ?",
            (now, item_id),
        )
        return None

    # status == "active"
    row = conn.execute(
        "SELECT price, title, favourite_count FROM items WHERE id = ?",
        (item_id,),
    ).fetchone()
    prev_price = (row["price"] if row else None) or 0
    drop: dict | None = None

    if price is not None and prev_price > 0 and price != prev_price:
        # Price changed since last observation. Persist the new value plus
        # the old, so the feed path can still see last_price next time.
        if price < prev_price and drop_threshold_pct > 0:
            drop_abs = prev_price - price
            drop_pct = (drop_abs / prev_price) * 100
            if drop_pct >= drop_threshold_pct and drop_abs >= drop_min_abs:
                drop_pct_int = round(drop_pct)
                conn.execute(
                    """
                    INSERT INTO price_drops (item_id, old_price, new_price,
                                             drop_pct, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (item_id, prev_price, price, drop_pct_int, now),
                )
                drop = {
                    "item_id":   item_id,
                    "title":     row["title"] if row else "",
                    "old_price": prev_price,
                    "new_price": price,
                    "drop_pct":  drop_pct_int,
                    "source":    "http",
                }

        conn.execute(
            """
            UPDATE items
               SET presumed_sold              = 0,
                   removed                    = 0,
                   removed_at                 = NULL,
                   presumed_sold_at           = NULL,
                   presumed_sold_price        = NULL,
                   presumed_sold_favourites   = NULL,
                   favourite_count            = COALESCE(?, favourite_count),
                   price                      = ?,
                   last_price                 = ?,
                   last_seen_at               = ?,
                   last_checked_at            = ?
             WHERE id = ?
            """,
            (favourites, price, prev_price, now, now, item_id),
        )
        return drop

    # No price provided OR price unchanged — refresh activity + likes.
    conn.execute(
        """
        UPDATE items
           SET presumed_sold              = 0,
               removed                    = 0,
               removed_at                 = NULL,
               presumed_sold_at           = NULL,
               presumed_sold_price        = NULL,
               presumed_sold_favourites   = NULL,
               favourite_count            = COALESCE(?, favourite_count),
               last_seen_at               = ?,
               last_checked_at            = ?
         WHERE id = ?
        """,
        (favourites, now, now, item_id),
    )
    return None


def mark_item_checked(conn: sqlite3.Connection, item_id: int,
                      alive: bool | None) -> None:
    """Back-compat shim around `record_http_check`. Maps the old boolean API
    to the richer status enum, with no price (so no drop detection). New
    callers should use `record_http_check` directly to also pass the price."""
    status = {True: "active", False: "sold"}.get(alive, "unknown")
    record_http_check(conn, item_id, status)


def stats_summary(conn: sqlite3.Connection, catalog_ids=None) -> dict:
    """High-level snapshot for logs / GUI overview. Excludes removed (404'd)
    items; optionally scoped to a category's leaf catalog_ids (the per-category
    filtered view of the shared DB)."""
    cf, ca = _cat_filter(catalog_ids)
    total = conn.execute(
        f"SELECT COUNT(*) FROM items WHERE removed = 0{cf}", ca
    ).fetchone()[0]
    sold  = conn.execute(
        f"SELECT COUNT(*) FROM items WHERE presumed_sold = 1 AND removed = 0{cf}", ca
    ).fetchone()[0]
    active = total - sold
    median_listed = conn.execute(
        f"""
        SELECT price FROM items
         WHERE presumed_sold = 0 AND removed = 0 AND price IS NOT NULL AND price > 0{cf}
         ORDER BY price
         LIMIT 1 OFFSET (
            SELECT COUNT(*) / 2 FROM items
             WHERE presumed_sold = 0 AND removed = 0 AND price IS NOT NULL AND price > 0{cf}
         )
        """,
        ca + ca,
    ).fetchone()
    median_sold = conn.execute(
        f"""
        SELECT presumed_sold_price FROM items
         WHERE presumed_sold = 1 AND removed = 0 AND presumed_sold_price > 0{cf}
         ORDER BY presumed_sold_price
         LIMIT 1 OFFSET (
            SELECT COUNT(*) / 2 FROM items
             WHERE presumed_sold = 1 AND removed = 0 AND presumed_sold_price > 0{cf}
         )
        """,
        ca + ca,
    ).fetchone()
    return {
        "total":          total,
        "active":         active,
        "sold":           sold,
        "median_listed":  median_listed[0] if median_listed else None,
        "median_sold":    median_sold[0]   if median_sold   else None,
    }
