"""NiceGUI app for the Vinted Tracker — replaces the legacy Tkinter gui.py.

Run:  python -m webgui   (serves on http://localhost:8080)

Organised into: a shell (header + tabs + live log) and three refreshable views
(Ricerche / Esplora / Categorie). All backend work goes through webgui.services
and core/. State is module-level (single local user).
"""

from __future__ import annotations

import asyncio
from nicegui import ui, context


def _safe_timer(interval: float, callback, *, once: bool = False):
    """ui.timer wrapper that no-ops once the client (browser tab) is gone.

    Without this, recurring timers keep firing after the user navigates away
    or refreshes the page → NiceGUI logs "Client has been deleted but is
    still being used" warnings (see the stack trace from 2026-05-27). The
    guard checks the live client's socket and skips the callback when it's
    closed. Async callbacks are supported."""
    async def _aguard(*a, **kw):
        try:
            if not context.client.has_socket_connection:
                return
        except Exception:
            return
        r = callback(*a, **kw)
        if asyncio.iscoroutine(r):
            await r
    return ui.timer(interval, _aguard, once=once)

from core.category_db import (
    shared_db_path,
    stats_summary, price_distribution, recent_sold,
    all_items_for_cluster, connect as cat_connect,
    stats_for_cluster, sold_history_for_cluster,
    search_items_by_title,
)
from core.discovery import (
    CATEGORIES, DOMAINS_DEFAULT, discover, find_category_db, enrich_with_median,
)
from webgui import services as svc

mm = svc.MonitorManager()


class S:
    """Shared app state (single local user)."""
    monitors: list = []
    sel_text: str | None = None
    sel_cat: str | None = None
    cluster_filter: str | None = None
    net_online: bool = True


def reload_monitors():
    S.monitors = svc.load_monitors()
    if S.sel_text is None or not any(m["id"] == S.sel_text and m["type"] == "text" for m in S.monitors):
        txt = [m for m in S.monitors if m["type"] == "text"]
        S.sel_text = txt[0]["id"] if txt else None
    if S.sel_cat is None or not any(m["id"] == S.sel_cat and m["type"] == "category" for m in S.monitors):
        cat = [m for m in S.monitors if m["type"] == "category"]
        S.sel_cat = cat[0]["id"] if cat else None


def _mon(mid: str | None) -> dict | None:
    return next((m for m in S.monitors if m["id"] == mid), None)


reload_monitors()


# ── Sidebar (monitor list) ───────────────────────────────────────────────────

@ui.refreshable
def sidebar(kind: str):
    """List of monitors of `kind` ('text'|'category') with running dots."""
    mons = [m for m in S.monitors if m["type"] == kind]
    sel = S.sel_text if kind == "text" else S.sel_cat
    if not mons:
        ui.label("Nessun monitor.").classes("text-grey-5 p-2 text-sm")
        return
    for m in mons:
        running = mm.is_running(m["id"])
        active = (m["id"] == sel)

        def pick(mid=m["id"]):
            if kind == "text":
                S.sel_text = mid
            else:
                S.sel_cat = mid
                S.cluster_filter = None
            sidebar.refresh()
            (searches_detail if kind == "text" else category_detail).refresh()

        with ui.row().classes(
            "items-center w-full no-wrap cursor-pointer rounded px-2 py-1 gap-2 "
            + ("bg-blue-grey-9" if active else "hover:bg-grey-9")
        ).on("click", pick):
            ui.icon("circle", size="10px").classes(
                "text-green" if running else "text-grey-6")
            with ui.column().classes("gap-0 min-w-0"):
                ui.label(m["label"]).classes(
                    "text-sm truncate " + ("text-white" if active else "text-grey-3"))
                st = svc.read_stats(m["id"])
                ui.label(("● in corso" if running else "○ fermo")
                         + f" · {svc.time_ago(st.get('last_found_at'))}").classes(
                    "text-xs text-grey-6")


# ── Ricerche (text monitors) ─────────────────────────────────────────────────

def _stat_card(label, value, color="text-white"):
    with ui.card().classes("p-3 flex-1 min-w-[120px] bg-grey-9"):
        ui.label(label).classes("text-xs text-grey-5")
        ui.label(str(value)).classes(f"text-2xl font-bold {color}")


@ui.refreshable
def searches_detail():
    m = _mon(S.sel_text)
    if m is None:
        ui.label("Crea una ricerca per iniziare.").classes("text-grey-5 p-4")
        return
    cfg = svc.load_config(m["config_file"])
    running = mm.is_running(m["id"])
    st = svc.read_stats(m["id"])
    ps = svc.read_price_stats(m["id"])
    seen = svc.read_seen_count(cfg.get("seen_file", f"seen_{m['id']}_ids.json"))

    with ui.row().classes("items-center w-full"):
        ui.label(cfg.get("label", m["label"])).classes("text-xl font-bold")
        ui.badge("in corso" if running else "fermo",
                 color="green" if running else "grey-7").classes("ml-2")
        ui.space()
        if running:
            ui.button("■ Stop", on_click=lambda: (mm.stop(m["id"]), _soon())).props("color=red dense")
        else:
            ui.button("▶ Start", on_click=lambda: (mm.start(m), _soon())).props("color=green dense")
        ui.button("▶¹ Test", on_click=lambda: mm.test(m)).props("flat dense")
        ui.button("↻", on_click=lambda: (mm.restart(m), _soon())).props("flat dense")

    with ui.row().classes("w-full gap-2 mt-2"):
        _stat_card("NOTIFICATI", st.get("total_notified", 0), "text-green")
        _stat_card("SEEN IDS", seen)
        _stat_card("CAMPIONI PREZZO", len(ps.get("prices", [])), "text-purple-4")
        med = svc.median(ps.get("prices", []))
        _stat_card("MEDIANA MERCATO", f"€{int(med)}" if med else "—", "text-amber")

    with ui.row().classes("w-full gap-3 mt-2"):
        with ui.column().classes("flex-1 gap-1"):
            ui.label("FILTRI").classes("text-xs text-grey-5 mt-2")
            for lab, val in (
                ("Prezzo", f"€{cfg.get('min_price',0)}–€{cfg.get('max_price','?')}"),
                ("Intervallo", f"{cfg.get('check_interval_minutes','?')} min"),
                ("Keyword", ", ".join(cfg.get("keywords_required", [])) or "—"),
                ("Brand", ", ".join(cfg.get("brands_required", [])) or "qualsiasi"),
                ("Escludi", ", ".join(cfg.get("keywords_exclude", [])) or "—"),
                ("Min ♥", cfg.get("min_favourites", 0) or "no"),
                ("Anomalie", f"-{cfg['anomaly_threshold_pct']}%" if cfg.get("anomaly_threshold_pct") else "off"),
            ):
                with ui.row().classes("w-full gap-2"):
                    ui.label(lab).classes("text-xs text-grey-5 w-24")
                    ui.label(str(val)).classes("text-xs text-grey-2")

    ui.label(f"STORICO NOTIFICHE ({len(svc.read_history(m['id']))})").classes("text-xs text-grey-5 mt-3")
    hist = list(reversed(svc.read_history(m["id"])))
    cols = [
        {"name": "when", "label": "Quando", "field": "when", "sortable": True, "align": "left"},
        {"name": "price", "label": "€", "field": "price", "sortable": True},
        {"name": "fav", "label": "♥", "field": "fav", "sortable": True},
        {"name": "title", "label": "Titolo", "field": "title", "align": "left"},
    ]
    rows = []
    for it in hist[:300]:
        drop = it.get("event_type") == "price_drop"
        price = (f"€{int(it.get('old_price',0))}→€{int(it.get('price',0))} -{it.get('drop_pct',0)}%"
                 if drop else f"€{it.get('price','?')}")
        rows.append({"when": (it.get("notified_at", "")[:16] or "—"),
                     "price": price, "fav": it.get("favourite_count", 0) or "",
                     "title": ("📉 " if drop else "") + (it.get("title", "")),
                     "url": it.get("url", "")})
    tbl = ui.table(columns=cols, rows=rows, row_key="when", pagination=15).classes("w-full")
    with tbl.add_slot("top-right"):
        ui.input(placeholder="cerca…").props("dense clearable").bind_value(tbl, "filter")
    tbl.on("rowClick", lambda e: _open_url(e.args[1].get("url")))


# ── Categorie ────────────────────────────────────────────────────────────────

def _cat_dashboard_data(cfg):
    """Read the shared DB filtered to this category's catalog_ids."""
    view = cfg.get("soldcheck_catalog_ids") or None
    out = {"stats": {"total": 0, "active": 0, "sold": 0, "median_listed": None, "median_sold": None},
           "clusters": [], "drops": [], "sold": [], "dist": [], "last": None, "view": view}
    db = shared_db_path()
    if not db.exists():
        return out
    cf = (" AND i.catalog_id IN (%s)" % ",".join("?" * len(view))) if view else ""
    ca = list(view) if view else []
    try:
        with cat_connect(db) as conn:
            r = conn.execute(f"SELECT MAX(last_seen_at) FROM items i WHERE 1=1{cf}", ca).fetchone()
            out["last"] = r[0] if r else None
            out["stats"] = stats_summary(conn, catalog_ids=view)
            out["dist"] = price_distribution(conn, bins=12, catalog_ids=view)
            out["sold"] = recent_sold(conn, limit=30, catalog_ids=view)
            out["drops"] = [dict(x) for x in conn.execute(f"""
                SELECT d.old_price, d.new_price, d.drop_pct, d.detected_at, i.title, i.id AS item_id
                  FROM price_drops d JOIN items i ON i.id=d.item_id
                 WHERE 1=1{cf} ORDER BY d.detected_at DESC LIMIT 30""", ca)]
            out["clusters"] = [dict(x) for x in conn.execute(f"""
                SELECT i.cluster_key AS cluster_key,
                       COALESCE(cc.display_label, i.cluster_key) AS label,
                       COUNT(*) AS total,
                       SUM(CASE WHEN i.presumed_sold=0 THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN i.presumed_sold=1 THEN 1 ELSE 0 END) AS sold,
                       AVG(CASE WHEN i.presumed_sold=0 THEN i.price END) AS avg_listed,
                       AVG(CASE WHEN i.presumed_sold=1 THEN i.presumed_sold_price END) AS avg_sold
                  FROM items i LEFT JOIN cluster_centroids cc ON cc.cluster_key=i.cluster_key
                 WHERE i.cluster_key IS NOT NULL AND i.removed=0{cf}
                 GROUP BY i.cluster_key HAVING total>=10 ORDER BY total DESC""", ca)]
    except Exception:
        pass   # build-time read: degrade to empty rather than break the page
    return out


@ui.refreshable
def category_detail():
    m = _mon(S.sel_cat)
    if m is None:
        with ui.column().classes("p-4 items-center"):
            ui.label("Nessun monitor categoria").classes("text-lg")
            ui.button("+ Nuova categoria", on_click=_new_category_dialog).props("color=primary")
        return
    cfg = svc.load_config(m["config_file"])
    running = mm.is_running(m["id"])
    d = _cat_dashboard_data(cfg)
    s = d["stats"]
    cov = svc.coverage_state(m["id"])

    with ui.row().classes("items-center w-full"):
        ui.label(f"📊 {cfg.get('label', m['label'])}").classes("text-xl font-bold")
        ui.badge("in corso" if running else "fermo", color="green" if running else "grey-7").classes("ml-2")
        pg = cov.get("pages")
        meta = f"cat {','.join(str(c) for c in cfg.get('catalog_ids', []))} · scrape {cfg.get('scrape_interval_minutes', cfg.get('check_interval_minutes','?'))}m"
        if pg:
            meta += f" · {pg} pag · ~{len(cov.get('last_ids', []))} item/scan"
        meta += f" · {svc.time_ago(d['last'])}"
        ui.label(meta).classes("text-xs text-grey-5 ml-2")
        ui.space()
        if running:
            ui.button("■ Stop", on_click=lambda: (mm.stop(m["id"]), _soon())).props("color=red dense")
        else:
            ui.button("▶ Start", on_click=lambda: (mm.start(m), _soon())).props("color=green dense")
        ui.button("▶¹ Test", on_click=lambda: mm.test(m)).props("flat dense")
        ui.button("↻", on_click=lambda: (mm.restart(m), _soon())).props("flat dense")
        ui.button("🔄 Rigenera cluster", on_click=lambda: _regen_clusters(m)).props("flat dense color=primary")
        ui.button("✕ Elimina", on_click=lambda: _delete_category(m)).props("flat dense color=red")

    if s["total"] == 0:
        ui.label("Database vuoto per questa categoria — avvia il monitor (▶ Start).").classes("text-grey-5 p-4")
        return

    if S.cluster_filter:
        with ui.row().classes("items-center bg-grey-9 rounded px-3 py-1 mt-1"):
            ui.label(f"🎯 cluster: {S.cluster_filter}").classes("text-sm")
            ui.button("✕ tutto", on_click=lambda: (_set_cluster(None))).props("flat dense").classes("ml-2")

    with ui.row().classes("w-full gap-2 mt-2"):
        _stat_card("ATTIVI", s["active"])
        _stat_card("PRESUNTI VENDUTI", s["sold"], "text-green")
        _stat_card("MEDIANA LISTATI", f"€{int(s['median_listed'])}" if s["median_listed"] else "—", "text-blue-4")
        _stat_card("MEDIANA VENDUTI", f"€{int(s['median_sold'])}" if s["median_sold"] else "—", "text-purple-4")

    # distribution chart
    if d["dist"]:
        labels = [f"€{int(lo)}" for (lo, hi, c) in d["dist"]]
        counts = [c for (lo, hi, c) in d["dist"]]
        ui.echart({
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": "#94a3b8"}},
            "yAxis": {"type": "value", "axisLabel": {"color": "#94a3b8"}},
            "series": [{"type": "bar", "data": counts, "itemStyle": {"color": "#a78bfa"}}],
            "grid": {"left": 40, "right": 10, "top": 20, "bottom": 30},
            "tooltip": {"trigger": "axis"},
        }).classes("w-full h-48")

    with ui.row().classes("w-full gap-3 mt-2 no-wrap"):
        # cluster table (searchable)
        with ui.column().classes("flex-1 min-w-0"):
            ui.label("CLUSTER").classes("text-xs text-grey-5")
            ccols = [
                {"name": "label", "label": "Cluster", "field": "label", "sortable": True, "align": "left"},
                {"name": "total", "label": "#", "field": "total", "sortable": True},
                {"name": "active", "label": "Att", "field": "active", "sortable": True},
                {"name": "sold", "label": "Vnd", "field": "sold", "sortable": True},
                {"name": "avg_listed", "label": "€ list", "field": "avg_listed", "sortable": True},
                {"name": "avg_sold", "label": "€ vnd", "field": "avg_sold", "sortable": True},
            ]
            crows = [{"label": c["label"] or c["cluster_key"], "total": c["total"],
                      "active": c["active"] or 0, "sold": c["sold"] or 0,
                      "avg_listed": f"€{int(c['avg_listed'])}" if c["avg_listed"] else "—",
                      "avg_sold": f"€{int(c['avg_sold'])}" if c["avg_sold"] else "—",
                      "key": c["cluster_key"]} for c in d["clusters"]]
            ctbl = ui.table(columns=ccols, rows=crows, row_key="key", pagination=12).classes("w-full")
            with ctbl.add_slot("top-right"):
                with ui.row().classes("gap-2 items-center"):
                    # Item search: DB-side LIKE %q% (multi-token AND), opens
                    # a popup with results — independent from the local
                    # cluster filter which only filters the visible table.
                    cats_for_search = cfg.get("soldcheck_catalog_ids") or None
                    search_input = ui.input(placeholder="cerca item nel titolo…") \
                        .props("dense clearable") \
                        .classes("w-48")
                    search_input.on("keydown.enter",
                        lambda e, ip=search_input, cs=cats_for_search:
                            _open_item_search_popup(ip.value, cs))
                    ui.button(icon="search",
                              on_click=lambda ip=search_input, cs=cats_for_search:
                                  _open_item_search_popup(ip.value, cs)) \
                        .props("dense flat") \
                        .tooltip("Ricerca item per titolo")
                    ui.input(placeholder="filtra cluster…") \
                        .props("dense clearable") \
                        .bind_value(ctbl, "filter")
            ctbl.on("rowClick", lambda e: _open_cluster_popup(e.args[1].get("key")))
        # drops + sold
        with ui.column().classes("w-72 gap-2"):
            ui.label("📉 RIBASSI").classes("text-xs text-grey-5")
            with ui.scroll_area().classes("h-40 bg-grey-9 rounded"):
                for dr in d["drops"][:30]:
                    with ui.row().classes("cursor-pointer w-full").on(
                            "click", lambda u=f"https://www.vinted.it/items/{dr['item_id']}": _open_url(u)):
                        ui.label(f"€{int(dr['old_price'])}→€{int(dr['new_price'])} -{dr['drop_pct']}% "
                                 f"{(dr['title'] or '')[:34]}").classes("text-xs text-amber")
            ui.label("✓ PRESUNTI VENDUTI").classes("text-xs text-grey-5")
            with ui.scroll_area().classes("h-40 bg-grey-9 rounded"):
                for so in d["sold"][:30]:
                    with ui.row().classes("cursor-pointer w-full").on(
                            "click", lambda u=f"https://www.vinted.it/items/{so['id']}": _open_url(u)):
                        ui.label(f"€{int(so['price'] or 0)} {(so['title'] or '')[:34]}").classes("text-xs text-green")


def _set_cluster(key):
    S.cluster_filter = key
    category_detail.refresh()


def _open_item_search_popup(query: str, catalog_ids=None):
    """Item-level search popup: queries `search_items_by_title` (LIKE %q%
    multi-token) and renders results in a clickable table. Each row opens
    Vinted in a new tab; a separate button jumps to the item's cluster."""
    q = (query or "").strip()
    if not q or len(q) < 2:
        ui.notify("Scrivi almeno 2 caratteri", type="warning")
        return
    try:
        with cat_connect(shared_db_path()) as conn:
            items = search_items_by_title(conn, q, catalog_ids=catalog_ids,
                                          limit=100)
    except Exception as e:
        ui.notify(f"DB error: {e}", type="negative")
        return
    if not items:
        ui.notify(f"Nessun item trovato per '{q}'", type="info")
        return
    with ui.dialog() as dlg, ui.card().classes("w-[820px] max-w-[95vw]"):
        ui.label(f"🔎 Ricerca: \"{q}\"  ({len(items)} risultati"
                 f"{', limite 100' if len(items) == 100 else ''})") \
            .classes("text-lg font-bold")
        cols = [
            {"name": "status", "label": "Stato", "field": "status", "align": "left"},
            {"name": "fav", "label": "♥", "field": "fav"},
            {"name": "price", "label": "€", "field": "price", "sortable": True},
            {"name": "title", "label": "Titolo", "field": "title", "align": "left"},
            {"name": "cluster", "label": "Cluster", "field": "cluster", "align": "left"},
        ]
        rows = []
        for it in items:
            sold = bool(it.get("presumed_sold"))
            rows.append({
                "status": "venduto" if sold else "attivo",
                "fav":    it.get("favourite_count") or 0,
                "price":  int((it.get("presumed_sold_price") if sold else it.get("price")) or 0),
                "title":  it.get("title") or "",
                "cluster": it.get("cluster_key") or "—",
                "url":    f"https://www.vinted.it/items/{it['id']}",
                "ckey":   it.get("cluster_key") or "",
            })
        t = ui.table(columns=cols, rows=rows, row_key="url",
                     pagination=15).classes("w-full")
        # Row click → open Vinted. Cluster cell click handled by button below.
        t.on("rowClick", lambda e: _open_url(e.args[1].get("url")))
        with t.add_slot("body-cell-cluster", """
            <q-td :props="props">
              <a class="text-blue-4 cursor-pointer"
                 @click.stop="$parent.$emit('clickCluster', props.row.ckey)">
                {{ props.value }}
              </a>
            </q-td>
        """):
            pass
        t.on("clickCluster", lambda e: (dlg.close(),
                                          _open_cluster_popup(e.args)))
        ui.button("Chiudi", on_click=dlg.close).props("flat")
    dlg.open()


def _open_cluster_popup(key):
    if not key:
        return
    cats = (_mon(S.sel_cat) and svc.load_config(_mon(S.sel_cat)["config_file"]).get("soldcheck_catalog_ids")) or None
    try:
        with cat_connect(shared_db_path()) as conn:
            items = all_items_for_cluster(conn, key, catalog_ids=cats)
            cstats = stats_for_cluster(conn, key, catalog_ids=cats)
            hist = sold_history_for_cluster(conn, key, catalog_ids=cats)
    except Exception as e:
        ui.notify(f"DB error: {e}", type="negative"); return
    charts = []  # ECharts must be resized once the dialog is actually visible
    with ui.dialog() as dlg, ui.card().classes("w-[760px]"):
        ui.label(f"Cluster: {key}  ({len(items)} item)").classes("text-lg font-bold")

        # stat cards
        with ui.row().classes("w-full gap-2"):
            _stat_card("ATTIVI", cstats["active"])
            _stat_card("VENDUTI", cstats["sold"], "text-green")
            _stat_card("MEDIANA LIST.",
                       f"€{int(cstats['median_listed'])}" if cstats["median_listed"] else "—", "text-blue-4")
            _stat_card("MEDIANA VEND.",
                       f"€{int(cstats['median_sold'])}" if cstats["median_sold"] else "—", "text-purple-4")

        with ui.row().classes("w-full gap-3 no-wrap"):
            # price histogram of active items — always available
            prices = sorted(int(it["price"]) for it in items
                            if not it["presumed_sold"] and (it["price"] or 0) > 0)
            if prices:
                lo, hi = prices[0], prices[-1]
                nb = 10 if hi > lo else 1
                width = max((hi - lo) / nb, 1)
                edges = [lo + i * width for i in range(nb + 1)]
                counts = [0] * nb
                for p in prices:
                    bi = min(int((p - lo) / width), nb - 1)
                    counts[bi] += 1
                with ui.column().classes("flex-1 min-w-0"):
                    ui.label("PREZZI ATTIVI").classes("text-xs text-grey-5 mt-1")
                    charts.append(ui.echart({
                        "xAxis": {"type": "category",
                                  "data": [f"€{int(edges[i])}" for i in range(nb)],
                                  "axisLabel": {"color": "#94a3b8"}},
                        "yAxis": {"type": "value", "axisLabel": {"color": "#94a3b8"}},
                        "series": [{"type": "bar", "data": counts,
                                    "itemStyle": {"color": "#a78bfa"}}],
                        "grid": {"left": 36, "right": 10, "top": 16, "bottom": 28},
                        "tooltip": {"trigger": "axis"},
                    }).classes("w-full h-44"))

            # sold-over-time chart (only if there are sales)
            pts = [[h["date"], int(h["price"])] for h in (hist or []) if h.get("price")]
            if pts:
                with ui.column().classes("flex-1 min-w-0"):
                    ui.label("VENDUTI NEL TEMPO").classes("text-xs text-grey-5 mt-1")
                    charts.append(ui.echart({
                        "xAxis": {"type": "category", "data": [p[0] for p in pts],
                                  "axisLabel": {"color": "#94a3b8"}},
                        "yAxis": {"type": "value", "axisLabel": {"color": "#94a3b8",
                                  "formatter": "€{value}"}},
                        "series": [{"type": "line", "data": [p[1] for p in pts],
                                    "smooth": True, "showSymbol": True,
                                    "itemStyle": {"color": "#4ade80"},
                                    "lineStyle": {"color": "#4ade80"}}],
                        "grid": {"left": 50, "right": 12, "top": 16, "bottom": 28},
                        "tooltip": {"trigger": "axis"},
                    }).classes("w-full h-44"))

        cols = [
            {"name": "status", "label": "Stato", "field": "status", "align": "left"},
            {"name": "fav", "label": "♥", "field": "fav"},
            {"name": "price", "label": "€", "field": "price", "sortable": True},
            {"name": "title", "label": "Titolo", "field": "title", "align": "left"},
        ]
        rows = []
        for it in items:
            sold = bool(it["presumed_sold"])
            rows.append({"status": "venduto" if sold else f"attivo {it['seen_count']}×",
                         "fav": it["favourite_count"] or 0,
                         "price": int((it["presumed_sold_price"] if sold else it["price"]) or 0),
                         "title": it["title"], "url": f"https://www.vinted.it/items/{it['id']}"})
        t = ui.table(columns=cols, rows=rows, row_key="url", pagination=15).classes("w-full")
        with t.add_slot("top-right"):
            ui.input(placeholder="cerca…").props("dense clearable").bind_value(t, "filter")
        t.on("rowClick", lambda e: _open_url(e.args[1].get("url")))
        ui.button("Chiudi", on_click=dlg.close).props("flat")
    dlg.open()
    # ECharts measures 0×0 inside a not-yet-visible dialog; resize after it opens.
    if charts:
        def _resize():
            for ch in charts:
                ch.run_chart_method("resize")
        ui.timer(0.25, _resize, once=True)


def _regen_clusters(m):
    if mm.regen_active():
        ui.notify("Una rigenerazione è già in corso — guarda l'Activity log.", type="warning")
        return
    with ui.dialog() as dlg, ui.card():
        ui.label("Rigenerare i cluster?").classes("text-lg font-bold")
        ui.label("Svuota i centroidi e ri-raggruppa TUTTI gli item del DB condiviso "
                 "(embedding + traduzione). Può richiedere alcuni minuti; un backup "
                 "del DB viene creato prima di iniziare. I monitor possono restare attivi.")\
            .classes("text-sm text-grey-5 max-w-md")
        with ui.row():
            ui.button("Annulla", on_click=dlg.close).props("flat")
            def do():
                dlg.close()
                if mm.regenerate_clusters(m["id"]):
                    ui.notify("Rigenerazione avviata — progresso nell'Activity log.", type="positive")
                else:
                    ui.notify("Una rigenerazione è già in corso.", type="warning")
            ui.button("Rigenera", on_click=do).props("color=primary")
    dlg.open()


def _delete_category(m):
    with ui.dialog() as dlg, ui.card():
        ui.label(f"Eliminare '{m['label']}'?").classes("text-lg")
        ui.label("Il config viene rimosso (il DB condiviso resta).").classes("text-sm text-grey-5")
        with ui.row():
            ui.button("Annulla", on_click=dlg.close).props("flat")
            def do():
                mm.stop(m["id"])
                svc.delete_config(m["config_file"])
                S.sel_cat = None
                dlg.close()
                reload_monitors()
                sidebar.refresh()
                category_detail.refresh()
            ui.button("Elimina", on_click=do).props("color=red")
    dlg.open()


def _new_category_dialog():
    subs = svc.subcategories()
    chosen: dict[int, bool] = {}
    with ui.dialog() as dlg, ui.card().classes("w-[560px]"):
        ui.label("Nuova categoria").classes("text-lg font-bold")
        name = ui.input("Nome").props("dense").classes("w-full")
        ui.label("Sotto-categorie").classes("text-xs text-grey-5 mt-2")
        search = ui.input(placeholder="cerca sotto-categoria…").props("dense clearable").classes("w-full")

        @ui.refreshable
        def sublist():
            q = (search.value or "").lower()
            with ui.scroll_area().classes("h-56 bg-grey-9 rounded w-full"):
                for nm, cid in subs:
                    if q and not all(t in f"{nm} {cid}".lower() for t in q.split()):
                        continue
                    cb = ui.checkbox(f"{nm} ({cid})", value=chosen.get(cid, False))
                    cb.on_value_change(lambda e, c=cid: chosen.__setitem__(c, e.value))
        sublist()
        search.on_value_change(lambda: sublist.refresh())

        manual = ui.input("catalog_id extra (virgola)").props("dense").classes("w-full")
        with ui.row().classes("w-full gap-2 mt-2"):
            order = ui.select({"favourites": "+ likes ♥", "oldest": "+ vecchi", "price": "prezzo"},
                              value="favourites", label="Priorità controllo").props("dense").classes("flex-1")
            minp = ui.number("Min €", value=10).props("dense").classes("w-24")
            maxp = ui.number("Max €", value=5000).props("dense").classes("w-28")
            itv = ui.number("Scrape min", value=3).props("dense").classes("w-28")

        def create():
            cats = [c for c, v in chosen.items() if v]
            for tok in (manual.value or "").replace(",", " ").split():
                if tok.isdigit():
                    cats.append(int(tok))
            cats = sorted(set(cats))
            if not (name.value or "").strip():
                ui.notify("Inserisci un nome", type="warning"); return
            if not cats:
                ui.notify("Spunta almeno una sotto-categoria", type="warning"); return
            cfg, path = svc.make_category_config(
                name.value.strip(), cats, order=order.value,
                min_price=int(minp.value or 0), max_price=int(maxp.value or 5000),
                scrape_interval=int(itv.value or 3))
            import json
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            S.sel_cat = cfg["id"]
            dlg.close()
            reload_monitors()
            sidebar.refresh()
            category_detail.refresh()
            ui.notify(f"Creata categoria '{cfg['label']}'", type="positive")

        with ui.row():
            ui.button("Annulla", on_click=dlg.close).props("flat")
            ui.button("Crea", on_click=create).props("color=primary")
    dlg.open()


# ── Esplora ──────────────────────────────────────────────────────────────────

@ui.refreshable
def explore_view():
    ui.label("🔥 Esplora occasioni").classes("text-xl font-bold")
    ui.label("Sfoglia una macro-categoria e ordina per ♥; confronto coi prezzi del DB.").classes("text-xs text-grey-5")
    with ui.row().classes("w-full gap-2 items-end mt-2"):
        cat = ui.select(list(CATEGORIES), value=list(CATEGORIES)[0], label="Categoria").props("dense").classes("w-44")
        maxp = ui.number("Max € (0=∞)", value=0).props("dense").classes("w-28")
        minf = ui.number("Min ♥", value=0).props("dense").classes("w-24")
        topk = ui.number("Top K", value=30).props("dense").classes("w-24")
        pages = ui.number("Pagine", value=1).props("dense").classes("w-24")
        btn = ui.button("🔍 Cerca").props("color=primary")
        status = ui.label("").classes("text-sm text-grey-5")

    cols = [
        {"name": "fav", "label": "♥", "field": "fav", "sortable": True},
        {"name": "price", "label": "€", "field": "price", "sortable": True},
        {"name": "vs", "label": "vs mediana", "field": "vs", "align": "left"},
        {"name": "brand", "label": "Brand", "field": "brand", "align": "left"},
        {"name": "title", "label": "Titolo", "field": "title", "align": "left"},
    ]
    tbl = ui.table(columns=cols, rows=[], row_key="url", pagination=20).classes("w-full mt-2")
    with tbl.add_slot("top-right"):
        ui.input(placeholder="cerca…").props("dense clearable").bind_value(tbl, "filter")
    tbl.on("rowClick", lambda e: _open_url(e.args[1].get("url")))

    async def run():
        status.text = "Cerco…"
        btn.disable()
        cat_ids = CATEGORIES[cat.value]
        try:
            items, errors = await asyncio.to_thread(
                discover, cat_ids, DOMAINS_DEFAULT, (float(maxp.value) or None),
                int(minf.value or 0), int(topk.value or 30), max(1, int(pages.value or 1)))
            db = find_category_db(cat_ids)
            if db and items:
                try:
                    await asyncio.to_thread(enrich_with_median, items, db)
                except Exception:
                    pass
            tbl.rows = [{"fav": it["favourite_count"], "price": f"€{it['price']:.0f}",
                         "vs": it.get("price_label", "—"), "brand": it["brand"] or "—",
                         "title": it["title"], "url": it["url"]} for it in items]
            tbl.update()
            pool = errors.get("_pool_total", "?")
            status.text = f"{len(items)} risultati · pool {pool}"
        except Exception as e:
            status.text = f"Errore: {e}"
        btn.enable()

    btn.on_click(run)


# ── helpers ──────────────────────────────────────────────────────────────────

def _open_url(url):
    """Open `url` in a new tab WITHOUT leaking our localhost referer.

    `ui.navigate.to(..., new_tab=True)` sends `Referer: http://127.0.0.1:8080`
    to Vinted → DataDome flags the request as suspicious and shows a "session
    refresh" challenge → the user can't open the item. We use a raw
    `window.open` with `noreferrer,noopener` so the target gets a clean tab
    (no referer header, no `window.opener` back-channel)."""
    if not url:
        return
    # Escape only the bare minimum for safety; the URL is already constructed
    # from trusted item ids / fixed paths, never from user input.
    safe = url.replace("\\", "\\\\").replace("'", "\\'")
    ui.run_javascript(f"window.open('{safe}', '_blank', 'noreferrer,noopener')")


def _soon():
    """Refresh the relevant detail + sidebar shortly after a process action."""
    ui.timer(0.6, lambda: (sidebar.refresh(), searches_detail.refresh(),
                           category_detail.refresh()), once=True)


# ── Page ─────────────────────────────────────────────────────────────────────

@ui.page("/")
def main_page():
    ui.dark_mode().enable()
    with ui.header().classes("items-center justify-between bg-grey-10"):
        ui.label("Vinted Tracker").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-3"):
            net = ui.icon("circle", size="12px")
            net_lbl = ui.label("…").classes("text-sm")
            # image worker indicator + toggle
            iw = ui.icon("image", size="14px")
            def _toggle_worker():
                if mm.image_worker_running():
                    mm.stop_image_worker()          # disables auto-start too
                else:
                    mm.auto_image = True
                    mm.start_image_worker()
            iw_btn = ui.button("img-worker", on_click=_toggle_worker).props("flat dense")
            # text-worker indicator (symmetric to image-worker)
            tw = ui.icon("text_fields", size="14px")
            def _toggle_text_worker():
                if mm.text_worker_running():
                    mm.stop_text_worker()
                else:
                    mm.auto_text = True
                    mm.start_text_worker()
            tw_btn = ui.button("txt-worker", on_click=_toggle_text_worker).props("flat dense")
            # sold-check worker: separate process, lives independently of the
            # scrape monitor (you can run the drip even with scrape OFF)
            sc = ui.icon("verified", size="14px")
            def _toggle_soldcheck():
                if mm.sold_check_worker_running():
                    mm.stop_sold_check_worker()
                else:
                    mm.auto_sold = True
                    mm.start_sold_check_worker()
            sc_btn = ui.button("sold-check", on_click=_toggle_soldcheck).props("flat dense")
            ui.button("↻ Refresh", on_click=lambda: (reload_monitors(), sidebar.refresh(),
                      searches_detail.refresh(), category_detail.refresh())).props("flat dense")

    # Run the (blocking) network probe off-thread, then update the UI.
    async def net_check():
        online = await asyncio.to_thread(svc.check_network)
        net.classes(replace="text-green" if online else "text-red")
        net_lbl.text = "Online" if online else "Offline"
    _safe_timer(10.0, net_check)
    ui.timer(0.1, net_check, once=True)

    # Image worker: auto-start whenever a category monitor is active (and keep
    # the header indicator in sync). Re-activates every time a category monitor
    # comes back up.
    def worker_tick():
        cat_active = any(mm.is_running(m["id"]) for m in S.monitors
                         if m.get("type") == "category")
        mm.sync_image_worker(cat_active)
        mm.sync_text_worker(cat_active)
        mm.sync_sold_check_worker()      # independent of cat_active
        if mm.regen_active():
            # deliberately paused during a cluster regeneration — show it
            iw.classes(replace="text-amber")
            iw_btn.text = "img-worker 💤 sleep (regen)"
            tw.classes(replace="text-amber")
            tw_btn.text = "txt-worker 💤 sleep (regen)"
            sc.classes(replace="text-amber")
            sc_btn.text = "sold-check 💤 sleep (regen)"
        else:
            running_img = mm.image_worker_running()
            iw.classes(replace="text-green" if running_img else "text-grey-6")
            iw_btn.text = "img-worker ON" if running_img else "img-worker off"
            running_txt = mm.text_worker_running()
            tw.classes(replace="text-green" if running_txt else "text-grey-6")
            tw_btn.text = "txt-worker ON" if running_txt else "txt-worker off"
            running_sc = mm.sold_check_worker_running()
            sc.classes(replace="text-green" if running_sc else "text-grey-6")
            sc_btn.text = "sold-check ON" if running_sc else "sold-check off"
    _safe_timer(5.0, worker_tick)
    ui.timer(0.4, worker_tick, once=True)

    with ui.tabs().classes("w-full") as tabs:
        t_search = ui.tab("▶ Ricerche")
        t_expl = ui.tab("🔥 Esplora")
        t_cat = ui.tab("📊 Categorie")

    # Charts built inside an inactive tab panel measure 0×0; rebuild the view
    # when its tab becomes visible so ECharts sizes correctly.
    def _on_tab(e):
        if e.value == "📊 Categorie":
            category_detail.refresh()
        elif e.value == "▶ Ricerche":
            searches_detail.refresh()
    tabs.on_value_change(_on_tab)

    with ui.tab_panels(tabs, value=t_search).classes("w-full"):
        with ui.tab_panel(t_search):
            with ui.row().classes("w-full no-wrap gap-3"):
                with ui.column().classes("w-60"):
                    ui.label("RICERCHE").classes("text-xs text-grey-5")
                    sidebar("text")
                with ui.column().classes("flex-1 min-w-0"):
                    searches_detail()
        with ui.tab_panel(t_expl):
            explore_view()
        with ui.tab_panel(t_cat):
            with ui.row().classes("w-full items-center"):
                ui.label("CATEGORIE").classes("text-xs text-grey-5")
                ui.space()
                ui.button("+ Nuova categoria", on_click=_new_category_dialog).props("dense color=primary")
            with ui.row().classes("w-full no-wrap gap-3"):
                with ui.column().classes("w-60"):
                    sidebar("category")
                with ui.column().classes("flex-1 min-w-0"):
                    category_detail()

    # live log
    with ui.expansion("Activity log", value=True, icon="terminal").classes("w-full"):
        log = ui.log(max_lines=600).classes("w-full h-48 bg-black text-xs")
        cursor = {"seq": 0}

        def drain():
            for e in mm.drain_log(cursor["seq"]):
                cursor["seq"] = e["seq"]
                mark = {"found": "✓", "error": "✗", "info": "›", "muted": "·"}.get(e["tag"], " ")
                log.push(f'{e["ts"]} {mark} [{e["mid"]}] {e["text"]}')
        _safe_timer(0.7, drain)

    _safe_timer(4.0, lambda: sidebar.refresh())   # running dots refresh


def run():
    import os
    show = os.environ.get("WEBGUI_SHOW", "1") == "1"
    port = int(os.environ.get("WEBGUI_PORT", "8080"))
    try:
        ui.run(host="127.0.0.1", port=port, title="Vinted Tracker",
               show=show, reload=False, dark=True)
    except KeyboardInterrupt:
        print("\nServer fermato.")
