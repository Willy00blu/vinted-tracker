#!/usr/bin/env python3
"""
Vinted Monitor Manager — GUI
gu
Launch this file to open the control panel:
    python gui.py

Layout:
  - Top bar: title + network indicator + Refresh + tab-contextual actions
  - Tab strip: ▶ Ricerche · 🔥 Esplora · 📊 Categorie
  - Tab content area:
      * Ricerche  — sidebar (text monitors) + detail panel (Overview / History / Stats / Settings)
      * Esplora   — on-demand category browser by favourite_count
      * Categorie — sidebar (category monitors) + dashboard with charts,
                    cluster table (click to filter, double-click to open
                    item list popup), recent drops, presumed-sold list
  - Activity log: live scraping output for all subprocess monitors
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import json
import re
import os
import threading
import time
import socket
import urllib.request
import webbrowser
import datetime
from collections import defaultdict
from pathlib import Path

from core import coverage
from core.discovery import (
    CATEGORIES, DOMAINS_DEFAULT, discover,
    find_category_db, enrich_with_median, subcategories_seen,
)
from core.category_db import (
    connect as category_db_connect,
    shared_db_path,
    stats_summary as category_stats_summary,
    load_snapshots as category_load_snapshots,
    recent_sold as category_recent_sold,
    price_distribution as category_price_distribution,
    stats_for_cluster as category_stats_for_cluster,
    sold_history_for_cluster as category_sold_history,
    cluster_drops as category_cluster_drops,
    cluster_sold  as category_cluster_sold,
    all_items_for_cluster as category_all_items_for_cluster,
)

# psutil is optional — we fall back to pgrep / os.kill on macOS/Linux
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
CONFIGS_DIR    = BASE_DIR / "configs"
DATA_DIR       = BASE_DIR / "data"
MONITOR_SCRIPT = BASE_DIR / "monitor.py"


# ── Color palette: dark professional (slate + blue) ───────────────────────────
BG          = "#0f172a"   # slate-900       — main background
BG_SIDEBAR  = "#0a1020"   # slightly darker — sidebar
BG_PANEL    = "#1e293b"   # slate-800       — cards / panels
BG_HOVER    = "#334155"   # slate-700       — hover background
BG_INPUT    = "#1e293b"   # input fields
BORDER      = "#334155"   # slate-700       — separators

FG          = "#e2e8f0"   # slate-200       — primary text
FG_MUTED    = "#94a3b8"   # slate-400       — secondary text
FG_DIM      = "#64748b"   # slate-500       — tertiary / placeholder
WHITE       = "#f8fafc"   # slate-50        — high-contrast text

ACCENT      = "#3b82f6"   # blue-500        — primary accent
GREEN       = "#10b981"   # emerald-500
RED         = "#ef4444"   # red-500
YELLOW      = "#f59e0b"   # amber-500
PURPLE      = "#a78bfa"   # violet-400

BTN_START   = GREEN
BTN_STOP    = RED
BTN_PRIMARY = ACCENT
BTN_NEUTRAL = BG_HOVER

# Fonts
F_TITLE     = ("SF Pro Display", 16, "bold")
F_H1        = ("SF Pro Display", 14, "bold")
F_H2        = ("SF Pro Display", 12, "bold")
F_BODY      = ("SF Pro", 11)
F_BODY_B    = ("SF Pro", 11, "bold")
F_SMALL     = ("SF Pro", 10)
F_SMALL_B   = ("SF Pro", 10, "bold")
F_TINY      = ("SF Pro", 9)
F_MONO      = ("Menlo", 10)
F_MONO_S    = ("Menlo", 9)


# ── Flat button widget ────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    """
    Clickable label that behaves like a button with hover feedback.
    macOS Tkinter ignores `bg` on `tk.Button`, so we roll our own.
    """

    def __init__(self, parent, text, color=BTN_NEUTRAL, fg=WHITE,
                 command=None, padx=12, pady=6, font=F_BODY, **kw):
        super().__init__(parent, text=text, bg=color, fg=fg,
                         font=font, padx=padx, pady=pady,
                         cursor="hand2", **kw)
        self._color = color
        self._cmd   = command or (lambda: None)
        self.bind("<Button-1>", lambda _: self._cmd())
        self.bind("<Enter>",    lambda _: super(FlatButton, self).config(bg=self._hover()))
        self.bind("<Leave>",    lambda _: super(FlatButton, self).config(bg=self._color))

    def _hover(self):
        """Slightly lighter shade for hover state."""
        r = min(255, int(self._color[1:3], 16) + 18)
        g = min(255, int(self._color[3:5], 16) + 18)
        b = min(255, int(self._color[5:7], 16) + 18)
        return f"#{r:02x}{g:02x}{b:02x}"

    def config(self, **kw):
        if "bg" in kw:
            self._color = kw["bg"]
        if "command" in kw:
            self._cmd = kw.pop("command")
        super().config(**kw)

    configure = config


# ── Network connectivity check ────────────────────────────────────────────────

def check_network() -> tuple[bool, str]:
    for url in ("https://www.google.com", "https://www.apple.com"):
        try:
            urllib.request.urlopen(url, timeout=4)
            return True, "Online"
        except Exception:
            pass
    for host, port in [("8.8.8.8", 53), ("1.1.1.1", 53)]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            return True, "Online"
        except OSError:
            pass
    return False, "Offline"


# ── Process utilities ─────────────────────────────────────────────────────────

def find_pid(pattern: str) -> int | None:
    if HAS_PSUTIL:
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                cmdline = proc.info.get("cmdline") or []
                if any(pattern in arg for arg in cmdline):
                    return proc.info["pid"]
        except Exception:
            pass
        return None
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if r.returncode == 0:
            pids = r.stdout.strip().splitlines()
            return int(pids[0]) if pids else None
    except Exception:
        pass
    return None


def kill_pid(pid: int):
    """Kill `pid` and BLOCK until the process is fully reaped. Returning
    before the kernel has cleaned up the process leaves it visible in pgrep
    / psutil, which makes the GUI think the monitor is still running and
    keeps the Stop button stuck."""
    if HAS_PSUTIL:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=4)
            except psutil.TimeoutExpired:
                p.kill()
                try:
                    p.wait(timeout=2)   # wait for SIGKILL to be reaped
                except psutil.TimeoutExpired:
                    pass
        except Exception:
            pass
        return
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):                 # up to ~4s of polling
            time.sleep(0.1)
            try:
                os.kill(pid, 0)             # signal 0 = "is it alive?"
            except ProcessLookupError:
                return                       # gone
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.3)
        except ProcessLookupError:
            pass
    except Exception:
        pass


# ── Config & stats helpers ────────────────────────────────────────────────────

def load_monitors() -> list[dict]:
    monitors = []
    for cfg_file in sorted(CONFIGS_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_file.read_text())
            monitors.append({
                "id":          cfg["id"],
                "label":       cfg["label"],
                "type":        cfg.get("type", "text"),
                "config_file": cfg_file,
            })
        except Exception:
            pass
    # Sort: text monitors first, then category monitors
    monitors.sort(key=lambda m: (m["type"] != "text", m["label"]))
    return monitors


def _read_seen_count(seen_file: str) -> int:
    p = DATA_DIR / seen_file
    if p.exists():
        try:
            return len(json.loads(p.read_text()))
        except Exception:
            pass
    return 0


def _read_stats(cfg_id: str) -> dict:
    p = DATA_DIR / f"stats_{cfg_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"total_notified": 0, "last_found_at": None}


def _read_price_stats(cfg_id: str) -> dict:
    p = DATA_DIR / f"price_stats_{cfg_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"prices": [], "sample_count": 0, "updated_at": None, "anomalies": []}


def _read_history(cfg_id: str) -> list:
    p = DATA_DIR / f"history_{cfg_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def _time_ago(ts: str | None) -> str:
    if not ts:
        return "never"
    try:
        dt   = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        diff = int(time.time() - dt)
        if diff < 60:    return "just now"
        if diff < 3600:  return f"{diff // 60}m ago"
        if diff < 86400: return f"{diff // 3600}h ago"
        return f"{diff // 86400}d ago"
    except Exception:
        return ts


def _median(values: list) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── Monitor form (embeddable) ─────────────────────────────────────────────────

class MonitorForm(tk.Frame):
    """
    Embeddable form for monitor configuration.
    Used both inline in the Settings tab and inside the 'Add new' popup.
    """

    FIELDS = [
        ("label",               "Name",                          "entry", "e.g. Gaming PC, Mechanical Keyboard"),
        ("min_price",           "Min price (€)",                 "entry", "e.g. 10"),
        ("max_price",           "Max price (€)",                 "entry", "e.g. 500"),
        ("check_interval",      "Check every (minutes)",         "entry", "e.g. 15"),
        ("queries",             "Search queries",                "text",  "one per line\ne.g. gaming pc rtx\ndesktop gaming"),
        ("keywords_required",   "Required keywords",             "entry", "comma-separated, e.g. rtx,gaming,i7"),
        ("brands_required",     "Required brands (optional)",    "entry", "comma-separated, e.g. asus,msi,lenovo"),
        ("keywords_exclude",    "Exclude keywords",              "entry", "comma-separated, e.g. broken,spare parts"),
        ("min_favourites",      "Min likes ♥ (0 = no filter)",   "entry", "e.g. 5"),
        ("max_pages",           "Pages per scan",                "entry", "1 = 96 items · 3 = up to 288"),
        ("request_delay",       "Delay between calls (sec)",     "entry", "e.g. 1  (lower = faster, risk ban)"),
        ("anomaly_threshold",   "Anomaly threshold %",           "entry", "e.g. 30  (0 = off)"),
        ("anomaly_min_samples", "Min samples before flagging",   "entry", "e.g. 20"),
        ("price_drop_threshold","Price-drop threshold %",        "entry", "e.g. 15  (0 = off — alert on seen items dropped by ≥X%)"),
        ("price_drop_min_abs",  "Price-drop min € absolute",     "entry", "e.g. 5   (ignore drops smaller than this in €)"),
    ]

    def __init__(self, parent, existing_cfg=None, config_file=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._existing    = existing_cfg
        self._config_file = config_file
        self._editing     = existing_cfg is not None
        self._widgets: dict[str, tk.Widget] = {}
        self._build()
        if self._editing:
            self._prefill(existing_cfg)

    def _build(self):
        for key, lbl, kind, placeholder in self.FIELDS:
            tk.Label(self, text=lbl, bg=BG, fg=FG_MUTED, font=F_SMALL).pack(
                anchor="w", pady=(8, 2))

            if kind == "entry":
                var = tk.StringVar()
                e = tk.Entry(self, textvariable=var, bg=BG_INPUT, fg=FG_DIM,
                             insertbackground=FG, relief="flat",
                             font=F_BODY, width=46,
                             selectbackground=ACCENT, selectforeground=WHITE)
                e.pack(fill="x", ipady=6)
                e.insert(0, placeholder)
                e.bind("<FocusIn>",  lambda ev, w=e, ph=placeholder: self._clear_ph(ev, w, ph))
                e.bind("<FocusOut>", lambda ev, w=e, ph=placeholder, v=var: self._restore_ph(ev, w, ph, v))
                self._widgets[key] = (e, var, placeholder)
            else:
                t = tk.Text(self, bg=BG_INPUT, fg=FG_DIM, insertbackground=FG,
                            relief="flat", font=F_BODY, width=46, height=4,
                            selectbackground=ACCENT, selectforeground=WHITE)
                t.pack(fill="x")
                t.insert("1.0", placeholder)
                t.bind("<FocusIn>",  lambda ev, w=t, ph=placeholder: self._clear_ph_text(ev, w, ph))
                t.bind("<FocusOut>", lambda ev, w=t, ph=placeholder: self._restore_ph_text(ev, w, ph))
                self._widgets[key] = t

    def _prefill(self, cfg: dict):
        def set_entry(key, value):
            e, _, _ = self._widgets[key]
            e.delete(0, "end")
            e.insert(0, str(value))
            e.config(fg=FG)

        def set_text(key, value):
            t = self._widgets[key]
            t.delete("1.0", "end")
            t.insert("1.0", value)
            t.config(fg=FG)

        set_entry("label",               cfg.get("label", ""))
        set_entry("min_price",           cfg.get("min_price", 0))
        set_entry("max_price",           cfg.get("max_price", ""))
        set_entry("check_interval",      cfg.get("check_interval_minutes", 15))
        set_text ("queries",             "\n".join(cfg.get("queries", [])))
        set_entry("keywords_required",   ", ".join(cfg.get("keywords_required", [])))
        set_entry("brands_required",     ", ".join(cfg.get("brands_required", [])))
        set_entry("keywords_exclude",    ", ".join(cfg.get("keywords_exclude", [])))
        set_entry("min_favourites",      cfg.get("min_favourites", 0))
        set_entry("max_pages",           cfg.get("max_pages", 1))
        set_entry("request_delay",       cfg.get("request_delay_seconds", 2.0))
        set_entry("anomaly_threshold",   cfg.get("anomaly_threshold_pct", 0))
        set_entry("anomaly_min_samples", cfg.get("anomaly_min_samples", 20))
        set_entry("price_drop_threshold",cfg.get("price_drop_threshold_pct", 15))
        set_entry("price_drop_min_abs",  cfg.get("price_drop_min_abs", 5))

    # placeholder helpers
    def _clear_ph(self, _, widget, placeholder):
        if widget.get() == placeholder:
            widget.delete(0, "end")
            widget.config(fg=FG)

    def _restore_ph(self, _, widget, placeholder, var):
        if not widget.get().strip():
            widget.delete(0, "end")
            widget.insert(0, placeholder)
            widget.config(fg=FG_DIM)

    def _clear_ph_text(self, _, widget, placeholder):
        if widget.get("1.0", "end-1c") == placeholder:
            widget.delete("1.0", "end")
            widget.config(fg=FG)

    def _restore_ph_text(self, _, widget, placeholder):
        if not widget.get("1.0", "end-1c").strip():
            widget.delete("1.0", "end")
            widget.insert("1.0", placeholder)
            widget.config(fg=FG_DIM)

    def _get(self, key: str) -> str:
        w = self._widgets[key]
        if isinstance(w, tk.Text):
            return w.get("1.0", "end-1c").strip()
        entry, _, placeholder = w
        val = entry.get().strip()
        return "" if val == placeholder else val

    def _parse_csv(self, raw: str) -> list[str]:
        return [x.strip().lower() for x in re.split(r"[,\n]", raw) if x.strip()]

    def save(self) -> tuple[dict, Path, bool] | None:
        """
        Validate the form, write the JSON config to disk and return
        (cfg, file_path, is_edit). Returns None if validation fails.
        """
        label = self._get("label")
        if not label:
            messagebox.showerror("Error", "Please enter a name.", parent=self.winfo_toplevel())
            return None

        queries = [q.strip() for q in self._get("queries").splitlines()
                   if q.strip()
                   and not q.startswith("one per line")
                   and not q.startswith("e.g.")]
        if not queries:
            messagebox.showerror("Error", "Add at least one search query.", parent=self.winfo_toplevel())
            return None

        try:
            min_price = int(self._get("min_price") or 0)
        except ValueError:
            min_price = 0

        try:
            max_price = int(self._get("max_price") or 9999)
        except ValueError:
            messagebox.showerror("Error", "Max price must be a number.", parent=self.winfo_toplevel())
            return None

        try:
            interval = int(self._get("check_interval") or 15)
        except ValueError:
            interval = 15

        try:
            min_favourites = int(self._get("min_favourites") or 0)
        except ValueError:
            min_favourites = 0

        try:
            max_pages = max(1, int(self._get("max_pages") or 1))
        except ValueError:
            max_pages = 1

        try:
            request_delay = max(0.0, float(self._get("request_delay") or 2.0))
        except ValueError:
            request_delay = 2.0

        try:
            anomaly_threshold = max(0, int(self._get("anomaly_threshold") or 0))
        except ValueError:
            anomaly_threshold = 0

        try:
            anomaly_min_samples = max(1, int(self._get("anomaly_min_samples") or 20))
        except ValueError:
            anomaly_min_samples = 20

        try:
            price_drop_threshold = max(0, int(self._get("price_drop_threshold") or 15))
        except ValueError:
            price_drop_threshold = 15

        try:
            price_drop_min_abs = max(0.0, float(self._get("price_drop_min_abs") or 5))
        except ValueError:
            price_drop_min_abs = 5.0

        if self._editing:
            mid       = self._existing["id"]
            seen_file = self._existing["seen_file"]
            cfg_path  = self._config_file
            price_db_file = self._existing.get("price_db_file")
        else:
            mid       = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            seen_file = f"seen_{mid}_ids.json"
            cfg_path  = CONFIGS_DIR / f"{mid}.json"
            price_db_file = None

        cfg = {
            "id":                     mid,
            "label":                  label,
            "min_price":              min_price,
            "max_price":              max_price,
            "check_interval_minutes": interval,
            "seen_file":              seen_file,
            "price_db_file":          price_db_file,
            "domains": [
                "https://www.vinted.it",
                "https://www.vinted.fr",
                "https://www.vinted.de",
                # vinted.com (US) intentionally dropped — Europe-only
            ],
            "queries":                queries,
            "keywords_required":      self._parse_csv(self._get("keywords_required")),
            "brands_required":        self._parse_csv(self._get("brands_required")),
            "keywords_exclude":       self._parse_csv(self._get("keywords_exclude")),
            "min_favourites":         min_favourites,
            "max_pages":              max_pages,
            "request_delay_seconds":  request_delay,
            "anomaly_threshold_pct":  anomaly_threshold,
            "anomaly_min_samples":    anomaly_min_samples,
            "price_drop_threshold_pct": price_drop_threshold,
            "price_drop_min_abs":       price_drop_min_abs,
            "email_subject":          f"🔔 {{count}} new listing(s) on Vinted — {label}!",
            "email_intro":            f"New listings found on Vinted for: {label}",
        }
        # Preserve email overrides from existing config
        if self._editing:
            cfg["email_subject"] = self._existing.get("email_subject", cfg["email_subject"])
            cfg["email_intro"]   = self._existing.get("email_intro",   cfg["email_intro"])

        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        return cfg, cfg_path, self._editing


# ── Add-new dialog ────────────────────────────────────────────────────────────

class AddMonitorDialog(tk.Toplevel):

    def __init__(self, parent, on_save):
        super().__init__(parent)
        self.title("New search")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self.on_save = on_save

        tk.Label(self, text="Create new search", bg=BG, fg=WHITE,
                 font=F_TITLE).pack(padx=20, pady=(16, 8), anchor="w")
        tk.Label(self, text="Configure how this search hunts for listings.",
                 bg=BG, fg=FG_MUTED, font=F_SMALL).pack(padx=20, pady=(0, 12), anchor="w")

        self.form = MonitorForm(self)
        self.form.pack(fill="both", expand=True, padx=20)

        btns = tk.Frame(self, bg=BG)
        btns.pack(fill="x", padx=20, pady=(16, 20))
        FlatButton(btns, "Create", color=BTN_PRIMARY, command=self._save,
                   padx=14, pady=6, font=F_BODY_B).pack(side="right")
        FlatButton(btns, "Cancel", color=BG_HOVER, command=self.destroy,
                   padx=14, pady=6).pack(side="right", padx=(0, 8))

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width()  - self.winfo_width())  // 2
        py = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        result = self.form.save()
        if result is None:
            return
        cfg, cfg_path, _ = result
        self.destroy()
        self.on_save({"id": cfg["id"], "label": cfg["label"], "config_file": cfg_path})


# ── Category monitor config + dialog ──────────────────────────────────────────

def make_category_config(name: str, catalog_ids: list[int], *,
                         order: str = "favourites", min_price: int = 10,
                         max_price: int = 5000, scrape_interval: int = 3) -> tuple[dict, Path]:
    """Build a category-monitor config from the form inputs. `catalog_ids` are
    the selected LEAF sub-category ids; they scope BOTH the scrape (server-side)
    and the sold-check (`soldcheck_catalog_ids`), so the collector works only on
    this category's items. Pure function → unit-testable without Tk."""
    mid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "categoria"
    cfg = {
        "id":                     mid,
        "label":                  name,
        "type":                   "category",
        "catalog_ids":            catalog_ids,
        "soldcheck_catalog_ids":  catalog_ids,
        "domains": [
            "https://www.vinted.it",
            "https://www.vinted.fr",
            "https://www.vinted.de",
        ],
        "min_price":                  min_price,
        "max_price":                  max_price,
        "check_interval_minutes":     8,
        "scrape_interval_minutes":    scrape_interval,
        "pages_per_domain":           3,
        "pages_per_domain_max":       10,
        "request_delay_seconds":      2.0,
        "price_drop_threshold_pct":   15,
        "price_drop_min_abs":         5,
        "presumed_sold_min_seen":     3,
        "presumed_sold_http_quota":   20,
        "soldcheck_order":            order,
    }
    return cfg, CONFIGS_DIR / f"{mid}.json"


class CategoryDialog(tk.Toplevel):
    """Create a custom category monitor: pick sub-categories, sold-check order,
    price range, scrape cadence → writes a config the collector picks up."""

    # Display label → config value for the sold-check priority. (Favourites can
    # only be done on our side — Vinted's feed can't sort by likes.)
    _ORDER_LABELS = {
        "più likes ♥":   "favourites",
        "più vecchi":    "oldest",
        "prezzo alto":   "price",
    }

    def __init__(self, parent, on_save):
        super().__init__(parent)
        self.title("Nuova categoria")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self.on_save = on_save
        self._subcat_vars: dict[int, tk.BooleanVar] = {}

        tk.Label(self, text="Nuova categoria personalizzata", bg=BG, fg=WHITE,
                 font=F_TITLE).pack(padx=20, pady=(16, 4), anchor="w")
        tk.Label(self, text="Scegli le sotto-categorie da seguire e come ordinare il sold-check.",
                 bg=BG, fg=FG_MUTED, font=F_SMALL).pack(padx=20, pady=(0, 10), anchor="w")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20)

        tk.Label(body, text="Nome", bg=BG, fg=FG_MUTED, font=F_SMALL).pack(anchor="w", pady=(6, 2))
        self.name_var = tk.StringVar()
        tk.Entry(body, textvariable=self.name_var, bg=BG_INPUT, fg=FG, insertbackground=FG,
                 relief="flat", font=F_BODY, width=44).pack(fill="x", ipady=5)

        sc_head = tk.Frame(body, bg=BG)
        sc_head.pack(fill="x", pady=(10, 2))
        tk.Label(sc_head, text="Sotto-categorie (spunta quelle da seguire)", bg=BG,
                 fg=FG_MUTED, font=F_SMALL).pack(side="left")
        self.subsearch_var = tk.StringVar()
        se = tk.Entry(sc_head, textvariable=self.subsearch_var, bg=BG_INPUT, fg=FG,
                      insertbackground=FG, relief="flat", font=F_SMALL, width=20,
                      selectbackground=ACCENT, selectforeground=WHITE)
        se.pack(side="right", ipady=3)
        tk.Label(sc_head, text="🔎", bg=BG, fg=FG_DIM, font=F_SMALL).pack(side="right", padx=(0, 4))

        sc_wrap = tk.Frame(body, bg=BG_PANEL)
        sc_wrap.pack(fill="both", expand=True)
        canvas = tk.Canvas(sc_wrap, bg=BG_PANEL, highlightthickness=0, height=200)
        sb = ttk.Scrollbar(sc_wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG_PANEL)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        subs = subcategories_seen()
        self._subcat_rows: list[tuple[tk.Widget, str]] = []   # (checkbutton, searchtext)
        if subs:
            for nm, cid in subs:
                v = tk.BooleanVar(value=False)
                self._subcat_vars[cid] = v
                cb = tk.Checkbutton(inner, text=f"{nm}  ({cid})", variable=v,
                                    bg=BG_PANEL, fg=FG, selectcolor=BG,
                                    activebackground=BG_PANEL, activeforeground=FG,
                                    anchor="w", font=F_SMALL)
                cb.pack(fill="x", anchor="w", padx=8)
                self._subcat_rows.append((cb, f"{nm} {cid}".lower()))
        else:
            tk.Label(inner, text="(nessuna sotto-categoria nota ancora — usa il campo manuale)",
                     bg=BG_PANEL, fg=FG_DIM, font=F_SMALL).pack(padx=8, pady=8)

        def _filter_subs(_e=None):
            q = self.subsearch_var.get().strip().lower()
            for cb, _txt in self._subcat_rows:   # re-pack in order to preserve sorting
                cb.pack_forget()
            for cb, txt in self._subcat_rows:
                if not q or all(tok in txt for tok in q.split()):
                    cb.pack(fill="x", anchor="w", padx=8)
            canvas.configure(scrollregion=canvas.bbox("all"))
        se.bind("<KeyRelease>", _filter_subs)

        tk.Label(body, text="catalog_id extra (manuali, separati da virgola)", bg=BG, fg=FG_MUTED,
                 font=F_SMALL).pack(anchor="w", pady=(8, 2))
        self.manual_var = tk.StringVar()
        tk.Entry(body, textvariable=self.manual_var, bg=BG_INPUT, fg=FG, insertbackground=FG,
                 relief="flat", font=F_BODY, width=44).pack(fill="x", ipady=5)

        row = tk.Frame(body, bg=BG)
        row.pack(fill="x", pady=(12, 2))

        def num_field(parent, label, var, width=7):
            f = tk.Frame(parent, bg=BG)
            f.pack(side="left", padx=(0, 12))
            tk.Label(f, text=label, bg=BG, fg=FG_MUTED, font=F_SMALL).pack(anchor="w")
            tk.Entry(f, textvariable=var, bg=BG_INPUT, fg=FG, insertbackground=FG,
                     relief="flat", font=F_BODY, width=width).pack(ipady=4)

        of = tk.Frame(row, bg=BG)
        of.pack(side="left", padx=(0, 12))
        tk.Label(of, text="Priorità controllo venduti", bg=BG, fg=FG_MUTED,
                 font=F_SMALL).pack(anchor="w")
        self.order_var = tk.StringVar(value="più likes ♥")
        ttk.Combobox(of, textvariable=self.order_var, state="readonly", width=16,
                     values=list(self._ORDER_LABELS), font=F_SMALL).pack()
        self.minp_var = tk.StringVar(value="10");   num_field(row, "Min €", self.minp_var)
        self.maxp_var = tk.StringVar(value="5000"); num_field(row, "Max €", self.maxp_var)
        self.itv_var  = tk.StringVar(value="3");    num_field(row, "Scrape ogni (min)", self.itv_var, 6)

        btns = tk.Frame(self, bg=BG)
        btns.pack(fill="x", padx=20, pady=(16, 18))
        FlatButton(btns, "Crea", color=BTN_PRIMARY, command=self._save,
                   padx=14, pady=6, font=F_BODY_B).pack(side="right")
        FlatButton(btns, "Annulla", color=BG_HOVER, command=self.destroy,
                   padx=14, pady=6).pack(side="right", padx=(0, 8))

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width()  - self.winfo_width())  // 2
        py = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, px)}+{max(0, py)}")

    def _save(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Errore", "Inserisci un nome.", parent=self)
            return
        cats = [cid for cid, v in self._subcat_vars.items() if v.get()]
        for tok in re.split(r"[,\s]+", self.manual_var.get().strip()):
            if tok.isdigit():
                cats.append(int(tok))
        cats = sorted(set(cats))
        if not cats:
            messagebox.showerror(
                "Errore",
                "Spunta almeno una sotto-categoria (o aggiungine una manuale).",
                parent=self)
            return
        try:    minp = int(self.minp_var.get() or 0)
        except ValueError: minp = 0
        try:    maxp = int(self.maxp_var.get() or 5000)
        except ValueError: maxp = 5000
        try:    itv = max(1, int(self.itv_var.get() or 3))
        except ValueError: itv = 3
        order = self._ORDER_LABELS.get(self.order_var.get(), "favourites")
        cfg, path = make_category_config(
            name, cats, order=order,
            min_price=minp, max_price=maxp, scrape_interval=itv)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        self.destroy()
        self.on_save({"id": cfg["id"], "label": cfg["label"],
                      "type": "category", "config_file": path})


# ── Main app ──────────────────────────────────────────────────────────────────

class VintedGUI(tk.Tk):

    SIDEBAR_W = 240
    LOG_H     = 170

    TABS = [("overview", "Overview"),
            ("history",  "History"),
            ("stats",    "Price stats"),
            ("settings", "Settings")]

    def __init__(self):
        super().__init__()
        self.title("Vinted Tracker")
        self.geometry("1180x780")
        self.minsize(980, 620)
        self.configure(bg=BG)

        self.monitors    = load_monitors()
        self.state:        dict[str, dict] = {}
        self.cards:        dict[str, dict] = {}   # sidebar row widgets per monitor
        self.selected_mid: str | None = None
        self.selected_cluster: str | None = None   # filter for category dashboard
        self.active_tab    = "overview"
        self.active_top_tab = "searches"   # "searches" | "explore" | "categories"
        self._top_tab_labels: dict[str, tk.Widget] = {}
        self._last_running_state: dict[str, bool] = {}
        self._last_cat_count: dict[str, int] = {}   # category DB row counts (auto-refresh)

        # Activity log link tags
        self._link_counter = 0
        self._link_map: dict[str, str] = {}

        for m in self.monitors:
            self._init_state(m["id"])

        # _build_ui calls _render_active_top_tab which handles initial selection
        self._build_ui()
        # Detect monitor subprocesses surviving a previous GUI session (the
        # parent died but the children orphaned to init). They keep talking to
        # Vinted in the background and would show as "Running" right away,
        # which is surprising — offer to kill them.
        self.after(200, self._prompt_cleanup_orphans)
        self._schedule_refresh()

    def _init_state(self, mid: str):
        self.state[mid] = {
            "proc":    None,
            "ext_pid": None,
            "enabled": tk.BooleanVar(value=True),
        }

    # ── Layout scaffolding ────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar + global tab strip (Searches / Explore / Categories)
        self._build_topbar()
        self._build_top_tabs()

        # Container that holds whatever the active tab decides to render.
        # Each tab rebuilds sidebar+detail_panel inside this container.
        self.tab_content = tk.Frame(self, bg=BG)
        self.tab_content.pack(fill="both", expand=True)

        # Activity log at the bottom (global, shared across tabs)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        self._build_log_panel()

        # Default tab + contextual topbar buttons
        self._update_topbar_actions()
        self._render_active_top_tab()

    def _build_topbar(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=58)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Title
        title = tk.Frame(bar, bg=BG_PANEL)
        title.pack(side="left", padx=20, pady=14)
        tk.Label(title, text="Vinted Tracker", bg=BG_PANEL, fg=WHITE,
                 font=F_TITLE).pack(side="left")
        tk.Label(title, text="  ·  monitor manager", bg=BG_PANEL,
                 fg=FG_DIM, font=F_SMALL).pack(side="left")

        # Right cluster: network status + actions
        right = tk.Frame(bar, bg=BG_PANEL)
        right.pack(side="right", padx=14, pady=10)

        # Network indicator
        self.net_frame = tk.Frame(right, bg=BG)
        self.net_frame.pack(side="left", padx=(0, 12), ipadx=10, ipady=4)
        self.net_dot = tk.Label(self.net_frame, text="●", bg=BG, fg=FG_DIM,
                                font=("SF Pro", 11))
        self.net_dot.pack(side="left", padx=(0, 4))
        self.net_label = tk.Label(self.net_frame, text="…", bg=BG, fg=FG_DIM,
                                  font=F_SMALL)
        self.net_label.pack(side="left")

        # Refresh is global — useful in every tab
        FlatButton(right, " ↻  Refresh ", color=BTN_NEUTRAL,
                   command=self._refresh, padx=10, pady=5,
                   font=F_SMALL).pack(side="left", padx=(0, 6))

        # Contextual actions: rebuilt by _update_topbar_actions() whenever the
        # active top-tab changes. Charts and "+ New ..." only make sense in a
        # specific tab and shouldn't follow the user across tabs.
        self.topbar_actions = tk.Frame(right, bg=BG_PANEL)
        self.topbar_actions.pack(side="left")

        # Bottom border under top bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    def _update_topbar_actions(self):
        """Rebuild the contextual buttons in the topbar based on the active
        top-level tab."""
        if not hasattr(self, "topbar_actions"):
            return
        for w in self.topbar_actions.winfo_children():
            w.destroy()

        if self.active_top_tab == "searches":
            FlatButton(self.topbar_actions, " 📊  Charts ", color=BTN_NEUTRAL,
                       command=self._show_charts, padx=10, pady=5,
                       font=F_SMALL).pack(side="left", padx=(0, 6))
            FlatButton(self.topbar_actions, " + Nuova ricerca ", color=BTN_PRIMARY,
                       command=self._open_new_monitor_dialog, padx=12, pady=5,
                       font=F_SMALL_B).pack(side="left")
        elif self.active_top_tab == "categories":
            FlatButton(self.topbar_actions, " + Nuova categoria ", color=BTN_PRIMARY,
                       command=self._open_new_category_dialog, padx=12, pady=5,
                       font=F_SMALL_B).pack(side="left")
        # explore: no contextual buttons (the search form lives in the tab itself)

    # ── Global tab strip ──────────────────────────────────────────────────────

    TOP_TABS = [
        ("searches",   "▶  Ricerche"),
        ("explore",    "🔥  Esplora"),
        ("categories", "📊  Categorie"),
    ]

    def _build_top_tabs(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        for key, label in self.TOP_TABS:
            active = (key == self.active_top_tab)
            lbl = tk.Label(
                bar, text=f"  {label}  ", bg=BG_PANEL,
                fg=WHITE if active else FG_MUTED,
                font=F_BODY_B if active else F_BODY,
                padx=18, pady=10, cursor="hand2",
            )
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, k=key: self._switch_top_tab(k))
            self._top_tab_labels[key] = lbl
            # Hover for inactive
            lbl.bind("<Enter>", lambda e, k=key, w=lbl:
                     w.config(fg=WHITE) if k != self.active_top_tab else None)
            lbl.bind("<Leave>", lambda e, k=key, w=lbl:
                     w.config(fg=FG_MUTED) if k != self.active_top_tab else None)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    def _switch_top_tab(self, key: str):
        if key == self.active_top_tab:
            return
        self.active_top_tab = key
        for k, lbl in self._top_tab_labels.items():
            if k == key:
                lbl.config(fg=WHITE, font=F_BODY_B)
            else:
                lbl.config(fg=FG_MUTED, font=F_BODY)
        self._update_topbar_actions()
        self._render_active_top_tab()

    def _render_active_top_tab(self):
        """Tear down whatever was in the tab container and rebuild for the
        currently-selected top tab. Each tab is responsible for instantiating
        its own sidebar (if any) and detail panel."""
        for w in self.tab_content.winfo_children():
            w.destroy()
        # Reset per-monitor card refs — sidebar rebuilds them
        self.cards = {}
        # Each tab is implemented by a dedicated render method
        if   self.active_top_tab == "searches":   self._render_searches_tab()
        elif self.active_top_tab == "explore":    self._render_explore_tab()
        elif self.active_top_tab == "categories": self._render_categories_tab()

    def _render_searches_tab(self):
        """Sidebar with text monitors + detail panel using the existing
        per-monitor render path (overview/history/stats/settings)."""
        main = tk.Frame(self.tab_content, bg=BG)
        main.pack(fill="both", expand=True)
        self._build_sidebar(main, "RICERCHE",
                            filter_fn=lambda m: m.get("type", "text") == "text")
        tk.Frame(main, bg=BORDER, width=1).pack(side="left", fill="y")
        self.detail_panel = tk.Frame(main, bg=BG)
        self.detail_panel.pack(side="left", fill="both", expand=True)

        text_monitors = [m for m in self.monitors if m.get("type", "text") == "text"]
        if not text_monitors:
            self._render_empty_detail()
            return
        # Preserve previous selection if still valid for this tab
        if self.selected_mid and any(m["id"] == self.selected_mid for m in text_monitors):
            self._select_monitor(self.selected_mid)
        else:
            self._select_monitor(text_monitors[0]["id"])

    def _render_explore_tab(self):
        """Explore feature rendered directly into tab_content (no sidebar)."""
        self.selected_mid = None
        self.detail_panel = self.tab_content   # _render_explore writes here
        self._render_explore()

    def _render_categories_tab(self):
        """Sidebar with category monitors + dashboard detail."""
        main = tk.Frame(self.tab_content, bg=BG)
        main.pack(fill="both", expand=True)
        self._build_sidebar(main, "CATEGORIE",
                            filter_fn=lambda m: m.get("type") == "category")
        tk.Frame(main, bg=BORDER, width=1).pack(side="left", fill="y")
        self.detail_panel = tk.Frame(main, bg=BG)
        self.detail_panel.pack(side="left", fill="both", expand=True)

        cat_monitors = [m for m in self.monitors if m.get("type") == "category"]
        if not cat_monitors:
            self._render_empty_categories()
            return
        if self.selected_mid and any(m["id"] == self.selected_mid for m in cat_monitors):
            self._select_monitor(self.selected_mid)
        else:
            self._select_monitor(cat_monitors[0]["id"])

    def _render_empty_categories(self):
        wrap = tk.Frame(self.detail_panel, bg=BG)
        wrap.pack(fill="both", expand=True)
        inner = tk.Frame(wrap, bg=BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(inner, text="Nessun monitor categoria",
                 bg=BG, fg=FG, font=F_TITLE).pack()
        tk.Label(inner,
                 text="Crea un file in configs/ con \"type\": \"category\" e un \"catalog_ids\" Vinted.\n"
                      "Vedi configs/category_tecnologia.json come esempio.",
                 bg=BG, fg=FG_MUTED, font=F_BODY, justify="center").pack(pady=(8, 0))

    def _build_sidebar(self, parent, section_label: str = "SEARCHES",
                       filter_fn=None):
        """Build the sidebar inside `parent`. If `filter_fn` is given, only
        monitors for which `filter_fn(monitor) == True` are listed."""
        filter_fn = filter_fn or (lambda m: True)
        self.sidebar = tk.Frame(parent, bg=BG_SIDEBAR, width=self.SIDEBAR_W)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Section header
        hdr = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        hdr.pack(fill="x", padx=14, pady=(16, 6))
        tk.Label(hdr, text=section_label, bg=BG_SIDEBAR, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(side="left")
        self.sidebar_count = tk.Label(hdr, text="0", bg=BG_SIDEBAR, fg=FG_DIM,
                                      font=("SF Pro", 9))
        self.sidebar_count.pack(side="right")

        # Scrollable list of monitors
        list_outer = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        list_outer.pack(fill="both", expand=True, padx=8)

        canvas = tk.Canvas(list_outer, bg=BG_SIDEBAR, highlightthickness=0,
                           bd=0)
        canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_inner = tk.Frame(canvas, bg=BG_SIDEBAR)
        self.sidebar_inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw",
                              width=self.SIDEBAR_W - 16)

        for m in self.monitors:
            if filter_fn(m):
                self._build_sidebar_row(m)

        # Bottom area: global controls
        footer = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        footer.pack(fill="x", side="bottom", padx=8, pady=12)
        tk.Frame(footer, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        FlatButton(footer, " Start enabled ", color=BTN_START,
                   command=self._start_all, padx=10, pady=6,
                   font=F_SMALL_B).pack(fill="x", pady=(0, 6))
        FlatButton(footer, " Stop all ", color=BG_HOVER,
                   command=self._stop_all, padx=10, pady=6,
                   font=F_SMALL).pack(fill="x")

    def _build_sidebar_row(self, m: dict):
        mid = m["id"]
        row = tk.Frame(self.sidebar_inner, bg=BG_SIDEBAR, cursor="hand2")
        row.pack(fill="x", pady=1)

        inner = tk.Frame(row, bg=BG_SIDEBAR)
        inner.pack(fill="x", padx=8, pady=6)

        dot = tk.Label(inner, text="●", bg=BG_SIDEBAR, fg=FG_DIM,
                       font=("SF Pro", 12))
        dot.pack(side="left", padx=(0, 8))

        info = tk.Frame(inner, bg=BG_SIDEBAR)
        info.pack(side="left", fill="x", expand=True)
        name_lbl = tk.Label(info, text=m["label"], bg=BG_SIDEBAR, fg=FG,
                            font=F_BODY, anchor="w")
        name_lbl.pack(anchor="w")
        sub_lbl = tk.Label(info, text="—", bg=BG_SIDEBAR, fg=FG_DIM,
                           font=F_TINY, anchor="w")
        sub_lbl.pack(anchor="w")

        chk = tk.Checkbutton(inner, variable=self.state[mid]["enabled"],
                             bg=BG_SIDEBAR, fg=FG, selectcolor=BG,
                             activebackground=BG_SIDEBAR,
                             activeforeground=FG, bd=0, padx=0)
        chk.pack(side="right")

        # Click anywhere on the row → select monitor (but not on the checkbox)
        for w in (row, inner, info, name_lbl, sub_lbl, dot):
            w.bind("<Button-1>", lambda e, i=mid: self._select_monitor(i))

        self.cards[mid] = {
            "row":      row,
            "inner":    inner,
            "dot":      dot,
            "name_lbl": name_lbl,
            "sub_lbl":  sub_lbl,
            "info":     info,
            "chk":      chk,
        }

    def _build_log_panel(self):
        panel = tk.Frame(self, bg=BG_PANEL, height=self.LOG_H)
        panel.pack(fill="x", side="bottom")
        panel.pack_propagate(False)

        hdr = tk.Frame(panel, bg=BG_PANEL)
        hdr.pack(fill="x", padx=14, pady=(8, 4))
        tk.Label(hdr, text="ACTIVITY LOG", bg=BG_PANEL, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(side="left")

        def _clear_log():
            self.log_box.config(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.config(state="disabled")

        FlatButton(hdr, "Clear", color=BG_HOVER, command=_clear_log,
                   padx=8, pady=2, font=F_TINY).pack(side="right")

        log_wrap = tk.Frame(panel, bg=BG_PANEL)
        log_wrap.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        self.log_box = tk.Text(log_wrap, bg=BG, fg=FG, font=F_MONO_S,
                               relief="flat", state="disabled", wrap="word",
                               bd=0, padx=10, pady=8,
                               selectbackground=ACCENT, selectforeground=WHITE)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_wrap, command=self.log_box.yview)
        sb.pack(side="right", fill="y")
        self.log_box.configure(yscrollcommand=sb.set)

        self.log_box.tag_config("found",   foreground=GREEN)
        self.log_box.tag_config("info",    foreground=ACCENT)
        self.log_box.tag_config("warn",    foreground=YELLOW)
        self.log_box.tag_config("error",   foreground=RED)
        self.log_box.tag_config("muted",   foreground=FG_DIM)
        self.log_box.tag_config("default", foreground=FG)

    # ── Detail panel rendering ────────────────────────────────────────────────

    def _clear_detail(self):
        for w in self.detail_panel.winfo_children():
            w.destroy()

    def _render_empty_detail(self):
        self._clear_detail()
        wrap = tk.Frame(self.detail_panel, bg=BG)
        wrap.pack(fill="both", expand=True)
        inner = tk.Frame(wrap, bg=BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(inner, text="No searches yet", bg=BG, fg=FG,
                 font=F_TITLE).pack()
        tk.Label(inner, text="Click  + New search  to start tracking listings.",
                 bg=BG, fg=FG_MUTED, font=F_BODY).pack(pady=(8, 16))
        FlatButton(inner, " + New search ", color=BTN_PRIMARY,
                   command=self._open_new_monitor_dialog,
                   padx=14, pady=8, font=F_BODY_B).pack()

    # ── Category monitor detail (📊 Mercato) ───────────────────────────────────

    def _render_category_detail(self, mid: str, m: dict, cfg: dict):
        db_path = shared_db_path()               # one shared DB for all categories
        cluster_filter = self.selected_cluster   # None = whole category
        # This category is a FILTERED VIEW of the shared DB, by its leaf
        # catalog_ids (None = the broad "everything" view, e.g. legacy [2994]).
        view_cats = cfg.get("soldcheck_catalog_ids") or None
        self._cat_view_cats = view_cats          # for the cluster-items popup
        # Inline-query fragment (aliased i.) for the catalog scope.
        _cf = (" AND i.catalog_id IN (%s)" % ",".join("?" * len(view_cats))) if view_cats else ""
        _ca = list(view_cats) if view_cats else []

        # Read DB snapshot — queries are cluster-aware when a cluster is selected
        stats: dict        = {"total": 0, "active": 0, "sold": 0,
                              "median_listed": None, "median_sold": None}
        clusters:     list = []
        recent_drops: list = []
        recent_sold_l: list = []
        snapshots:    list = []
        distribution: list = []
        sold_history: list = []
        last_scan_at: str | None = None
        cluster_filter_label: str | None = cluster_filter
        if db_path.exists():
            try:
                with category_db_connect(db_path) as conn:
                    row = conn.execute(
                        f"SELECT MAX(last_seen_at) FROM items i WHERE 1=1{_cf}", _ca
                    ).fetchone()
                    if row and row[0]:
                        last_scan_at = row[0]
                    # Cluster list, scoped to this category's catalog_ids.
                    # LEFT JOIN on cluster_centroids lets us prefer the English
                    # translated display_label (if available) over the raw key.
                    clusters = conn.execute(f"""
                        SELECT i.cluster_key                                                  AS cluster_key,
                               COALESCE(cc.display_label, i.cluster_key)                      AS display_label,
                               COUNT(*)                                                       AS total,
                               SUM(CASE WHEN i.presumed_sold = 0 THEN 1 ELSE 0 END)           AS active,
                               SUM(CASE WHEN i.presumed_sold = 1 THEN 1 ELSE 0 END)           AS sold,
                               AVG(CASE WHEN i.presumed_sold = 0 THEN i.price END)            AS avg_listed,
                               AVG(CASE WHEN i.presumed_sold = 1 THEN i.presumed_sold_price END) AS avg_sold
                          FROM items i
                          LEFT JOIN cluster_centroids cc ON cc.cluster_key = i.cluster_key
                         WHERE i.cluster_key IS NOT NULL AND i.removed = 0{_cf}
                         GROUP BY i.cluster_key
                         HAVING total >= 2
                         ORDER BY total DESC
                         LIMIT 100
                    """, _ca).fetchall()

                    if cluster_filter:
                        stats        = category_stats_for_cluster(conn, cluster_filter, catalog_ids=view_cats)
                        distribution = category_price_distribution(
                            conn, bins=12, cluster_key=cluster_filter, catalog_ids=view_cats)
                        sold_history = category_sold_history(conn, cluster_filter, catalog_ids=view_cats)
                        # English display label for the filter banner
                        row = conn.execute(
                            "SELECT display_label FROM cluster_centroids WHERE cluster_key = ?",
                            (cluster_filter,),
                        ).fetchone()
                        cluster_filter_label = (row and row["display_label"]) or cluster_filter
                        recent_drops = conn.execute(f"""
                            SELECT d.old_price, d.new_price, d.drop_pct, d.detected_at,
                                   i.title, i.id AS item_id
                              FROM price_drops d
                              JOIN items i ON i.id = d.item_id
                             WHERE i.cluster_key = ?{_cf}
                             ORDER BY d.detected_at DESC
                             LIMIT 30
                        """, [cluster_filter] + _ca).fetchall()
                        recent_sold_l = category_cluster_sold(conn, cluster_filter, limit=30, catalog_ids=view_cats)
                    else:
                        stats        = category_stats_summary(conn, catalog_ids=view_cats)
                        # The 14-day median trend (daily_snapshots) is DB-wide and
                        # has no per-category dimension → only show it for the broad
                        # (unfiltered) view, not for a scoped category.
                        snapshots    = category_load_snapshots(conn, days=14) if not view_cats else []
                        distribution = category_price_distribution(conn, bins=12, catalog_ids=view_cats)
                        recent_drops = conn.execute(f"""
                            SELECT d.old_price, d.new_price, d.drop_pct, d.detected_at,
                                   i.title, i.id AS item_id
                              FROM price_drops d
                              JOIN items i ON i.id = d.item_id
                             WHERE 1=1{_cf}
                             ORDER BY d.detected_at DESC
                             LIMIT 30
                        """, _ca).fetchall()
                        recent_sold_l = category_recent_sold(conn, limit=30, catalog_ids=view_cats)
            except Exception as e:
                self._log(f"[{mid.upper()}] ✗ DB read error: {e}", tag="error")

        # Header
        hdr = tk.Frame(self.detail_panel, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 6))
        left = tk.Frame(hdr, bg=BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=f"📊  {cfg.get('label', mid)}",
                 bg=BG, fg=WHITE, font=("SF Pro Display", 18, "bold")
                 ).pack(anchor="w")
        meta = tk.Frame(left, bg=BG)
        meta.pack(anchor="w", pady=(4, 0))

        running = self._is_running(mid)
        status_color = GREEN if running else FG_DIM
        status_text  = "Running" if running else "Inactive"
        badge = tk.Frame(meta, bg=BG_PANEL)
        badge.pack(side="left", padx=(0, 10), ipadx=8, ipady=2)
        tk.Label(badge, text="●", bg=BG_PANEL, fg=status_color,
                 font=("SF Pro", 10)).pack(side="left", padx=(2, 4))
        tk.Label(badge, text=status_text, bg=BG_PANEL, fg=FG,
                 font=F_SMALL_B).pack(side="left", padx=(0, 2))

        cat_str = f"cat {','.join(str(c) for c in cfg.get('catalog_ids', []))}"
        # Live-status string: how fresh is the DB vs how often it should refresh
        interval_min = cfg.get("check_interval_minutes", 30)
        if last_scan_at:
            scan_str = f"ultima scansione: {_time_ago(last_scan_at)}"
        else:
            scan_str = "mai scansionato"
        # Pages are now adaptive (coverage controller) — show the live value and
        # the items/scan it settled on, not the static config number.
        cov_state = coverage.state(f"cat_{cfg['id']}")
        dyn_pages = cov_state.get("pages")
        if dyn_pages:
            n_items = len(cov_state.get("last_ids", []))
            pages_str = f"{dyn_pages} pag/dominio · ~{n_items} item/scan (auto)"
        else:
            pages_str = f"{cfg.get('pages_per_domain', 1)} pag/dominio"
        meta_parts = [
            cat_str, "·",
            f"every {interval_min} min", "·",
            pages_str, "·",
            scan_str,
        ]
        for txt in meta_parts:
            color = FG_DIM if txt == "·" else FG_MUTED
            tk.Label(meta, text=txt, bg=BG, fg=color, font=F_SMALL).pack(
                side="left", padx=2)

        # Live indicator: if monitor is running and last scan is fresh → green
        # "Live"; if running but stale → yellow warning; if not running → grey.
        live_state = self._compute_live_state(running, last_scan_at, interval_min)
        if live_state:
            text, color = live_state
            tk.Label(meta, text=f"  {text}", bg=BG, fg=color,
                     font=F_SMALL_B).pack(side="left", padx=(8, 0))

        actions = tk.Frame(hdr, bg=BG)
        actions.pack(side="right")
        if running:
            FlatButton(actions, " ■  Stop ", color=BTN_STOP,
                       command=lambda: self._toggle_monitor(mid),
                       padx=12, pady=6, font=F_SMALL_B).pack(side="left")
        else:
            FlatButton(actions, " ▶  Start ", color=BTN_START,
                       command=lambda: self._toggle_monitor(mid),
                       padx=12, pady=6, font=F_SMALL_B).pack(side="left")
        FlatButton(actions, " ▶¹  Test ", color=BG_HOVER,
                   command=lambda: self._test_monitor(mid),
                   padx=10, pady=6, font=F_SMALL).pack(side="left", padx=(6, 0))
        FlatButton(actions, " ↻ ", color=BTN_NEUTRAL,
                   command=lambda: self._restart_monitor(mid),
                   padx=10, pady=6, font=F_SMALL).pack(side="left", padx=(6, 0))
        FlatButton(actions, " 🔄  Rigenera cluster ", color=BG_HOVER,
                   command=lambda mid_=mid, cfg_=cfg:
                       self._regen_clusters(mid_, cfg_),
                   padx=10, pady=6, font=F_SMALL).pack(side="left", padx=(6, 0))
        FlatButton(actions, " ✕  Elimina ", color="#7f1d1d",
                   command=lambda: self._delete_monitor(mid),
                   padx=10, pady=6, font=F_SMALL).pack(side="left", padx=(6, 0))

        tk.Frame(self.detail_panel, bg=BORDER, height=1).pack(
            fill="x", padx=24, pady=(16, 0))

        # Body
        body = tk.Frame(self.detail_panel, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(16, 18))

        # Empty-state
        if stats["total"] == 0:
            empty = tk.Frame(body, bg=BG)
            empty.pack(fill="both", expand=True)
            inner = tk.Frame(empty, bg=BG)
            inner.place(relx=0.5, rely=0.4, anchor="center")
            tk.Label(inner, text="Database vuoto", bg=BG, fg=FG,
                     font=F_TITLE).pack()
            tk.Label(inner,
                     text="Avvia il monitor (▶ Start) per popolare il DB.\n"
                          "Nessun item di questa categoria ancora nel DB condiviso.",
                     bg=BG, fg=FG_MUTED, font=F_BODY, justify="center"
                     ).pack(pady=(8, 0))
            return

        # ── Cluster filter banner (when a cluster row is selected) ───────────
        if cluster_filter:
            banner = tk.Frame(body, bg=BG_PANEL)
            banner.pack(fill="x", pady=(0, 12), ipady=4)
            tk.Label(banner, text=f"🎯  Cluster: ", bg=BG_PANEL, fg=FG_DIM,
                     font=F_SMALL).pack(side="left", padx=(12, 0))
            tk.Label(banner, text=cluster_filter_label, bg=BG_PANEL, fg=WHITE,
                     font=F_BODY_B).pack(side="left", padx=(0, 12))
            FlatButton(banner, " ✕  Tutto ", color=BG_HOVER,
                       command=self._clear_cluster_filter,
                       padx=8, pady=2, font=F_TINY).pack(side="right", padx=(0, 10))

        # Stats row
        stats_row = tk.Frame(body, bg=BG)
        stats_row.pack(fill="x")

        def stat_card(label, value, color=FG):
            card = tk.Frame(stats_row, bg=BG_PANEL)
            card.pack(side="left", fill="both", expand=True, padx=(0, 10))
            tk.Label(card, text=label, bg=BG_PANEL, fg=FG_DIM,
                     font=("SF Pro", 9, "bold")).pack(
                anchor="w", padx=14, pady=(12, 2))
            tk.Label(card, text=str(value), bg=BG_PANEL, fg=color,
                     font=("SF Pro Display", 18, "bold")).pack(
                anchor="w", padx=14, pady=(0, 12))

        ml = stats["median_listed"]
        ms = stats["median_sold"]
        stat_card("ATTIVI",           stats["active"],                                FG)
        stat_card("PRESUNTI VENDUTI", stats["sold"],                                  GREEN)
        stat_card("MEDIANA LISTATI",  f"€{int(ml)}" if ml else "—",                  ACCENT)
        stat_card("MEDIANA VENDUTI",  f"€{int(ms)}" if ms else "—",                  PURPLE)

        # ── Charts row: median trend (14d) OR cluster sold history + price distribution
        charts_row = tk.Frame(body, bg=BG)
        charts_row.pack(fill="x", pady=(16, 0))

        median_card = tk.Frame(charts_row, bg=BG_PANEL)
        median_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        if cluster_filter:
            median_title = "PREZZI VENDUTI · CRONOLOGIA CLUSTER"
        else:
            median_title = "MEDIANA PREZZI · ULTIMI 14 GG"
        tk.Label(median_card, text=median_title, bg=BG_PANEL, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        median_canvas = tk.Canvas(median_card, height=140, bg=BG_PANEL,
                                  highlightthickness=0)
        median_canvas.pack(fill="x", padx=10, pady=(0, 12))
        # Redraw on every resize so the chart uses the actual canvas width
        # instead of the (stale) requested width.
        if cluster_filter:
            median_canvas.bind(
                "<Configure>",
                lambda e: self._draw_sold_history_chart(median_canvas, sold_history, e.width))
            median_canvas.after_idle(
                lambda: self._draw_sold_history_chart(median_canvas, sold_history))
        else:
            median_canvas.bind(
                "<Configure>",
                lambda e: self._draw_median_chart(median_canvas, snapshots, e.width))
            median_canvas.after_idle(
                lambda: self._draw_median_chart(median_canvas, snapshots))

        dist_card = tk.Frame(charts_row, bg=BG_PANEL)
        dist_card.pack(side="left", fill="both", expand=True)
        dist_title = ("DISTRIBUZIONE PREZZI · CLUSTER" if cluster_filter
                      else "DISTRIBUZIONE PREZZI · ITEM ATTIVI")
        tk.Label(dist_card, text=dist_title, bg=BG_PANEL, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        dist_canvas = tk.Canvas(dist_card, height=140, bg=BG_PANEL,
                                highlightthickness=0)
        dist_canvas.pack(fill="x", padx=10, pady=(0, 12))
        dist_canvas.bind(
            "<Configure>",
            lambda e: self._draw_distribution_chart(dist_canvas, distribution, e.width))
        dist_canvas.after_idle(
            lambda: self._draw_distribution_chart(dist_canvas, distribution))

        # ── Three-column split: clusters (large) + drops + sold ──────────────
        cols = tk.Frame(body, bg=BG)
        cols.pack(fill="both", expand=True, pady=(16, 0))

        sold_col = tk.Frame(cols, bg=BG, width=260)
        sold_col.pack(side="right", fill="y", padx=(10, 0))
        sold_col.pack_propagate(False)

        drops_col = tk.Frame(cols, bg=BG, width=260)
        drops_col.pack(side="right", fill="y", padx=(10, 0))
        drops_col.pack_propagate(False)

        cluster_col = tk.Frame(cols, bg=BG)
        cluster_col.pack(side="left", fill="both", expand=True)

        # Cluster table — header + search bar on the same row
        head = tk.Frame(cluster_col, bg=BG)
        head.pack(fill="x", pady=(0, 6))
        tk.Label(head, text="CLUSTER (top per # item)",
                 bg=BG, fg=FG_DIM, font=("SF Pro", 9, "bold")
                 ).pack(side="left")
        search_var = tk.StringVar()
        search_entry = tk.Entry(head, textvariable=search_var,
                                bg=BG_INPUT, fg=FG, insertbackground=FG,
                                relief="flat", font=F_SMALL, width=22,
                                selectbackground=ACCENT, selectforeground=WHITE)
        search_entry.pack(side="right", ipady=3)
        tk.Label(head, text="🔎  Filtra:", bg=BG, fg=FG_DIM,
                 font=F_SMALL).pack(side="right", padx=(0, 6))

        cluster_cols = ("name", "total", "active", "sold", "listed", "soldp")
        ctree = ttk.Treeview(cluster_col, columns=cluster_cols,
                             show="headings", height=14)
        ctree.heading("name",   text="Cluster")
        ctree.heading("total",  text="#")
        ctree.heading("active", text="Att.")
        ctree.heading("sold",   text="Vend.")
        ctree.heading("listed", text="€ list.")
        ctree.heading("soldp",  text="€ vend.")
        ctree.column("name",   width=240, anchor="w",      stretch=True)
        ctree.column("total",  width=50,  anchor="center", stretch=False)
        ctree.column("active", width=55,  anchor="center", stretch=False)
        ctree.column("sold",   width=55,  anchor="center", stretch=False)
        ctree.column("listed", width=75,  anchor="e",      stretch=False)
        ctree.column("soldp",  width=75,  anchor="e",      stretch=False)
        cvsb = ttk.Scrollbar(cluster_col, orient="vertical", command=ctree.yview)
        ctree.configure(yscrollcommand=cvsb.set)
        ctree.pack(side="left", fill="both", expand=True)
        cvsb.pack(side="right", fill="y")
        # Map tree-iid → cluster_key so handlers can recover it
        iid_to_cluster: dict[str, str] = {}

        def populate_clusters(query: str = ""):
            ctree.delete(*ctree.get_children())
            iid_to_cluster.clear()
            q = (query or "").strip().lower()
            for row in clusters:
                key   = row["cluster_key"] or ""
                # Prefer English translated label; fall back to raw key
                label = row["display_label"] or key or "—"
                # Search matches BOTH the displayed label and the raw key
                # (so users can find clusters by original-language word too)
                if q and q not in label.lower() and q not in key.lower():
                    continue
                avg_l = row["avg_listed"]
                avg_s = row["avg_sold"]
                iid = ctree.insert("", "end", values=(
                    label,
                    row["total"],
                    row["active"] or 0,
                    row["sold"]   or 0,
                    f"€{int(avg_l)}" if avg_l else "—",
                    f"€{int(avg_s)}" if avg_s else "—",
                ))
                if key:
                    iid_to_cluster[iid] = key
                    if key == cluster_filter:
                        ctree.selection_set(iid)

        populate_clusters()

        # Restore the scroll position from before the re-render so clicking a
        # cluster in the middle of the list doesn't snap the table to the top.
        pending_scroll = getattr(self, "_cluster_tree_scroll", None)
        if pending_scroll is not None:
            ctree.after(0, lambda y=pending_scroll: ctree.yview_moveto(y))
            self._cluster_tree_scroll = None

        # Live-filter on every keystroke
        def on_search(_event=None):
            populate_clusters(search_var.get())
        search_entry.bind("<KeyRelease>", on_search)

        # Single-click triggers cluster filter (which re-renders the whole
        # dashboard, destroying this very treeview). To let the user still
        # double-click for the items-popup, we DELAY the filter action by
        # ~280ms (standard double-click window). If a double-click arrives
        # in that window, we cancel the pending filter and open the popup
        # instead — no re-render race.
        def apply_cluster_filter(key: str):
            self._cluster_click_after_id = None
            try:
                self._cluster_tree_scroll = ctree.yview()[0]
            except Exception:
                self._cluster_tree_scroll = None
            self.selected_cluster = key
            self._render_detail(self.selected_mid)

        def on_cluster_click(_event=None):
            sel = ctree.selection()
            if not sel:
                return
            key = iid_to_cluster.get(sel[0])
            if not key or key == self.selected_cluster:
                return
            # Cancel a previously scheduled filter (rapid sequential clicks)
            prev = getattr(self, "_cluster_click_after_id", None)
            if prev:
                self.after_cancel(prev)
            self._cluster_click_after_id = self.after(
                280, lambda k=key: apply_cluster_filter(k),
            )
        ctree.bind("<<TreeviewSelect>>", on_cluster_click)

        def on_cluster_dblclick(_event=None):
            # Preempt the pending single-click filter so the popup wins
            pending = getattr(self, "_cluster_click_after_id", None)
            if pending:
                self.after_cancel(pending)
                self._cluster_click_after_id = None
            sel = ctree.selection()
            if not sel:
                return
            key = iid_to_cluster.get(sel[0])
            if key:
                self._open_cluster_items_popup(db_path, key)
        ctree.bind("<Double-Button-1>", on_cluster_dblclick)
        ctree.bind("<Return>",          on_cluster_dblclick)

        # Drops column
        tk.Label(drops_col, text="📉 RECENTI RIBASSI", bg=BG, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(anchor="w", pady=(0, 6))
        self._build_event_list(
            drops_col,
            items=recent_drops,
            empty_msg="Nessun ribasso ancora.",
            row_renderer=self._drop_row,
        )

        # Sold column
        tk.Label(sold_col, text="✓ PRESUNTI VENDUTI", bg=BG, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(anchor="w", pady=(0, 6))
        self._build_event_list(
            sold_col,
            items=recent_sold_l,
            empty_msg="Nessun item venduto ancora.\n(serve seen_count≥3 + 2 cicli senza vista)",
            row_renderer=self._sold_row,
        )

    # ── Category-dashboard helpers ────────────────────────────────────────────

    def _build_event_list(self, parent, items, empty_msg, row_renderer):
        """Scrollable list of clickable rows for drops/sold panels."""
        wrap = tk.Frame(parent, bg=BG_PANEL)
        wrap.pack(fill="both", expand=True)
        if not items:
            tk.Label(wrap, text=empty_msg, bg=BG_PANEL, fg=FG_DIM,
                     font=F_SMALL, justify="center").pack(pady=20, padx=10)
            return
        canvas = tk.Canvas(wrap, bg=BG_PANEL, highlightthickness=0)
        sb     = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=BG_PANEL)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for it in items:
            row_renderer(inner, it)

    def _drop_row(self, parent, d):
        row = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        row.pack(fill="x", padx=10, pady=4)
        url = f"https://www.vinted.it/items/{d['item_id']}"
        price_lbl = tk.Label(
            row,
            text=f"€{int(d['old_price'])}→€{int(d['new_price'])} -{d['drop_pct']}%",
            bg=BG_PANEL, fg=YELLOW, font=F_SMALL_B)
        price_lbl.pack(anchor="w")
        title_lbl = tk.Label(row, text=(d["title"] or "")[:48],
                             bg=BG_PANEL, fg=FG, font=F_SMALL, anchor="w")
        title_lbl.pack(anchor="w")
        date_lbl = tk.Label(row, text=_time_ago(d["detected_at"]),
                            bg=BG_PANEL, fg=FG_DIM, font=F_TINY)
        date_lbl.pack(anchor="w")
        for w in (row, price_lbl, title_lbl, date_lbl):
            w.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

    def _sold_row(self, parent, s):
        row = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        row.pack(fill="x", padx=10, pady=4)
        url = f"https://www.vinted.it/items/{s['id']}"
        price_lbl = tk.Label(row, text=f"€{int(s['price'] or 0)}  presunto venduto",
                             bg=BG_PANEL, fg=GREEN, font=F_SMALL_B)
        price_lbl.pack(anchor="w")
        title_lbl = tk.Label(row, text=(s["title"] or "")[:48],
                             bg=BG_PANEL, fg=FG, font=F_SMALL, anchor="w")
        title_lbl.pack(anchor="w")
        date_lbl = tk.Label(row, text=_time_ago(s["sold_at"]),
                            bg=BG_PANEL, fg=FG_DIM, font=F_TINY)
        date_lbl.pack(anchor="w")
        for w in (row, price_lbl, title_lbl, date_lbl):
            w.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

    def _draw_median_chart(self, canvas, snapshots, width=None):
        """Line chart: median_listed (blue) + median_sold (purple) over time."""
        W = width or canvas.winfo_width() or canvas.winfo_reqwidth() or 380
        H = 140
        PAD_L, PAD_R, PAD_T, PAD_B = 36, 12, 14, 22
        canvas.delete("all")
        if not snapshots:
            canvas.create_text(W // 2, H // 2,
                               text="Serve almeno 1 ciclo per i dati storici",
                               fill=FG_DIM, font=("SF Pro", 9))
            return

        ml_series = [s["median_listed"] for s in snapshots if s["median_listed"]]
        ms_series = [s["median_sold"]   for s in snapshots if s["median_sold"]]
        all_vals = ml_series + ms_series
        if not all_vals:
            canvas.create_text(W // 2, H // 2, text="Nessun prezzo registrato",
                               fill=FG_DIM, font=("SF Pro", 9))
            return
        lo, hi = min(all_vals), max(all_vals)
        if hi <= lo:
            hi = lo + 1
        n = len(snapshots)

        # Axes
        draw_w = max(1, W - PAD_L - PAD_R)
        draw_h = H - PAD_T - PAD_B
        for f in (0, 0.5, 1):
            y = PAD_T + int(f * draw_h)
            val = hi - f * (hi - lo)
            canvas.create_line(PAD_L, y, W - PAD_R, y, fill=BORDER)
            canvas.create_text(PAD_L - 4, y, text=f"€{int(val)}",
                               anchor="e", fill=FG_DIM, font=("Menlo", 8))

        def x_of(i): return PAD_L + (int(i * draw_w / max(1, n - 1)) if n > 1 else draw_w // 2)
        def y_of(v): return PAD_T + int((hi - v) / (hi - lo) * draw_h)

        # Plot listed (blue) + sold (purple)
        def plot(series_key, color):
            pts = []
            for i, s in enumerate(snapshots):
                v = s[series_key]
                if v is not None:
                    pts.append((x_of(i), y_of(v)))
            if len(pts) >= 2:
                flat = [c for xy in pts for c in xy]
                canvas.create_line(*flat, fill=color, width=2, smooth=True)
            for x, y in pts:
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2,
                                   fill=color, outline="")
        plot("median_listed", ACCENT)
        plot("median_sold",   PURPLE)

        # X-axis date labels (first, middle, last)
        for i in (0, n // 2, n - 1):
            if 0 <= i < n:
                date_str = snapshots[i]["date"][5:]   # MM-DD
                canvas.create_text(x_of(i), H - PAD_B + 6,
                                   text=date_str, anchor="n",
                                   fill=FG_DIM, font=("Menlo", 8))
        # Legend
        canvas.create_text(PAD_L, 4, text="● listati", anchor="nw",
                           fill=ACCENT, font=("SF Pro", 8, "bold"))
        canvas.create_text(PAD_L + 60, 4, text="● venduti", anchor="nw",
                           fill=PURPLE, font=("SF Pro", 8, "bold"))

    def _draw_distribution_chart(self, canvas, bins, width=None):
        """Histogram of active listing prices."""
        W = width or canvas.winfo_width() or canvas.winfo_reqwidth() or 380
        H = 140
        PAD_L, PAD_R, PAD_T, PAD_B = 36, 12, 14, 22
        canvas.delete("all")
        if not bins:
            canvas.create_text(W // 2, H // 2,
                               text="Nessun item per il grafico",
                               fill=FG_DIM, font=("SF Pro", 9))
            return
        counts = [c for (_lo, _hi, c) in bins]
        max_c = max(counts) or 1
        n     = len(bins)
        draw_w = max(1, W - PAD_L - PAD_R)
        draw_h = H - PAD_T - PAD_B
        bar_w  = max(4, draw_w // n - 3)

        for f in (0, 0.5, 1):
            y = PAD_T + int(f * draw_h)
            val = max_c - f * max_c
            canvas.create_line(PAD_L, y, W - PAD_R, y, fill=BORDER)
            canvas.create_text(PAD_L - 4, y, text=str(int(val)),
                               anchor="e", fill=FG_DIM, font=("Menlo", 8))

        for i, (lo, hi, c) in enumerate(bins):
            x_center = PAD_L + int((i + 0.5) * draw_w / n)
            bar_h    = int(c / max_c * draw_h) if max_c else 0
            x0, x1   = x_center - bar_w // 2, x_center + bar_w // 2
            y0, y1   = PAD_T + draw_h - bar_h, PAD_T + draw_h
            color    = PURPLE if c > 0 else BG_HOVER
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            if i % 3 == 0:
                canvas.create_text(x_center, H - PAD_B + 6,
                                   text=f"€{int(lo)}", anchor="n",
                                   fill=FG_DIM, font=("Menlo", 8))

    def _draw_sold_history_chart(self, canvas, history, width=None):
        """Scatter+line chart of sold prices over time for the selected cluster."""
        W = width or canvas.winfo_width() or canvas.winfo_reqwidth() or 380
        H = 140
        PAD_L, PAD_R, PAD_T, PAD_B = 36, 12, 14, 22
        canvas.delete("all")
        if not history:
            canvas.create_text(
                W // 2, H // 2,
                text="Nessun item venduto in questo cluster\n(serve seen_count≥3 + 2 cicli senza vista)",
                fill=FG_DIM, font=("SF Pro", 9), justify="center")
            return
        prices = [h["price"] for h in history]
        lo, hi = min(prices), max(prices)
        if hi <= lo:
            hi = lo + 1
        n = len(history)
        draw_w = max(1, W - PAD_L - PAD_R)
        draw_h = H - PAD_T - PAD_B

        for f in (0, 0.5, 1):
            y = PAD_T + int(f * draw_h)
            val = hi - f * (hi - lo)
            canvas.create_line(PAD_L, y, W - PAD_R, y, fill=BORDER)
            canvas.create_text(PAD_L - 4, y, text=f"€{int(val)}",
                               anchor="e", fill=FG_DIM, font=("Menlo", 8))

        def x_of(i): return PAD_L + (int(i * draw_w / max(1, n - 1)) if n > 1 else draw_w // 2)
        def y_of(v): return PAD_T + int((hi - v) / (hi - lo) * draw_h)

        pts = [(x_of(i), y_of(h["price"])) for i, h in enumerate(history)]
        if len(pts) >= 2:
            flat = [c for xy in pts for c in xy]
            canvas.create_line(*flat, fill=PURPLE, width=1)
        for x, y in pts:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                               fill=PURPLE, outline="")
        # Edge dates
        for i in (0, n - 1):
            d = history[i]["date"]
            if d:
                canvas.create_text(x_of(i), H - PAD_B + 6,
                                   text=d[5:], anchor="n",
                                   fill=FG_DIM, font=("Menlo", 8))
        # Median line (faint)
        med = sorted(prices)[n // 2]
        ym = y_of(med)
        canvas.create_line(PAD_L, ym, W - PAD_R, ym, fill=PURPLE,
                           dash=(2, 4))
        canvas.create_text(W - PAD_R, ym - 4,
                           text=f"med €{int(med)}", anchor="se",
                           fill=PURPLE, font=("SF Pro", 8, "bold"))

    def _compute_live_state(self, running: bool, last_scan_at: str | None,
                            interval_min: int):
        """Return (text, color) for the live-status pill next to the category
        header, or None if there's nothing meaningful to show."""
        if not running:
            if last_scan_at is None:
                return ("◯ idle · mai avviato",   FG_DIM)
            return  (f"◯ idle · ultima scansione {_time_ago(last_scan_at)}", FG_DIM)
        if last_scan_at is None:
            return ("● live · scansione in corso…", ACCENT)
        # Parse age
        try:
            dt = time.mktime(time.strptime(last_scan_at, "%Y-%m-%d %H:%M:%S"))
            age_sec = int(time.time() - dt)
        except Exception:
            return ("● live", GREEN)
        # Grace period: 2× the configured interval before we flag staleness
        grace = max(120, int(interval_min) * 60 * 2)
        if age_sec > grace:
            return (f"⚠ ferma da {_time_ago(last_scan_at)} — atteso ogni {interval_min}m", YELLOW)
        if age_sec < 90:
            return ("● live · appena scansionato", GREEN)
        next_min = max(0, interval_min - age_sec // 60)
        return (f"● live · prossima ~{next_min}m", GREEN)

    def _open_cluster_items_popup(self, db_path, cluster_key: str):
        """Toplevel listing every item in a cluster with filters + clickable
        rows that open the Vinted URL in the browser."""
        display_label = cluster_key
        view_cats = getattr(self, "_cat_view_cats", None)
        try:
            with category_db_connect(db_path) as conn:
                items = category_all_items_for_cluster(conn, cluster_key, catalog_ids=view_cats)
                row = conn.execute(
                    "SELECT display_label FROM cluster_centroids WHERE cluster_key = ?",
                    (cluster_key,),
                ).fetchone()
                if row and row["display_label"]:
                    display_label = row["display_label"]
        except Exception as e:
            messagebox.showerror("Errore DB", str(e), parent=self)
            return

        win = tk.Toplevel(self)
        win.title(f"Cluster: {display_label}")
        win.configure(bg=BG)
        win.geometry("780x520")

        # Header
        hdr = tk.Frame(win, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(hdr, text=display_label, bg=BG, fg=WHITE,
                 font=("SF Pro Display", 14, "bold")).pack(anchor="w")
        if display_label != cluster_key:
            tk.Label(hdr, text=f"originale: {cluster_key}", bg=BG, fg=FG_DIM,
                     font=F_TINY).pack(anchor="w")
        meta = tk.Frame(hdr, bg=BG)
        meta.pack(anchor="w", pady=(2, 0))
        n_active = sum(1 for it in items if not it["presumed_sold"])
        n_sold   = sum(1 for it in items if     it["presumed_sold"])
        for text in (f"{len(items)} totali", "·",
                     f"{n_active} attivi", "·",
                     f"{n_sold} presunti venduti"):
            color = FG_DIM if text == "·" else FG_MUTED
            tk.Label(meta, text=text, bg=BG, fg=color, font=F_SMALL
                     ).pack(side="left", padx=2)

        # Filter toolbar
        filter_var = tk.StringVar(value="all")
        toolbar = tk.Frame(win, bg=BG)
        toolbar.pack(fill="x", padx=16, pady=(8, 6))

        # Table
        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        cols = ("status", "likes", "price", "title")
        tree = ttk.Treeview(body, columns=cols, show="headings")
        tree.heading("status", text="Stato")
        tree.heading("likes",  text="♥")
        tree.heading("price",  text="€")
        tree.heading("title",  text="Titolo (doppio click per aprire)")
        tree.column("status", width=110, anchor="w",      stretch=False)
        tree.column("likes",  width=50,  anchor="center", stretch=False)
        tree.column("price",  width=80,  anchor="e",      stretch=False)
        tree.column("title",  width=500, anchor="w",      stretch=True)
        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        iid_to_url: dict[str, str] = {}

        def populate():
            tree.delete(*tree.get_children())
            iid_to_url.clear()
            mode = filter_var.get()
            for it in items:
                is_sold = bool(it["presumed_sold"])
                if mode == "active" and is_sold:
                    continue
                if mode == "sold"   and not is_sold:
                    continue
                if is_sold:
                    status = f"✓ vend. {_time_ago(it['presumed_sold_at'])}"
                    price  = it["presumed_sold_price"] or 0
                else:
                    status = f"● attivo · {it['seen_count']}× visto"
                    price  = it["price"] or 0
                iid = tree.insert("", "end", values=(
                    status,
                    f"♥{it['favourite_count'] or 0}",
                    f"€{int(price)}",
                    it["title"],
                ))
                iid_to_url[iid] = f"https://www.vinted.it/items/{it['id']}"

        def set_filter(mode: str):
            filter_var.set(mode)
            for k, btn in filter_buttons.items():
                btn.config(bg=BTN_PRIMARY if k == mode else BG_HOVER)
            populate()

        filter_buttons: dict[str, FlatButton] = {
            "all":    FlatButton(toolbar, " Tutti ",   color=BTN_PRIMARY,
                                 command=lambda: set_filter("all"),
                                 padx=10, pady=4, font=F_SMALL),
            "active": FlatButton(toolbar, " Attivi ", color=BG_HOVER,
                                 command=lambda: set_filter("active"),
                                 padx=10, pady=4, font=F_SMALL),
            "sold":   FlatButton(toolbar, " Venduti ", color=BG_HOVER,
                                 command=lambda: set_filter("sold"),
                                 padx=10, pady=4, font=F_SMALL),
        }
        for btn in filter_buttons.values():
            btn.pack(side="left", padx=(0, 6))

        def on_open(_event=None):
            sel = tree.selection()
            if sel and (url := iid_to_url.get(sel[0])):
                webbrowser.open(url)
        tree.bind("<Double-Button-1>", on_open)
        tree.bind("<Return>",          on_open)

        populate()

        # Footer
        footer = tk.Frame(win, bg=BG)
        footer.pack(fill="x", padx=16, pady=(0, 14))
        FlatButton(footer, " Chiudi ", color=BG_HOVER,
                   command=win.destroy, padx=14, pady=5,
                   font=F_SMALL).pack(side="right")

    def _open_new_category_dialog(self):
        CategoryDialog(self, on_save=self._on_new_category_created)

    def _on_new_category_created(self, m: dict):
        mid = m["id"]
        if mid in self.state:
            self._log(f"[{mid.upper()}] Config categoria aggiornata.", tag="info")
            return
        m.setdefault("type", "category")
        self.monitors.append(m)
        self._init_state(mid)
        self._log(f"[{mid.upper()}] Nuova categoria: {m['label']}", tag="found")
        # Rebuild the Categorie tab (sidebar picks up the new monitor) and select it.
        self.selected_mid = mid
        if self.active_top_tab == "categories":
            self._render_active_top_tab()
        else:
            self._switch_top_tab("categories")

    def _clear_cluster_filter(self):
        """Exit cluster-focused view and re-render the dashboard for the whole
        category."""
        if self.selected_cluster is None:
            return
        self.selected_cluster = None
        if self.selected_mid:
            self._render_detail(self.selected_mid)

    def _select_monitor(self, mid: str, tab: str | None = None):
        if mid not in {m["id"] for m in self.monitors}:
            self._render_empty_detail()
            return
        if mid != self.selected_mid:
            self.selected_cluster = None   # reset cluster filter on monitor change
        self.selected_mid = mid
        if tab:
            self.active_tab = tab
        self._render_detail(mid)
        self._refresh_sidebar()

    def _switch_tab(self, key: str):
        if key == self.active_tab:
            return
        self.active_tab = key
        if self.selected_mid:
            self._render_detail(self.selected_mid)

    def _render_detail(self, mid: str):
        self._clear_detail()
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            self._render_empty_detail()
            return

        try:
            cfg = json.loads(m["config_file"].read_text())
        except Exception:
            cfg = {"label": m["label"]}

        # Category monitors use a completely different render path because
        # they store data in SQLite and don't share the seen/history files
        # that the text-monitor tabs assume.
        if cfg.get("type") == "category":
            self._render_category_detail(mid, m, cfg)
            return

        # Header strip
        hdr = tk.Frame(self.detail_panel, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 6))

        left = tk.Frame(hdr, bg=BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=cfg.get("label", m["label"]),
                 bg=BG, fg=WHITE, font=("SF Pro Display", 18, "bold")
                 ).pack(anchor="w")

        # Status badge + quick stats line
        meta = tk.Frame(left, bg=BG)
        meta.pack(anchor="w", pady=(4, 0))

        running = self._is_running(mid)
        status_color = GREEN if running else FG_DIM
        status_text  = "Running" if running else "Inactive"
        s = self.state[mid]
        if not running and s["proc"] is not None and s["proc"].poll() is not None:
            status_color = RED
            status_text  = f"Crashed (exit {s['proc'].poll()})"

        badge = tk.Frame(meta, bg=BG_PANEL)
        badge.pack(side="left", padx=(0, 10), ipadx=8, ipady=2)
        tk.Label(badge, text="●", bg=BG_PANEL, fg=status_color,
                 font=("SF Pro", 10)).pack(side="left", padx=(2, 4))
        tk.Label(badge, text=status_text, bg=BG_PANEL, fg=FG,
                 font=F_SMALL_B).pack(side="left", padx=(0, 2))

        # Quick stats
        stats     = _read_stats(mid)
        seen_n    = _read_seen_count(cfg.get("seen_file", f"seen_{mid}_ids.json"))
        last_str  = _time_ago(stats.get("last_found_at"))
        total_str = f"{stats.get('total_notified', 0)} notified"
        seen_str  = f"{seen_n} seen IDs"
        for txt in (total_str, "·", seen_str, "·", f"last find {last_str}"):
            color = FG_DIM if txt == "·" else FG_MUTED
            tk.Label(meta, text=txt, bg=BG, fg=color, font=F_SMALL).pack(
                side="left", padx=2)

        # Action buttons on the right
        actions = tk.Frame(hdr, bg=BG)
        actions.pack(side="right")
        if running:
            FlatButton(actions, " ■  Stop ", color=BTN_STOP,
                       command=lambda: self._toggle_monitor(mid),
                       padx=12, pady=6, font=F_SMALL_B).pack(side="left")
        else:
            FlatButton(actions, " ▶  Start ", color=BTN_START,
                       command=lambda: self._toggle_monitor(mid),
                       padx=12, pady=6, font=F_SMALL_B).pack(side="left")
        FlatButton(actions, " ↻ ", color=BTN_NEUTRAL,
                   command=lambda: self._restart_monitor(mid),
                   padx=10, pady=6, font=F_SMALL).pack(side="left", padx=(6, 0))

        # Tab bar
        tabs_bar = tk.Frame(self.detail_panel, bg=BG)
        tabs_bar.pack(fill="x", padx=24, pady=(16, 0))

        for key, label in self.TABS:
            active = (key == self.active_tab)
            color  = WHITE if active else FG_MUTED
            font   = F_BODY_B if active else F_BODY
            tab = tk.Label(tabs_bar, text=label, bg=BG, fg=color,
                           font=font, padx=4, pady=8, cursor="hand2")
            tab.pack(side="left", padx=(0, 18))
            if active:
                under = tk.Frame(tabs_bar, bg=ACCENT, height=2)
                under.place(in_=tab, x=0, rely=1.0, relwidth=1.0, anchor="sw")
            tab.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))

        tk.Frame(self.detail_panel, bg=BORDER, height=1).pack(
            fill="x", padx=24, pady=(0, 0))

        # Tab content area
        content = tk.Frame(self.detail_panel, bg=BG)
        content.pack(fill="both", expand=True, padx=24, pady=(16, 18))

        if self.active_tab == "overview":
            self._render_overview(content, mid, cfg)
        elif self.active_tab == "history":
            self._render_history(content, mid, cfg)
        elif self.active_tab == "stats":
            self._render_stats(content, mid, cfg)
        elif self.active_tab == "settings":
            self._render_settings(content, m, cfg)

    # ── Overview tab ──────────────────────────────────────────────────────────

    def _render_overview(self, parent, mid: str, cfg: dict):
        # Two-column layout: left = info cards, right = quick actions.
        # IMPORTANT: pack the fixed-width column FIRST. Otherwise, when `left`
        # is packed with expand=True before `right`, long content in `left`
        # (e.g. many keywords/queries) can push `right` outside the viewport
        # and the ACTIONS column disappears for that monitor.
        cols = tk.Frame(parent, bg=BG)
        cols.pack(fill="both", expand=True)
        right = tk.Frame(cols, bg=BG, width=240)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        left  = tk.Frame(cols, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # Stats cards row
        stats_row = tk.Frame(left, bg=BG)
        stats_row.pack(fill="x")

        st = _read_stats(mid)
        ps = _read_price_stats(mid)
        seen_n = _read_seen_count(cfg.get("seen_file", ""))

        def stat_card(parent_, label, value, color=FG):
            card = tk.Frame(parent_, bg=BG_PANEL)
            card.pack(side="left", fill="both", expand=True, padx=(0, 10))
            tk.Label(card, text=label, bg=BG_PANEL, fg=FG_DIM,
                     font=("SF Pro", 9, "bold")).pack(
                anchor="w", padx=14, pady=(12, 2))
            tk.Label(card, text=str(value), bg=BG_PANEL, fg=color,
                     font=("SF Pro Display", 20, "bold")).pack(
                anchor="w", padx=14, pady=(0, 12))

        stat_card(stats_row, "NOTIFIED",     st.get("total_notified", 0), GREEN)
        stat_card(stats_row, "SEEN IDS",     seen_n,                       FG)
        stat_card(stats_row, "PRICE SAMPLES", len(ps.get("prices", [])),  PURPLE)
        # Anomaly card
        med = _median(ps.get("prices", []))
        if cfg.get("anomaly_threshold_pct", 0):
            ready = len(ps.get("prices", [])) >= cfg.get("anomaly_min_samples", 20)
            val = f"€{int(med)}" if (med and ready) else "—"
            color = YELLOW if (med and ready) else FG_DIM
            stat_card(stats_row, "MARKET MEDIAN", val, color)
        else:
            stat_card(stats_row, "ANOMALY", "off", FG_DIM)

        # Filters summary card
        filt_card = tk.Frame(left, bg=BG_PANEL)
        filt_card.pack(fill="x", pady=(14, 0))
        tk.Label(filt_card, text="FILTERS",
                 bg=BG_PANEL, fg=FG_DIM, font=("SF Pro", 9, "bold")
                 ).pack(anchor="w", padx=16, pady=(12, 6))

        def filter_row(label, value, accent=FG):
            row = tk.Frame(filt_card, bg=BG_PANEL)
            row.pack(fill="x", padx=16, pady=2)
            tk.Label(row, text=label, bg=BG_PANEL, fg=FG_MUTED, font=F_SMALL,
                     width=22, anchor="w").pack(side="left")
            tk.Label(row, text=value, bg=BG_PANEL, fg=accent, font=F_SMALL_B,
                     anchor="w").pack(side="left")

        filter_row("Price range",
                   f"€{cfg.get('min_price', 0)} – €{cfg.get('max_price', '?')}")
        filter_row("Check interval",
                   f"{cfg.get('check_interval_minutes', '?')} min")
        filter_row("Required keywords",
                   ", ".join(cfg.get("keywords_required", [])) or "—")
        filter_row("Required brands",
                   ", ".join(cfg.get("brands_required", [])) or "any")
        filter_row("Exclude keywords",
                   ", ".join(cfg.get("keywords_exclude", [])) or "—")
        filter_row("Min likes ♥",
                   str(cfg.get("min_favourites", 0)) if cfg.get("min_favourites", 0) else "no filter")
        filter_row("Pages per scan",
                   f"{cfg.get('max_pages', 1)} (up to {96 * cfg.get('max_pages', 1)} items)")
        filter_row("Request delay",
                   f"{cfg.get('request_delay_seconds', 2.0)}s")

        thr = cfg.get("anomaly_threshold_pct", 0)
        anomaly_val = (f"-{thr}% below median (min {cfg.get('anomaly_min_samples', 20)} samples)"
                       if thr else "disabled")
        filter_row("Anomaly detection",
                   anomaly_val,
                   YELLOW if thr else FG_DIM)

        drop_thr = cfg.get("price_drop_threshold_pct", 15)
        drop_min = cfg.get("price_drop_min_abs", 5)
        drop_val = (f"-{drop_thr}% AND ≥ €{drop_min} drop on seen items"
                    if drop_thr else "disabled")
        filter_row("Price-drop alerts",
                   drop_val,
                   YELLOW if drop_thr else FG_DIM)

        tk.Frame(filt_card, bg=BG_PANEL, height=10).pack()

        # Queries card
        q_card = tk.Frame(left, bg=BG_PANEL)
        q_card.pack(fill="both", expand=True, pady=(14, 0))
        tk.Label(q_card, text="SEARCH QUERIES",
                 bg=BG_PANEL, fg=FG_DIM, font=("SF Pro", 9, "bold")
                 ).pack(anchor="w", padx=16, pady=(12, 6))
        for q in cfg.get("queries", []):
            tk.Label(q_card, text=f"• {q}", bg=BG_PANEL, fg=FG,
                     font=F_MONO, anchor="w").pack(anchor="w", padx=16, pady=1)
        tk.Frame(q_card, bg=BG_PANEL, height=12).pack()

        # Right column: action panel
        ap = tk.Frame(right, bg=BG_PANEL)
        ap.pack(fill="both", expand=True)
        tk.Label(ap, text="ACTIONS", bg=BG_PANEL, fg=FG_DIM,
                 font=("SF Pro", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 8))

        def action_btn(text, color, cmd):
            FlatButton(ap, f"  {text}  ", color=color, command=cmd,
                       padx=10, pady=8, font=F_SMALL_B).pack(
                fill="x", padx=14, pady=4)

        running = self._is_running(mid)
        if running:
            action_btn("■  Stop monitor",       BTN_STOP,  lambda: self._toggle_monitor(mid))
            action_btn("↻  Restart",            BTN_NEUTRAL, lambda: self._restart_monitor(mid))
        else:
            action_btn("▶  Start monitor",      BTN_START, lambda: self._toggle_monitor(mid))
        action_btn("▶¹  Test scan (once)",     BG_HOVER,  lambda: self._test_monitor(mid))
        tk.Frame(ap, bg=BG_PANEL, height=6).pack()
        action_btn("🗑  Reset seen IDs",        BG_HOVER,  lambda: self._reset_seen(mid))
        action_btn("✕  Delete search",         "#7f1d1d", lambda: self._delete_monitor(mid))

    # ── History tab ───────────────────────────────────────────────────────────

    def _render_history(self, parent, mid: str, cfg: dict):
        history = _read_history(mid)

        # Toolbar
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(0, 10))
        tk.Label(bar, text=f"{len(history)} items in history",
                 bg=BG, fg=FG_MUTED, font=F_SMALL).pack(side="left")

        def _clear():
            if not messagebox.askyesno("Clear history",
                    f"Clear all notification history for '{cfg.get('label', mid)}'?",
                    parent=self):
                return
            (DATA_DIR / f"history_{mid}.json").write_text("[]")
            self._log(f"[{mid.upper()}] History cleared", tag="warn")
            self._render_detail(mid)

        FlatButton(bar, " Clear history ", color=BG_HOVER, command=_clear,
                   padx=10, pady=5, font=F_SMALL).pack(side="right")

        if not history:
            tk.Label(parent,
                     text="No notifications yet. Start the monitor to begin tracking.",
                     bg=BG, fg=FG_DIM, font=F_BODY).pack(pady=40)
            return

        # Scrollable table
        container = tk.Frame(parent, bg=BG)
        container.pack(fill="both", expand=True)

        canvas  = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner   = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollb.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollb.pack(side="right", fill="y")

        headers = [("Date", 24), ("Price", 8), ("♥", 5), ("Title", 60)]
        for col, (txt, w) in enumerate(headers):
            tk.Label(inner, text=txt.upper(), bg=BG, fg=FG_DIM,
                     font=("SF Pro", 9, "bold"), width=w, anchor="w").grid(
                row=0, column=col, padx=8, pady=(0, 6), sticky="w")

        for i, item in enumerate(reversed(history), start=1):
            notified_at = item.get("notified_at", "")
            date  = f"{notified_at[:16]}  ·  {_time_ago(notified_at)}" if notified_at else "—"
            fav   = item.get("favourite_count", 0)
            title = item.get("title", "")
            url   = item.get("url", "")
            is_drop = item.get("event_type") == "price_drop"

            # Price column: for drops, show "old → new" with red→green
            if is_drop:
                old = item.get("old_price", 0)
                new = item.get("price", 0)
                pct = item.get("drop_pct", 0)
                price_text = f"€{int(old)}→€{int(new)} -{pct}%"
                price_color = YELLOW
            else:
                price_text  = f"€{item.get('price', '?')}"
                price_color = GREEN

            tk.Label(inner, text=date, bg=BG, fg=FG_MUTED, font=F_MONO_S,
                     anchor="w").grid(row=i, column=0, padx=8, pady=2, sticky="w")
            tk.Label(inner, text=price_text, bg=BG, fg=price_color, font=F_SMALL_B,
                     anchor="w").grid(row=i, column=1, padx=8, pady=2, sticky="w")
            tk.Label(inner, text=str(fav) if fav else "—", bg=BG,
                     fg=PURPLE if fav else FG_DIM, font=F_SMALL,
                     anchor="w").grid(row=i, column=2, padx=8, pady=2, sticky="w")

            title_prefix = "📉  " if is_drop else ""
            title_lbl = tk.Label(inner, text=f"{title_prefix}{title}", bg=BG, fg=FG,
                                 font=F_SMALL, anchor="w", cursor="hand2",
                                 wraplength=500)
            title_lbl.grid(row=i, column=3, padx=8, pady=2, sticky="w")
            if url:
                title_lbl.config(fg=ACCENT)
                title_lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
                title_lbl.bind("<Enter>",    lambda e, l=title_lbl: l.config(fg=WHITE))
                title_lbl.bind("<Leave>",    lambda e, l=title_lbl: l.config(fg=ACCENT))

    # ── Price stats tab ───────────────────────────────────────────────────────

    def _render_stats(self, parent, mid: str, cfg: dict):
        ps = _read_price_stats(mid)
        prices    = ps.get("prices", [])
        anomalies = ps.get("anomalies", [])
        med       = _median(prices)
        threshold = cfg.get("anomaly_threshold_pct", 0)
        min_samp  = cfg.get("anomaly_min_samples", 20)
        ready     = len(prices) >= min_samp

        # Stat cards
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 14))

        def card(label, value, color=FG):
            c = tk.Frame(row, bg=BG_PANEL)
            c.pack(side="left", fill="both", expand=True, padx=(0, 10))
            tk.Label(c, text=label, bg=BG_PANEL, fg=FG_DIM,
                     font=("SF Pro", 9, "bold")).pack(
                anchor="w", padx=14, pady=(12, 2))
            tk.Label(c, text=str(value), bg=BG_PANEL, fg=color,
                     font=("SF Pro Display", 18, "bold")).pack(
                anchor="w", padx=14, pady=(0, 12))

        card("WINDOW", f"{len(prices)}/500", FG)
        card("LIFETIME", str(ps.get("sample_count", 0)), FG_MUTED)
        card("MEDIAN", f"€{int(med)}" if med else "—", YELLOW if med else FG_DIM)
        if prices:
            card("RANGE", f"€{int(min(prices))} – €{int(max(prices))}", FG_MUTED)
        else:
            card("RANGE", "—", FG_DIM)
        card("UPDATED", _time_ago(ps.get("updated_at")), FG_MUTED)

        # Detection status banner
        banner = tk.Frame(parent, bg=BG_PANEL)
        banner.pack(fill="x", pady=(0, 14))
        if threshold and ready:
            txt   = f"✓  Anomaly detection active  ·  flagging prices below €{int(med * (1 - threshold / 100))} (-{threshold}% from median)"
            color = GREEN
        elif threshold:
            need = min_samp - len(prices)
            txt   = f"⏳  Collecting data — need {need} more sample(s) before flagging (-{threshold}% threshold)"
            color = YELLOW
        else:
            txt   = "○  Anomaly detection disabled  ·  set a threshold in Settings to enable"
            color = FG_DIM
        tk.Label(banner, text="  " + txt, bg=BG_PANEL, fg=color,
                 font=F_SMALL_B).pack(anchor="w", padx=16, pady=10)

        # Histogram
        if prices:
            tk.Label(parent, text="DISTRIBUTION", bg=BG, fg=FG_DIM,
                     font=("SF Pro", 9, "bold")).pack(anchor="w", pady=(0, 4))

            W, H = 880, 160
            PAD_L, PAD_R, PAD_T, PAD_B = 50, 16, 18, 28
            canvas = tk.Canvas(parent, width=W, height=H, bg=BG_PANEL,
                               highlightthickness=0)
            canvas.pack(fill="x")

            n_bins = 28
            p_min, p_max = min(prices), max(prices)
            if p_max == p_min:
                p_max = p_min + 1
            bins = [0] * n_bins
            for p in prices:
                idx = min(n_bins - 1, int((p - p_min) / (p_max - p_min) * n_bins))
                bins[idx] += 1
            max_count = max(bins) if bins else 1

            draw_w = W - PAD_L - PAD_R
            draw_h = H - PAD_T - PAD_B
            bar_w  = max(2, draw_w // n_bins - 2)

            for i, count in enumerate(bins):
                x_center = PAD_L + int((i + 0.5) * draw_w / n_bins)
                bar_h = int(count / max_count * draw_h) if max_count else 0
                x0 = x_center - bar_w // 2
                x1 = x_center + bar_w // 2
                y0 = PAD_T + draw_h - bar_h
                y1 = PAD_T + draw_h
                color = ACCENT if count > 0 else BG_HOVER
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

            if med is not None:
                x_med = PAD_L + int((med - p_min) / (p_max - p_min) * draw_w)
                x_med = max(PAD_L, min(W - PAD_R, x_med))
                canvas.create_line(x_med, PAD_T, x_med, PAD_T + draw_h,
                                   fill=YELLOW, dash=(4, 2), width=2)
                canvas.create_text(x_med, PAD_T - 4,
                                   text=f"median €{int(med)}", anchor="s",
                                   fill=YELLOW, font=("Menlo", 9, "bold"))

                if threshold:
                    cutoff = med * (1 - threshold / 100)
                    if cutoff >= p_min:
                        x_cut = PAD_L + int((cutoff - p_min) / (p_max - p_min) * draw_w)
                        x_cut = max(PAD_L, min(W - PAD_R, x_cut))
                        canvas.create_line(x_cut, PAD_T, x_cut, PAD_T + draw_h,
                                           fill=RED, dash=(2, 2), width=2)
                        canvas.create_text(x_cut, PAD_T + draw_h + 4,
                                           text=f"flag below €{int(cutoff)}",
                                           anchor="n", fill=RED,
                                           font=("Menlo", 9, "bold"))

            canvas.create_text(PAD_L, H - PAD_B + 14, text=f"€{int(p_min)}",
                               anchor="nw", fill=FG_DIM, font=F_MONO_S)
            canvas.create_text(W - PAD_R, H - PAD_B + 14, text=f"€{int(p_max)}",
                               anchor="ne", fill=FG_DIM, font=F_MONO_S)

        # Anomalies list
        tk.Label(parent, text=f"RECENT ANOMALIES  ({len(anomalies)})",
                 bg=BG, fg=FG_DIM, font=("SF Pro", 9, "bold")
                 ).pack(anchor="w", pady=(16, 4))

        if not anomalies:
            tk.Label(parent,
                     text="No anomalies flagged yet."
                          if threshold
                          else "Detection disabled — set a threshold in Settings to start flagging.",
                     bg=BG, fg=FG_DIM, font=F_SMALL).pack(anchor="w", pady=4)
            return

        # Scrollable table
        container = tk.Frame(parent, bg=BG)
        container.pack(fill="both", expand=True)

        canvas2 = tk.Canvas(container, bg=BG, highlightthickness=0)
        scr     = ttk.Scrollbar(container, orient="vertical", command=canvas2.yview)
        inner   = tk.Frame(canvas2, bg=BG)
        inner.bind("<Configure>", lambda e: canvas2.configure(scrollregion=canvas2.bbox("all")))
        canvas2.create_window((0, 0), window=inner, anchor="nw")
        canvas2.configure(yscrollcommand=scr.set)
        canvas2.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

        for col, (txt, w) in enumerate([("Date", 14), ("Price", 8),
                                         ("Discount", 9), ("vs Median", 10),
                                         ("Title", 50)]):
            tk.Label(inner, text=txt.upper(), bg=BG, fg=FG_DIM,
                     font=("SF Pro", 9, "bold"), width=w, anchor="w").grid(
                row=0, column=col, padx=8, pady=(0, 6), sticky="w")

        for i, item in enumerate(reversed(anomalies), start=1):
            date    = item.get("flagged_at", "")[:16]
            price_v = f"€{item.get('price', '?')}"
            med_v   = f"€{int(item.get('median', 0))}"
            disc    = f"-{item.get('discount_pct', 0)}%"
            title   = item.get("title", "")
            url     = item.get("url", "")

            tk.Label(inner, text=date, bg=BG, fg=FG_MUTED, font=F_MONO_S,
                     anchor="w").grid(row=i, column=0, padx=8, pady=2, sticky="w")
            tk.Label(inner, text=price_v, bg=BG, fg=GREEN, font=F_SMALL_B,
                     anchor="w").grid(row=i, column=1, padx=8, pady=2, sticky="w")
            tk.Label(inner, text=disc, bg=BG, fg=RED, font=F_SMALL_B,
                     anchor="w").grid(row=i, column=2, padx=8, pady=2, sticky="w")
            tk.Label(inner, text=med_v, bg=BG, fg=YELLOW, font=F_SMALL,
                     anchor="w").grid(row=i, column=3, padx=8, pady=2, sticky="w")

            title_lbl = tk.Label(inner, text=title, bg=BG, fg=FG,
                                 font=F_SMALL, anchor="w", cursor="hand2",
                                 wraplength=440)
            title_lbl.grid(row=i, column=4, padx=8, pady=2, sticky="w")
            if url:
                title_lbl.config(fg=ACCENT)
                title_lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
                title_lbl.bind("<Enter>",    lambda e, l=title_lbl: l.config(fg=WHITE))
                title_lbl.bind("<Leave>",    lambda e, l=title_lbl: l.config(fg=ACCENT))

    # ── Settings tab ──────────────────────────────────────────────────────────

    def _render_settings(self, parent, m: dict, cfg: dict):
        intro = tk.Frame(parent, bg=BG)
        intro.pack(fill="x", pady=(0, 8))
        tk.Label(intro, text="Edit search configuration",
                 bg=BG, fg=WHITE, font=F_H1).pack(anchor="w")
        tk.Label(intro, text="Changes are saved to the JSON file. Restart the monitor for them to take effect.",
                 bg=BG, fg=FG_MUTED, font=F_SMALL).pack(anchor="w", pady=(2, 0))

        # Scrollable form
        container = tk.Frame(parent, bg=BG)
        container.pack(fill="both", expand=True, pady=(10, 0))

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scr    = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_inner = tk.Frame(canvas, bg=BG)
        scroll_inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=scroll_inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.configure(yscrollcommand=scr.set)
        canvas.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

        form = MonitorForm(scroll_inner, existing_cfg=cfg, config_file=m["config_file"])
        form.pack(fill="both", expand=True, padx=4, pady=4)

        # Save bar
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(12, 0))

        def _on_save():
            result = form.save()
            if result is None:
                return
            new_cfg, _path, _ = result
            # Update label in monitors list and sidebar row
            for entry in self.monitors:
                if entry["id"] == m["id"]:
                    entry["label"] = new_cfg["label"]
                    break
            if m["id"] in self.cards:
                self.cards[m["id"]]["name_lbl"].config(text=new_cfg["label"])
            self._log(f"[{m['id'].upper()}] Config saved.", tag="info")

            if self._is_running(m["id"]):
                if messagebox.askyesno("Restart?",
                        f"'{new_cfg['label']}' is running.\nRestart to apply changes?",
                        parent=self):
                    self._restart_monitor(m["id"])
            self._render_detail(m["id"])

        FlatButton(bar, " Save changes ", color=BTN_PRIMARY, command=_on_save,
                   padx=14, pady=7, font=F_BODY_B).pack(side="right")

    # ── Monitor management ────────────────────────────────────────────────────

    def _open_new_monitor_dialog(self):
        AddMonitorDialog(self, on_save=self._on_new_monitor_created)

    def _on_new_monitor_created(self, m: dict):
        mid = m["id"]
        if mid in self.state:
            self._log(f"[{mid.upper()}] Existing config updated.", tag="info")
            return
        # Ensure the new monitor has a `type` key so filtering by tab works
        # consistently with load_monitors().
        m.setdefault("type", "text")
        self.monitors.append(m)
        self._init_state(mid)
        # Only add a sidebar row if it'd be visible in the current tab; the
        # next time the user switches tabs the sidebar is rebuilt from scratch
        # and will pick the monitor up via its filter_fn.
        if self.active_top_tab == "searches" and m["type"] == "text":
            self._build_sidebar_row(m)
        self._log(f"[{mid.upper()}] New search added: {m['label']}", tag="found")
        self._select_monitor(mid)

    def _delete_monitor(self, mid: str):
        m = next((x for x in self.monitors if x["id"] == mid), None)
        is_cat = bool(m) and m.get("type") == "category"
        kept = ("il DB condiviso è mantenuto" if is_cat
                else "data files are kept")
        if not messagebox.askyesno(
                "Elimina" if is_cat else "Delete search",
                f"Eliminare '{m['label'] if m else mid}'?\n"
                f"Il file di config viene rimosso ({kept}).",
                parent=self):
            return
        self._stop_monitor(mid)
        if m and m["config_file"].exists():
            m["config_file"].unlink()
        if mid in self.cards:
            self.cards[mid]["row"].destroy()
            del self.cards[mid]
        self.monitors = [x for x in self.monitors if x["id"] != mid]
        if mid in self.state:
            del self.state[mid]

        if self.selected_mid == mid:
            self.selected_mid = None
        # Rebuild the active tab so the sidebar drops the row and a valid monitor
        # of THIS tab is selected (not a text monitor while on Categorie).
        self._render_active_top_tab()
        self._refresh_sidebar()

    # ── Process management ────────────────────────────────────────────────────

    def _is_running(self, mid: str) -> bool:
        s = self.state[mid]
        if s["proc"] is not None and s["proc"].poll() is None:
            return True
        pid = find_pid(f"{mid}.json")
        s["ext_pid"] = pid
        return pid is not None

    def _monitor_launch_argv(self, m: dict, once: bool = False) -> list[str]:
        """Return the subprocess argv to launch a text- or category-monitor."""
        argv = [sys.executable, "-u"]
        if m.get("type") == "category":
            argv += ["-m", "core.category_monitor"]
        else:
            argv += [str(MONITOR_SCRIPT)]
        argv += ["--config", str(m["config_file"])]
        if once:
            argv += ["--once"]
        return argv

    def _start_monitor(self, mid: str):
        s = self.state[mid]
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None or not MONITOR_SCRIPT.exists():
            self._log(f"[{mid.upper()}] ✗ Script or config not found", tag="error")
            return
        try:
            proc = subprocess.Popen(
                self._monitor_launch_argv(m),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR),
            )
            s["proc"] = proc
            self._log(f"[{mid.upper()}] ▶ Started — running first scan... (PID {proc.pid})", tag="info")
            threading.Thread(target=self._stdout_reader, args=(mid, proc), daemon=True).start()
        except Exception as e:
            self._log(f"[{mid.upper()}] ✗ Failed to start: {e}", tag="error")

    def _stop_monitor(self, mid: str):
        s = self.state[mid]
        stopped = False
        if s["proc"] is not None:
            kill_pid(s["proc"].pid)
            s["proc"] = None
            stopped = True
        if s["ext_pid"] is not None:
            kill_pid(s["ext_pid"])
            s["ext_pid"] = None
            stopped = True
        if stopped:
            self._log(f"[{mid.upper()}] ■ Stopped", tag="muted")

    def _toggle_monitor(self, mid: str):
        if self._is_running(mid):
            self._stop_monitor(mid)
        else:
            self._start_monitor(mid)
        # Force an immediate re-render of the detail panel so the Start/Stop
        # button label flips right away, without waiting for the next periodic
        # refresh's state-change check. Without this, if kill_pid returned
        # before find_pid noticed the process was gone, _is_running could
        # still report True and the button would stay stuck on Stop.
        if self.selected_mid == mid and self.active_tab == "overview":
            self._render_detail(mid)
        self._refresh()

    def _restart_monitor(self, mid: str):
        self._stop_monitor(mid)
        time.sleep(0.4)
        self._start_monitor(mid)
        self._refresh()

    def _test_monitor(self, mid: str):
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            return
        self._log(f"[{mid.upper()}] ▶¹ Test run started...", tag="info")
        try:
            proc = subprocess.Popen(
                self._monitor_launch_argv(m, once=True),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR),
            )
            threading.Thread(target=self._stdout_reader, args=(mid, proc), daemon=True).start()
        except Exception as e:
            self._log(f"[{mid.upper()}] ✗ Test run failed: {e}", tag="error")

    def _regen_clusters(self, mid: str, cfg: dict):
        """Wipe + recompute cluster_centroids for one category DB, streaming
        progress into the live log panel. Runs in a worker thread because
        the embedding pass on a 10k-item DB takes ~10 min."""
        db_path = shared_db_path()
        if not db_path.exists():
            messagebox.showinfo("Rigenera cluster",
                                f"Il DB {db_path.name} non esiste ancora.",
                                parent=self)
            return
        if not messagebox.askyesno(
                "Rigenera cluster",
                f"Vuoi davvero rigenerare i cluster di '{cfg.get('label', mid)}'?\n\n"
                "L'operazione svuota cluster_centroids, riapplica embedding+traduzione\n"
                "a tutti gli item esistenti e può richiedere alcuni minuti.\n\n"
                "Un backup del DB viene creato automaticamente prima di iniziare.",
                parent=self):
            return

        label = mid.upper()
        self._log(f"[{label}] 🔄 Rigenerazione cluster avviata in background...",
                  tag="info")

        def emit(line: str) -> None:
            # Worker thread → marshal log lines back to the Tk main thread.
            self.after(0, self._log, f"[{label}] {line}", "default")

        def worker():
            try:
                # Lazy import — pulls in sentence-transformers/torch (~120MB
                # memory). No reason to pay that at GUI startup.
                from scripts.regenerate_clusters import regenerate
                stats = regenerate(db_path, log_fn=emit)
                self.after(0, self._log,
                           f"[{label}] ✓ Rigenerazione completata: "
                           f"{stats['clusters']} cluster · "
                           f"{stats['translated']} con label inglese", "found")
                # Re-render the dashboard so cluster table reflects new state
                self.after(0, self._refresh)
            except Exception as e:
                err = str(e)
                self.after(0, self._log,
                           f"[{label}] ✗ Rigenerazione fallita: {err}",
                           "error")

        threading.Thread(target=worker, daemon=True).start()

    def _reset_seen(self, mid: str):
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            return
        cfg = json.loads(m["config_file"].read_text())
        seen_path = DATA_DIR / cfg["seen_file"]
        if not messagebox.askyesno(
                "Reset seen IDs",
                f"Clear seen history for '{cfg['label']}'?\n"
                "The next scan will re-notify all matching listings.",
                parent=self):
            return
        if seen_path.exists():
            seen_path.write_text("[]")
        self._log(f"[{mid.upper()}] Seen IDs cleared", tag="warn")
        self._refresh()

    def _find_orphan_monitors(self) -> list[tuple[str, int]]:
        """Return (mid, pid) for monitor subprocesses NOT spawned by this GUI
        session but currently running on the system. Identified by config
        filename appearing in the process cmdline (same mechanism used by
        the regular Running indicator)."""
        out: list[tuple[str, int]] = []
        for m in self.monitors:
            mid = m["id"]
            s = self.state[mid]
            # If we started it this session, it's not an orphan.
            if s["proc"] is not None and s["proc"].poll() is None:
                continue
            pid = find_pid(f"{mid}.json")
            if pid is not None:
                out.append((mid, pid))
        return out

    def _prompt_cleanup_orphans(self):
        orphans = self._find_orphan_monitors()
        if not orphans:
            return
        names = ", ".join(f"{mid} (PID {pid})" for mid, pid in orphans)
        if messagebox.askyesno(
            "Monitor orfani trovati",
            f"Ho trovato {len(orphans)} monitor in esecuzione da una "
            f"sessione precedente:\n\n{names}\n\n"
            f"Vuoi fermarli ora? (Sì = pulisci, No = lasciali girare)",
            parent=self,
        ):
            for mid, pid in orphans:
                try:
                    kill_pid(pid)
                    self.state[mid]["ext_pid"] = None
                    self._log(f"[{mid.upper()}] ✗ Orfano fermato (PID {pid})",
                              "warn")
                except Exception as e:
                    self._log(f"[{mid.upper()}] ⚠ kill orfano fallito: {e}",
                              "error")
            self._refresh()
        else:
            self._log(
                f"Lasciati attivi {len(orphans)} monitor della sessione "
                f"precedente: {names}",
                "muted",
            )

    def _start_all(self):
        """Stagger monitor starts so Vinted's anti-bot doesn't flag the IP
        from a burst of N parallel cookie-fetch + catalog-scrape calls. When
        all monitors fire together the IP gets blocked even for the browser
        (~1-2 min) before relaxing. 8s spread keeps each first-cycle's HTTP
        traffic in its own window."""
        STAGGER_S = 8
        enabled  = [m["id"] for m in self.monitors
                    if self.state[m["id"]]["enabled"].get()]
        already  = [mid for mid in enabled if self._is_running(mid)]
        pending  = [mid for mid in enabled if mid not in already]

        if already:
            self._log(
                f"⏭  {len(already)} monitor già in esecuzione "
                f"(saltati): {', '.join(already)}",
                "muted",
            )
        if not pending:
            if not already:
                self._log("Nessun monitor abilitato da avviare", "warn")
            return

        self._log(
            f"▶ Avvio scaglionato di {len(pending)} monitor "
            f"({STAGGER_S}s tra uno e l'altro): {', '.join(pending)}",
            "info",
        )
        for i, mid in enumerate(pending):
            # default-arg captures mid; .after schedules in Tk main loop
            self.after(i * STAGGER_S * 1000,
                       lambda m=mid: self._start_monitor(m))
        # Refresh once after the last scheduled start has fired
        self.after(len(pending) * STAGGER_S * 1000 + 500, self._refresh)

    def _stop_all(self):
        for m in self.monitors:
            self._stop_monitor(m["id"])
        self._refresh()

    # ── Stdout reader / log ───────────────────────────────────────────────────

    def _stdout_reader(self, mid: str, proc):
        label = mid.upper()
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                ll = line.lower()

                url = None
                display_line = line
                if " | https://" in line:
                    parts = line.rsplit(" | ", 1)
                    display_line = parts[0]
                    url = parts[1].strip()

                if any(x in ll for x in ("new listings found", "→", "telegram sent")):
                    tag = "found"
                elif "no new listings" in ll:
                    tag = "muted"
                elif "no internet" in ll:
                    tag = "warn"
                elif any(x in ll for x in ("connection restored", "immediate scan")):
                    tag = "info"
                elif line.lstrip().startswith("✓"):
                    tag = "found"
                elif line.lstrip().startswith("✗") or any(
                        x in ll for x in ("errore", "error", "exception",
                                          "traceback", "cannot")):
                    tag = "error"
                else:
                    tag = "default"

                if url:
                    self.after(0, self._log_link, f"[{label}] {display_line}", tag, url)
                else:
                    self.after(0, self._log, f"[{label}] {display_line}", tag)
        except Exception:
            pass

    def _log(self, msg: str, tag: str = "default"):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{ts}  {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _log_link(self, msg: str, tag: str, url: str):
        ts       = time.strftime("%H:%M:%S")
        link_tag = f"link_{self._link_counter}"
        self._link_counter += 1
        self._link_map[link_tag] = url
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{ts}  {msg}  ↗\n", (tag, link_tag))
        self.log_box.tag_config(link_tag, underline=True)
        self.log_box.tag_bind(link_tag, "<Button-1>",
                              lambda e, u=url: webbrowser.open(u))
        self.log_box.tag_bind(link_tag, "<Enter>",
                              lambda e: self.log_box.config(cursor="hand2"))
        self.log_box.tag_bind(link_tag, "<Leave>",
                              lambda e: self.log_box.config(cursor=""))
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ── Refresh loop ──────────────────────────────────────────────────────────

    def _refresh(self):
        net_ok, net_msg = check_network()
        self.net_dot.config(fg=GREEN if net_ok else RED)
        self.net_label.config(text=net_msg, fg=FG if net_ok else RED)
        self._refresh_sidebar()
        # Re-render Overview only when the monitor's running state changes —
        # otherwise the action buttons get destroyed every 4s and the user
        # can't click them in peace. Settings/History/Stats tabs never
        # auto-refresh (would lose form state / scroll position).
        if self.selected_mid and self.active_tab == "overview":
            running_now = self._is_running(self.selected_mid)
            last = self._last_running_state.get(self.selected_mid)
            self._last_running_state[self.selected_mid] = running_now
            if last is not None and last != running_now:
                self._render_detail(self.selected_mid)

        # Category dashboard: re-render when the DB gained items (a scrape landed)
        # so new listings show up without manually switching tabs. Gated on a
        # cheap row-count delta so it doesn't flicker during the rest of the cycle
        # (sold/removed updates are UPDATEs, they don't change the row count).
        if self.active_top_tab == "categories" and self.selected_mid:
            m = next((x for x in self.monitors if x["id"] == self.selected_mid), None)
            if m and m.get("type") == "category":
                cnt  = self._category_item_count(self.selected_mid)
                prev = self._last_cat_count.get(self.selected_mid)
                self._last_cat_count[self.selected_mid] = cnt
                if prev is not None and cnt is not None and cnt != prev:
                    self._render_detail(self.selected_mid)

    def _category_item_count(self, cfg_id: str):
        """Cheap read-only row count of a category DB (no schema/migrate), used to
        detect when a scrape added items so the dashboard can auto-refresh."""
        import sqlite3
        db = shared_db_path()
        if not db.exists():
            return None
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            try:
                return c.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            finally:
                c.close()
        except Exception:
            return None

    def _refresh_sidebar(self):
        # Count only monitors visible in the currently-rendered sidebar
        # (self.cards is rebuilt by _build_sidebar with the active filter).
        if hasattr(self, "sidebar_count") and self.sidebar_count.winfo_exists():
            self.sidebar_count.config(text=str(len(self.cards)))
        for m in self.monitors:
            mid  = m["id"]
            card = self.cards.get(mid)
            if card is None:
                continue
            running = self._is_running(mid)

            if running:
                dot_color  = GREEN
                sub_text   = "Running"
                sub_color  = GREEN
            else:
                s = self.state[mid]
                if s["proc"] is not None and s["proc"].poll() is not None:
                    ret = s["proc"].poll()
                    s["proc"] = None
                    dot_color  = RED
                    sub_text   = f"Crashed (exit {ret})"
                    sub_color  = RED
                    self._log(f"[{mid.upper()}] ✗ Process exited with code {ret}", tag="error")
                else:
                    dot_color  = FG_DIM
                    sub_text   = "Inactive"
                    sub_color  = FG_DIM

            # Add quick stat to sub line
            st = _read_stats(mid)
            last = _time_ago(st.get("last_found_at"))
            sub_full = f"{sub_text} · last {last}"

            card["dot"].config(fg=dot_color)
            card["sub_lbl"].config(text=sub_full, fg=sub_color)

            # Selected highlight
            selected = (mid == self.selected_mid)
            bg = BG_HOVER if selected else BG_SIDEBAR
            for w in (card["row"], card["inner"], card["info"],
                      card["dot"], card["name_lbl"], card["sub_lbl"], card["chk"]):
                w.config(bg=bg)
            card["name_lbl"].config(fg=WHITE if selected else FG)

    def _schedule_refresh(self):
        self._refresh()
        self.after(4000, self._schedule_refresh)

    # ── Charts (global) ───────────────────────────────────────────────────────

    def _show_charts(self):
        today = datetime.date.today()
        days  = [(today - datetime.timedelta(days=i)).isoformat()
                 for i in range(13, -1, -1)]
        labels = [d[5:] for d in days]

        # Only text monitors have a history_*.json file. Category monitors
        # live in SQLite and have their own dashboard inside the Categorie tab.
        text_monitors = [m for m in self.monitors if m.get("type", "text") == "text"]
        if not text_monitors:
            messagebox.showinfo(
                "Charts",
                "Nessuna ricerca text-based ancora — il grafico è specifico per i monitor "
                "delle Ricerche. I monitor categoria hanno la loro dashboard nella tab 📊 Categorie.",
                parent=self,
            )
            return

        all_counts = {}
        for m in text_monitors:
            counts: dict[str, int] = defaultdict(int)
            hp = DATA_DIR / f"history_{m['id']}.json"
            if hp.exists():
                try:
                    for item in json.loads(hp.read_text()):
                        d = item.get("notified_at", "")[:10]
                        if d in days:
                            counts[d] += 1
                except Exception:
                    pass
            all_counts[m["id"]] = [counts.get(d, 0) for d in days]

        win = tk.Toplevel(self)
        win.title("Charts — Items found per day")
        win.configure(bg=BG)
        win.resizable(False, False)

        tk.Label(win, text="Items notified per day (last 14 days)",
                 bg=BG, fg=WHITE, font=F_H1).pack(anchor="w", padx=20, pady=(16, 4))
        tk.Label(win, text="One panel per active search.",
                 bg=BG, fg=FG_MUTED, font=F_SMALL).pack(anchor="w", padx=20, pady=(0, 12))

        PAD_L, PAD_R, PAD_T, PAD_B = 48, 16, 30, 28
        W, H   = 820, 180
        n_days = len(days)

        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        for m in text_monitors:
            values  = all_counts[m["id"]]
            max_val = max(values) if any(values) else 1

            frame = tk.Frame(body, bg=BG, pady=6)
            frame.pack(fill="x")
            tk.Label(frame, text=m["label"], bg=BG, fg=FG,
                     font=F_BODY_B).pack(anchor="w")

            c = tk.Canvas(frame, width=W, height=H, bg=BG_PANEL,
                          highlightthickness=0)
            c.pack()

            draw_w = W - PAD_L - PAD_R
            draw_h = H - PAD_T - PAD_B
            bar_w  = max(4, draw_w // n_days - 2)

            for step in range(0, max_val + 1):
                y = PAD_T + draw_h - int(step / max_val * draw_h)
                c.create_line(PAD_L - 4, y, W - PAD_R, y, fill=BORDER)
                c.create_text(PAD_L - 6, y, text=str(step),
                              anchor="e", fill=FG_DIM, font=("Menlo", 8))

            for i, (val, lbl) in enumerate(zip(values, labels)):
                x_center = PAD_L + int((i + 0.5) * draw_w / n_days)
                bar_h    = int(val / max_val * draw_h) if max_val else 0
                x0 = x_center - bar_w // 2
                x1 = x_center + bar_w // 2
                y0 = PAD_T + draw_h - bar_h
                y1 = PAD_T + draw_h
                color = ACCENT if val > 0 else BG_HOVER
                c.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                if val > 0:
                    c.create_text(x_center, y0 - 4, text=str(val),
                                  anchor="s", fill=FG, font=("Menlo", 8))
                if i % 2 == 0:
                    c.create_text(x_center, H - PAD_B + 6, text=lbl,
                                  anchor="n", fill=FG_DIM, font=("Menlo", 8))

        win.geometry(f"848x{(H + 50) * max(1, len(text_monitors)) + 90}")

    # ── Explore tab (global category browser) ─────────────────────────────────
    # _render_explore is invoked by _render_explore_tab (top-level tab).

    def _render_explore(self):
        self._clear_detail()

        # Header
        hdr = tk.Frame(self.detail_panel, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 6))
        tk.Label(hdr, text="🔥  Esplora occasioni",
                 bg=BG, fg=WHITE, font=("SF Pro Display", 18, "bold")
                 ).pack(anchor="w")
        tk.Label(hdr,
                 text="Sfoglia una macro-categoria di Vinted e ordina per ♥ like — più like = più occhi addosso = potenziale occasione.",
                 bg=BG, fg=FG_MUTED, font=F_SMALL, wraplength=720, justify="left"
                 ).pack(anchor="w", pady=(4, 0))

        # Form row
        form = tk.Frame(self.detail_panel, bg=BG_PANEL)
        form.pack(fill="x", padx=24, pady=(14, 0), ipady=10)

        grid = tk.Frame(form, bg=BG_PANEL)
        grid.pack(padx=14, pady=6, anchor="w")

        def lbl(txt, r, c):
            tk.Label(grid, text=txt, bg=BG_PANEL, fg=FG_MUTED,
                     font=F_SMALL).grid(row=r, column=c, sticky="w",
                                        padx=(0, 6), pady=(0, 2))

        # Category
        lbl("Categoria", 0, 0)
        cat_var = tk.StringVar(value=list(CATEGORIES.keys())[0])
        cat_dd = ttk.Combobox(grid, textvariable=cat_var,
                              values=list(CATEGORIES.keys()),
                              state="readonly", width=18, font=F_SMALL)
        cat_dd.grid(row=1, column=0, padx=(0, 12))

        # Max price
        lbl("Max € (0 = nessun limite)", 0, 1)
        price_var = tk.StringVar(value="0")
        tk.Entry(grid, textvariable=price_var, width=10, bg=BG_INPUT,
                 fg=FG, insertbackground=FG, relief="flat",
                 font=F_SMALL).grid(row=1, column=1, padx=(0, 12), ipady=4)

        # Min likes — defaults to 0 because the vast majority of Vinted listings
        # have ♥0–1; a higher default would filter everything out. The Top-K
        # cap + descending sort still surfaces the most-liked items first.
        lbl("Min ♥ like", 0, 2)
        likes_var = tk.StringVar(value="0")
        tk.Entry(grid, textvariable=likes_var, width=8, bg=BG_INPUT,
                 fg=FG, insertbackground=FG, relief="flat",
                 font=F_SMALL).grid(row=1, column=2, padx=(0, 12), ipady=4)

        # Top K
        lbl("Top K risultati", 0, 3)
        topk_var = tk.StringVar(value="30")
        tk.Entry(grid, textvariable=topk_var, width=8, bg=BG_INPUT,
                 fg=FG, insertbackground=FG, relief="flat",
                 font=F_SMALL).grid(row=1, column=3, padx=(0, 12), ipady=4)

        # Pages per domain
        lbl("Pagine per dominio (1-5)", 0, 4)
        pages_var = tk.StringVar(value="1")
        tk.Entry(grid, textvariable=pages_var, width=6, bg=BG_INPUT,
                 fg=FG, insertbackground=FG, relief="flat",
                 font=F_SMALL).grid(row=1, column=4, padx=(0, 12), ipady=4)

        # Search button + status
        action = tk.Frame(form, bg=BG_PANEL)
        action.pack(padx=14, pady=(8, 0), anchor="w", fill="x")
        search_btn = FlatButton(action, " 🔍  Cerca occasioni ",
                                color=BTN_PRIMARY, padx=14, pady=6,
                                font=F_SMALL_B)
        search_btn.pack(side="left")
        status_lbl = tk.Label(action, text="", bg=BG_PANEL, fg=FG_MUTED,
                              font=F_SMALL)
        status_lbl.pack(side="left", padx=12)

        # Results table
        body = tk.Frame(self.detail_panel, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(14, 18))

        cols = ("likes", "price", "vs_med", "brand", "title")
        tree = ttk.Treeview(body, columns=cols, show="headings", height=20)
        tree.heading("likes",  text="♥")
        tree.heading("price",  text="€")
        tree.heading("vs_med", text="vs mediana (vnd=venduti · att=attivi)")
        tree.heading("brand",  text="Brand")
        tree.heading("title",  text="Titolo (doppio click per aprire)")
        tree.column("likes",  width=60,  anchor="center", stretch=False)
        tree.column("price",  width=80,  anchor="e",      stretch=False)
        tree.column("vs_med", width=210, anchor="w",      stretch=False)
        tree.column("brand",  width=130, anchor="w",      stretch=False)
        tree.column("title",  width=420, anchor="w",      stretch=True)

        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # url_map: tree row iid → item url
        url_map: dict[str, str] = {}

        def on_open(_event=None):
            sel = tree.selection()
            if not sel:
                return
            url = url_map.get(sel[0])
            if url:
                webbrowser.open(url)
        tree.bind("<Double-1>", on_open)
        tree.bind("<Return>",   on_open)

        # Run handler
        def run_search():
            try:
                max_price = float(price_var.get() or 0) or None
                min_fav   = int(likes_var.get() or 0)
                top_k     = max(1, int(topk_var.get() or 30))
                pages     = max(1, min(5, int(pages_var.get() or 1)))
            except ValueError:
                messagebox.showerror("Input non valido",
                                     "Controlla i valori numerici.")
                return
            cat_name = cat_var.get()
            cat_ids  = CATEGORIES[cat_name]

            tree.delete(*tree.get_children())
            url_map.clear()
            status_lbl.config(text=f"Cerco in {cat_name}…", fg=FG_MUTED)
            search_btn.config(text=" … ricerca in corso ", bg=BG_HOVER)

            self._log(
                f"[ESPLORA] 🔥 {cat_name} — cat_ids={cat_ids} · max€{max_price or '∞'} · "
                f"min♥{min_fav} · top{top_k} · {pages}pag/dominio",
                tag="info",
            )

            # Discovery runs in a worker thread; Tkinter is not thread-safe so
            # every UI / log update is bounced back to the main thread via
            # self.after(0, ...).
            def progress_cb(domain, ok, error=None):
                short = domain.replace("https://www.", "")
                def emit():
                    # Log panel is global so it's still alive across tab
                    # switches, but be defensive in case it's been destroyed
                    # (e.g. window closing).
                    if not self.log_box.winfo_exists():
                        return
                    if ok:
                        self._log(f"[ESPLORA] ✓ {short}", tag="found")
                    else:
                        self._log(f"[ESPLORA] ✗ {short}: {error}", tag="error")
                self.after(0, emit)

            def worker():
                t0 = time.time()
                try:
                    items, errors = discover(
                        catalog_ids       = cat_ids,
                        domains           = DOMAINS_DEFAULT,
                        max_price         = max_price,
                        min_favourites    = min_fav,
                        top_k             = top_k,
                        pages_per_domain  = pages,
                        progress_cb       = progress_cb,
                    )
                    # Enrich with price-vs-median when a matching category DB
                    # exists. Failures here are non-fatal: we still show the
                    # items, just without the comparison column.
                    db_path = find_category_db(cat_ids)
                    if db_path and items:
                        try:
                            mstats = enrich_with_median(items, db_path)
                            self.after(0, lambda: self._log(
                                f"[ESPLORA] 📊 confronto {db_path.name}: "
                                f"{mstats['matched']} mediana · "
                                f"{mstats['no_cluster']} non in DB · "
                                f"{mstats['thin_sample']} campione scarso",
                                tag="info",
                            ))
                        except Exception as e:
                            err = str(e)
                            self.after(0, lambda: self._log(
                                f"[ESPLORA] ⚠ confronto mediana fallito: {err}",
                                tag="warn",
                            ))
                    elif items:
                        self.after(0, lambda: self._log(
                            "[ESPLORA] ⚠ nessun DB categoria — niente confronto mediana",
                            tag="warn",
                        ))
                    elapsed = time.time() - t0
                    self.after(0, lambda: populate(items, errors, elapsed))
                except Exception as e:
                    msg = str(e)
                    self.after(0, lambda: fail(msg))

            def populate(items, errors, elapsed):
                # The worker thread can complete after the user has switched
                # tabs and destroyed all explore widgets — guard against that
                # before touching anything.
                if not tree.winfo_exists():
                    return
                pool_total   = int(errors.pop("_pool_total",   "0") or 0)
                filtered_out = int(errors.pop("_filtered_out", "0") or 0)
                for it in items:
                    iid = tree.insert("", "end", values=(
                        f"♥{it['favourite_count']}",
                        f"€{it['price']:.0f}",
                        it.get("price_label", "—"),
                        it["brand"] or "—",
                        it["title"],
                    ))
                    url_map[iid] = it["url"]
                parts = [f"{len(items)} risultati", f"pool {pool_total}"]
                if filtered_out:
                    parts.append(f"{filtered_out} esclusi da filtro ♥")
                if errors:
                    parts.append(f"{len(errors)} domini falliti")
                status_text = " · ".join(parts) + f"  ({elapsed:.1f}s)"
                status_lbl.config(
                    text=status_text,
                    fg=GREEN if items else YELLOW,
                )
                search_btn.config(text=" 🔍  Cerca occasioni ", bg=BTN_PRIMARY)
                tag = "found" if items else "warn"
                self._log(f"[ESPLORA] {status_text}", tag=tag)

            def fail(msg):
                if not tree.winfo_exists():
                    return
                status_lbl.config(text=f"Errore: {msg}", fg=RED)
                search_btn.config(text=" 🔍  Cerca occasioni ", bg=BTN_PRIMARY)
                self._log(f"[ESPLORA] ✗ Errore: {msg}", tag="error")

            threading.Thread(target=worker, daemon=True).start()

        search_btn.config(command=run_search)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = VintedGUI()
    app.mainloop()
