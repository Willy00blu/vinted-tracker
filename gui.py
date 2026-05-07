#!/usr/bin/env python3
"""
Vinted Monitor Manager — GUI

Launch this file to open the control panel:
    python gui.py

From here you can:
  - See at a glance which monitors are running
  - Start / stop / restart individual monitors
  - Add a new monitor for anything via the "+ Aggiungi" button
  - Watch live log output from each running monitor

Monitors are auto-discovered from the configs/ folder.
Each JSON file there represents one item category to track on Vinted.
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
from pathlib import Path

import datetime
from collections import defaultdict

# ── psutil opzionale (fallback nativo su macOS) ───────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Costanti ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
CONFIGS_DIR    = BASE_DIR / "configs"
DATA_DIR       = BASE_DIR / "data"
MONITOR_SCRIPT = BASE_DIR / "monitor.py"

# Colori
BG        = "#1e1e2e"
BG2       = "#2a2a3e"
BG3       = "#313145"
FG        = "#cdd6f4"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
YELLOW    = "#f9e2af"
BLUE      = "#89b4fa"
GRAY      = "#6c7086"
WHITE     = "#ffffff"
BTN_START = "#40a02b"
BTN_STOP  = "#d20f39"
BTN_RST   = "#7287fd"
BTN_ADD   = "#7287fd"


# ── Flat button (Label-based) ─────────────────────────────────────────────────
# macOS Tkinter ignores bg on tk.Button (uses native rendering).
# Using tk.Label with click bindings is the reliable cross-platform workaround.

class FlatButton(tk.Label):
    def __init__(self, parent, text, color, command,
                 padx=10, pady=4, font=("SF Pro", 11), **kw):
        super().__init__(parent, text=text, bg=color, fg=WHITE,
                         font=font, padx=padx, pady=pady,
                         cursor="hand2", **kw)
        self._color = color
        self._cmd   = command
        self.bind("<Button-1>", lambda _: self._cmd())
        self.bind("<Enter>",    lambda _: super(FlatButton, self).config(bg=self._dim()))
        self.bind("<Leave>",    lambda _: super(FlatButton, self).config(bg=self._color))

    def _dim(self):
        """Return a slightly darker shade of the current color for hover."""
        r = max(0, int(self._color[1:3], 16) - 20)
        g = max(0, int(self._color[3:5], 16) - 20)
        b = max(0, int(self._color[5:7], 16) - 20)
        return f"#{r:02x}{g:02x}{b:02x}"

    def config(self, **kw):
        if "bg" in kw:
            self._color = kw["bg"]
        if "command" in kw:
            self._cmd = kw.pop("command")
        super().config(**kw)

    configure = config


def load_monitors() -> list[dict]:
    monitors = []
    for cfg_file in sorted(CONFIGS_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_file.read_text())
            monitors.append({
                "id":          cfg["id"],
                "label":       cfg["label"],
                "config_file": cfg_file,
            })
        except Exception:
            pass
    return monitors


# ── Rete ──────────────────────────────────────────────────────────────────────

def check_network() -> tuple[bool, str]:
    for url in ("https://www.google.com", "https://www.apple.com"):
        try:
            urllib.request.urlopen(url, timeout=4)
            return True, "Connected"
        except Exception:
            pass
    for host, port in [("8.8.8.8", 53), ("1.1.1.1", 53)]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            return True, "Connected"
        except OSError:
            pass
    return False, "No connection"


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
    if HAS_PSUTIL:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=4)
            except psutil.TimeoutExpired:
                p.kill()
        except Exception:
            pass
        return
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except Exception:
        pass


# ── Dialog aggiunta / modifica ricerca ───────────────────────────────────────

class MonitorDialog(tk.Toplevel):
    """
    Modal form used both to CREATE a new search config and to EDIT an existing one.

    Pass `existing_cfg` (the parsed JSON dict) and `config_file` (Path) to open
    in edit mode — all fields are pre-filled and saving overwrites the same file.
    """

    FIELDS = [
        ("label",             "Name",                        "entry", "e.g. Gaming PC, Mechanical Keyboard"),
        ("min_price",         "Min price (€)",               "entry", "e.g. 10"),
        ("max_price",         "Max price (€)",               "entry", "e.g. 500"),
        ("check_interval",    "Check every (minutes)",       "entry", "e.g. 15"),
        ("queries",           "Search queries",              "text",  "one per line\ne.g. gaming pc rtx\ndesktop gaming"),
        ("keywords_required", "Required keywords",           "entry", "comma-separated, e.g. rtx,gaming,i7"),
        ("brands_required",   "Required brands (optional)", "entry", "comma-separated, e.g. asus,msi,lenovo"),
        ("keywords_exclude",  "Exclude keywords",            "entry", "comma-separated, e.g. broken,spare parts"),
    ]

    def __init__(self, parent, on_save, existing_cfg: dict | None = None, config_file=None):
        super().__init__(parent)
        self.on_save      = on_save
        self._existing    = existing_cfg   # None → create mode, dict → edit mode
        self._config_file = config_file    # Path of the file to overwrite when editing
        self._editing     = existing_cfg is not None
        self._widgets: dict[str, tk.Widget] = {}

        self.title("Edit search" if self._editing else "New search")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()

        self._build()
        if self._editing:
            self._prefill(existing_cfg)

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width()  - self.winfo_width())  // 2
        py = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        title_text = "Edit search" if self._editing else "New search"
        tk.Label(self, text=title_text, bg=BG, fg=WHITE,
                 font=("SF Pro Display", 15, "bold")).pack(padx=20, pady=(16, 12), anchor="w")

        form = tk.Frame(self, bg=BG)
        form.pack(fill="x", padx=20)

        for key, lbl, kind, placeholder in self.FIELDS:
            tk.Label(form, text=lbl, bg=BG, fg=GRAY,
                     font=("SF Pro", 10)).pack(anchor="w", pady=(8, 2))

            if kind == "entry":
                var = tk.StringVar()
                e = tk.Entry(form, textvariable=var, bg=BG3, fg=GRAY,
                             insertbackground=FG, relief="flat",
                             font=("SF Pro", 11), width=46,
                             selectbackground=BLUE, selectforeground=BG)
                e.pack(fill="x", ipady=5)
                e.insert(0, placeholder)
                e.bind("<FocusIn>",  lambda ev, w=e, ph=placeholder: self._clear_ph(ev, w, ph))
                e.bind("<FocusOut>", lambda ev, w=e, ph=placeholder, v=var: self._restore_ph(ev, w, ph, v))
                self._widgets[key] = (e, var, placeholder)
            else:
                t = tk.Text(form, bg=BG3, fg=GRAY, insertbackground=FG,
                            relief="flat", font=("SF Pro", 11), width=46, height=4,
                            selectbackground=BLUE, selectforeground=BG)
                t.pack(fill="x")
                t.insert("1.0", placeholder)
                t.bind("<FocusIn>",  lambda ev, w=t, ph=placeholder: self._clear_ph_text(ev, w, ph))
                t.bind("<FocusOut>", lambda ev, w=t, ph=placeholder: self._restore_ph_text(ev, w, ph))
                self._widgets[key] = t

        btns = tk.Frame(self, bg=BG)
        btns.pack(fill="x", padx=20, pady=(16, 20))
        save_text = "Save changes" if self._editing else "Add"
        FlatButton(btns, text=save_text, color=BTN_ADD,
                   command=self._save, padx=12, pady=5,
                   font=("SF Pro", 11, "bold")).pack(side="right")
        FlatButton(btns, text="Cancel", color=BG3,
                   command=self.destroy, padx=12, pady=5,
                   font=("SF Pro", 11)).pack(side="right", padx=(0, 8))

    # ── Pre-fill (edit mode) ──────────────────────────────────────────────────

    def _prefill(self, cfg: dict):
        """Populate all fields with values from an existing config."""
        def set_entry(key, value):
            e, var, _ = self._widgets[key]
            e.delete(0, "end")
            e.insert(0, str(value))
            e.config(fg=FG)

        def set_text(key, value):
            t = self._widgets[key]
            t.delete("1.0", "end")
            t.insert("1.0", value)
            t.config(fg=FG)

        set_entry("label",             cfg.get("label", ""))
        set_entry("min_price",         cfg.get("min_price", ""))
        set_entry("max_price",         cfg.get("max_price", ""))
        set_entry("check_interval",    cfg.get("check_interval_minutes", 15))
        set_text ("queries",           "\n".join(cfg.get("queries", [])))
        set_entry("keywords_required", ", ".join(cfg.get("keywords_required", [])))
        set_entry("brands_required",   ", ".join(cfg.get("brands_required", [])))
        set_entry("keywords_exclude",  ", ".join(cfg.get("keywords_exclude", [])))

    # ── Placeholder helpers ───────────────────────────────────────────────────

    def _clear_ph(self, _, widget, placeholder):
        if widget.get() == placeholder:
            widget.delete(0, "end")
            widget.config(fg=FG)

    def _restore_ph(self, _, widget, placeholder, var):
        if not widget.get().strip():
            widget.delete(0, "end")
            widget.insert(0, placeholder)
            widget.config(fg=GRAY)

    def _clear_ph_text(self, _, widget, placeholder):
        if widget.get("1.0", "end-1c") == placeholder:
            widget.delete("1.0", "end")
            widget.config(fg=FG)

    def _restore_ph_text(self, _, widget, placeholder):
        if not widget.get("1.0", "end-1c").strip():
            widget.delete("1.0", "end")
            widget.insert("1.0", placeholder)
            widget.config(fg=GRAY)

    # ── Read values ───────────────────────────────────────────────────────────

    def _get(self, key: str) -> str:
        w = self._widgets[key]
        if isinstance(w, tk.Text):
            return w.get("1.0", "end-1c").strip()
        entry, _, placeholder = w
        val = entry.get().strip()
        return "" if val == placeholder else val

    def _parse_csv(self, raw: str) -> list[str]:
        return [x.strip().lower() for x in re.split(r"[,\n]", raw) if x.strip()]

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self):
        label = self._get("label")
        if not label:
            messagebox.showerror("Errore", "Inserisci un nome.", parent=self)
            return

        queries = [q.strip() for q in self._get("queries").splitlines()
                   if q.strip() and not q.startswith("one per line") and not q.startswith("e.g.")]
        if not queries:
            messagebox.showerror("Errore", "Inserisci almeno una query di ricerca.", parent=self)
            return

        try:
            min_price = int(self._get("min_price") or 0)
        except ValueError:
            min_price = 0

        try:
            max_price = int(self._get("max_price") or 9999)
        except ValueError:
            messagebox.showerror("Errore", "Prezzo massimo deve essere un numero.", parent=self)
            return

        try:
            interval = int(self._get("check_interval") or 15)
        except ValueError:
            interval = 15

        if self._editing:
            # Keep original id/seen_file so we don't lose history
            mid        = self._existing["id"]
            seen_file  = self._existing["seen_file"]
            cfg_path   = self._config_file
        else:
            mid        = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            seen_file  = f"seen_{mid}_ids.json"
            cfg_path   = CONFIGS_DIR / f"{mid}.json"

        cfg = {
            "id":                     mid,
            "label":                  label,
            "min_price":              min_price,
            "max_price":              max_price,
            "check_interval_minutes": interval,
            "seen_file":              seen_file,
            "price_db_file":          self._existing.get("price_db_file") if self._editing else None,
            "domains": [
                "https://www.vinted.it",
                "https://www.vinted.fr",
                "https://www.vinted.de",
                "https://www.vinted.com",
            ],
            "queries":           queries,
            "keywords_required": self._parse_csv(self._get("keywords_required")),
            "brands_required":   self._parse_csv(self._get("brands_required")),
            "keywords_exclude":  self._parse_csv(self._get("keywords_exclude")),
            "email_subject":     f"🔔 {{count}} new listing(s) on Vinted — {label}!",
            "email_intro":       f"New listings found on Vinted for: {label}",
        }

        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        self.destroy()
        self.on_save({"id": mid, "label": label, "config_file": cfg_path}, self._editing)


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _read_seen_count(cfg_id: str, seen_file: str) -> int:
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


def _time_ago(ts: str | None) -> str:
    if not ts:
        return "never"
    try:
        dt   = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        diff = int(time.time() - dt)
        if diff < 60:     return "just now"
        if diff < 3600:   return f"{diff // 60}m ago"
        if diff < 86400:  return f"{diff // 3600}h ago"
        return f"{diff // 86400}d ago"
    except Exception:
        return ts


# ── App principale ─────────────────────────────────────────────────────────────

class VintedGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Vinted Tracker")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.monitors    = load_monitors()
        self.state: dict[str, dict] = {}
        self._link_counter = 0
        self._link_map: dict[str, str] = {}  # tag → url
        for m in self.monitors:
            self._init_state(m["id"])

        self._build_ui()
        self._refresh()
        self._schedule_refresh()

    def _init_state(self, mid: str):
        self.state[mid] = {
            "proc":    None,
            "ext_pid": None,
            "enabled": tk.BooleanVar(value=True),
        }

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(hdr, text="Vinted Tracker", bg=BG, fg=WHITE,
                 font=("SF Pro Display", 17, "bold")).pack(side="left")
        # Pulsanti header
        FlatButton(hdr, text=" + Add ", color=BTN_ADD,
                   command=self._open_new_monitor_dialog,
                   padx=8, pady=3,
                   font=("SF Pro", 11, "bold")).pack(side="right")
        FlatButton(hdr, text=" 📊 Charts ", color=BG3,
                   command=self._show_charts,
                   padx=8, pady=3,
                   font=("SF Pro", 11)).pack(side="right", padx=(0, 6))

        # Rete
        net_frame = tk.Frame(self, bg=BG2)
        net_frame.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(net_frame, text="  Network:", bg=BG2, fg=GRAY,
                 font=("SF Pro", 11)).pack(side="left", pady=6)
        self.net_dot   = tk.Label(net_frame, text="●", bg=BG2, fg=GRAY, font=("SF Pro", 12))
        self.net_dot.pack(side="left", padx=4)
        self.net_label = tk.Label(net_frame, text="…", bg=BG2, fg=GRAY, font=("SF Pro", 11))
        self.net_label.pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=14, pady=4)

        # Contenitore card (permette aggiunta dinamica)
        self.cards_frame = tk.Frame(self, bg=BG)
        self.cards_frame.pack(fill="x")

        self.cards: dict[str, dict] = {}
        for m in self.monitors:
            self._build_monitor_card(m)

        self._sep_bottom = ttk.Separator(self, orient="horizontal")
        self._sep_bottom.pack(fill="x", padx=14, pady=4)

        # Pulsanti globali
        glob = tk.Frame(self, bg=BG)
        glob.pack(fill="x", padx=14, pady=(4, 6))
        self._btn(glob, "▶  Start enabled", BTN_START, self._start_all).pack(side="left", padx=(0, 8))
        self._btn(glob, "■  Stop all",      BTN_STOP,  self._stop_all).pack(side="left", padx=(0, 8))
        self._btn(glob, "↺  Refresh",       BG3,       self._refresh).pack(side="right")

        # Log
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=14, pady=4)
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="x", padx=14, pady=(0, 14))
        tk.Label(log_frame, text="Activity log", bg=BG, fg=GRAY,
                 font=("SF Pro", 10, "bold")).pack(anchor="w")
        self.log_box = tk.Text(log_frame, height=10, bg=BG2, fg=FG,
                               font=("Menlo", 10), relief="flat",
                               state="disabled", wrap="word", bd=0,
                               selectbackground=BLUE, selectforeground=BG)
        self.log_box.pack(fill="x")
        sb = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.tag_config("found",   foreground=GREEN)
        self.log_box.tag_config("info",    foreground=BLUE)
        self.log_box.tag_config("warn",    foreground=YELLOW)
        self.log_box.tag_config("error",   foreground=RED)
        self.log_box.tag_config("muted",   foreground=GRAY)
        self.log_box.tag_config("default", foreground=FG)

    def _build_monitor_card(self, m: dict):
        mid = m["id"]
        card = tk.Frame(self.cards_frame, bg=BG2)
        card.pack(fill="x", padx=14, pady=4)

        left = tk.Frame(card, bg=BG2)
        left.pack(side="left", padx=10, pady=10, fill="x", expand=True)

        dot = tk.Label(left, text="●", bg=BG2, fg=GRAY, font=("SF Pro", 14))
        dot.pack(side="left", padx=(0, 6))

        info = tk.Frame(left, bg=BG2)
        info.pack(side="left")
        tk.Label(info, text=m["label"], bg=BG2, fg=WHITE,
                 font=("SF Pro", 13, "bold")).pack(anchor="w")
        status_lbl = tk.Label(info, text="…", bg=BG2, fg=GRAY, font=("SF Pro", 10))
        status_lbl.pack(anchor="w")
        stats_lbl = tk.Label(info, text="", bg=BG2, fg=GRAY, font=("SF Pro", 9))
        stats_lbl.pack(anchor="w")

        right = tk.Frame(card, bg=BG2)
        right.pack(side="right", padx=10, pady=10)

        chk = tk.Checkbutton(right, text="Enabled",
                             variable=self.state[mid]["enabled"],
                             bg=BG2, fg=FG, selectcolor=BG3,
                             activebackground=BG2, activeforeground=FG,
                             font=("SF Pro", 11))
        chk.pack(side="left", padx=(0, 10))

        btn_action = self._btn(right, "Start", BTN_START,
                               lambda i=mid: self._toggle_monitor(i))
        btn_action.pack(side="left", padx=(0, 6))

        btn_rst = self._btn(right, "↺", BTN_RST,
                            lambda i=mid: self._restart_monitor(i))
        btn_rst.pack(side="left", padx=(0, 6))

        btn_test = self._btn(right, "▶1", BG3,
                             lambda i=mid: self._test_monitor(i))
        btn_test.config(padx=8)
        btn_test.pack(side="left", padx=(0, 6))

        btn_hist = self._btn(right, "📋", BG3,
                             lambda i=mid: self._show_history(i))
        btn_hist.config(padx=8)
        btn_hist.pack(side="left", padx=(0, 6))

        btn_reset = self._btn(right, "🗑", BG3,
                              lambda i=mid: self._reset_seen(i))
        btn_reset.config(padx=8)
        btn_reset.pack(side="left", padx=(0, 6))

        btn_edit = self._btn(right, "✎", BG3,
                             lambda i=mid: self._open_edit_dialog(i))
        btn_edit.config(padx=8)
        btn_edit.pack(side="left", padx=(0, 6))

        btn_del = self._btn(right, "✕", BTN_STOP,
                            lambda i=mid: self._delete_monitor(i))
        btn_del.config(padx=6)
        btn_del.pack(side="left")

        self.cards[mid] = {
            "frame":      card,
            "dot":        dot,
            "status_lbl": status_lbl,
            "stats_lbl":  stats_lbl,
            "btn_action": btn_action,
        }

    def _btn(self, parent, text, color, command):
        return FlatButton(parent, text=text, color=color, command=command)

    # ── Dialog nuovo / modifica ───────────────────────────────────────────────

    def _open_new_monitor_dialog(self):
        MonitorDialog(self, on_save=self._on_dialog_save)

    def _open_edit_dialog(self, mid: str):
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            return
        existing_cfg = json.loads(m["config_file"].read_text())
        MonitorDialog(self, on_save=self._on_dialog_save,
                      existing_cfg=existing_cfg, config_file=m["config_file"])

    def _on_dialog_save(self, m: dict, is_edit: bool):
        mid = m["id"]
        if is_edit:
            # Update label in monitors list and card widget
            for entry in self.monitors:
                if entry["id"] == mid:
                    entry["label"] = m["label"]
                    break
            if mid in self.cards:
                # Find and update the bold label widget inside the card
                card_frame = self.cards[mid]["frame"]
                for child in card_frame.winfo_children():
                    for subchild in child.winfo_children():
                        for widget in subchild.winfo_children():
                            if isinstance(widget, tk.Label) and widget.cget("font") and "bold" in str(widget.cget("font")):
                                widget.config(text=m["label"])
                                break
            self._log(f"[{mid.upper()}] ✓ Config updated: {m['label']}", tag="info")
            # If running, ask to restart so changes take effect
            if self._is_running(mid):
                if messagebox.askyesno("Restart?",
                    f"'{m['label']}' is running.\nRestart to apply changes?",
                    parent=self):
                    self._restart_monitor(mid)
        else:
            if mid in self.state:
                self._log(f"[{mid.upper()}] Config updated.", tag="info")
                return
            self.monitors.append(m)
            self._init_state(mid)
            self._build_monitor_card(m)
            self._log(f"[{mid.upper()}] ✓ New search added: {m['label']}", tag="found")

    # ── Eliminazione monitor ──────────────────────────────────────────────────

    def _delete_monitor(self, mid: str):
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if not messagebox.askyesno(
            "Delete search",
            f"Delete '{m['label'] if m else mid}'?\nThe config file will be removed.",
            parent=self,
        ):
            return
        self._stop_monitor(mid)
        # Rimuovi config file
        if m and m["config_file"].exists():
            m["config_file"].unlink()
        # Rimuovi card UI
        if mid in self.cards:
            self.cards[mid]["frame"].destroy()
            del self.cards[mid]
        # Rimuovi stato
        self.monitors = [x for x in self.monitors if x["id"] != mid]
        del self.state[mid]

    # ── Logica processi ───────────────────────────────────────────────────────

    def _is_running(self, mid: str) -> bool:
        s = self.state[mid]
        if s["proc"] is not None and s["proc"].poll() is None:
            return True
        pid = find_pid(f"monitor_{mid}.json")
        s["ext_pid"] = pid
        return pid is not None

    def _test_monitor(self, mid: str):
        """Run a single scan cycle (no email, no loop) and show results in log."""
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            return
        self._log(f"[{mid.upper()}] ▶1 Test run started...", tag="info")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(MONITOR_SCRIPT),
                 "--config", str(m["config_file"]), "--once"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR),
            )
            threading.Thread(target=self._stdout_reader, args=(mid, proc), daemon=True).start()
        except Exception as e:
            self._log(f"[{mid.upper()}] ✗ Test run failed: {e}", tag="error")

    def _reset_seen(self, mid: str):
        """Clear the seen IDs file so the next scan re-notifies everything."""
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            return
        cfg = json.loads(m["config_file"].read_text())
        seen_path = DATA_DIR / cfg["seen_file"]
        if not messagebox.askyesno(
            "Reset seen IDs",
            f"Clear seen history for '{cfg['label']}'?\n"
            "The next scan will re-notify all matching listings.",
            parent=self,
        ):
            return
        if seen_path.exists():
            seen_path.write_text("[]")
        self._log(f"[{mid.upper()}] 🗑 Seen history cleared", tag="warn")
        self._refresh()

    def _start_monitor(self, mid: str):
        s = self.state[mid]
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None or not MONITOR_SCRIPT.exists():
            self._log(f"[{mid.upper()}] ✗ Script or config not found", tag="error")
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(MONITOR_SCRIPT),
                 "--config", str(m["config_file"])],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR),
            )
            s["proc"] = proc
            self._log(f"[{mid.upper()}] ▶ Started — running first scan... (PID {proc.pid})", tag="info")
            threading.Thread(target=self._stdout_reader, args=(mid, proc), daemon=True).start()
        except Exception as e:
            self._log(f"[{mid.upper()}] ✗ Failed to start: {e}", tag="error")

    def _stdout_reader(self, mid: str, proc):
        label = mid.upper()
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                ll = line.lower()

                # Detect URL embedded as " | https://..." suffix
                url = None
                display_line = line
                if " | https://" in line:
                    parts = line.rsplit(" | ", 1)
                    display_line = parts[0]
                    url = parts[1].strip()

                if any(x in ll for x in ("new listings found", "→", "email sent")):
                    tag = "found"
                elif any(x in ll for x in ("no new listings",)):
                    tag = "muted"
                elif any(x in ll for x in ("no internet",)):
                    tag = "warn"
                elif any(x in ll for x in ("connection restored", "immediate scan")):
                    tag = "info"
                elif line.lstrip().startswith("✓"):
                    tag = "found"
                elif line.lstrip().startswith("✗") or any(x in ll for x in ("errore", "error", "exception", "traceback", "cannot")):
                    tag = "error"
                else:
                    tag = "default"

                if url:
                    self.after(0, self._log_link, f"[{label}] {display_line}", tag, url)
                else:
                    self.after(0, self._log, f"[{label}] {display_line}", tag)
        except Exception:
            pass

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
        self._refresh()

    def _restart_monitor(self, mid: str):
        self._stop_monitor(mid)
        time.sleep(0.5)
        self._start_monitor(mid)
        self._refresh()

    def _start_all(self):
        for m in self.monitors:
            mid = m["id"]
            if self.state[mid]["enabled"].get() and not self._is_running(mid):
                self._start_monitor(mid)
        self._refresh()

    def _stop_all(self):
        for m in self.monitors:
            self._stop_monitor(m["id"])
        self._refresh()

    # ── Refresh UI ────────────────────────────────────────────────────────────

    def _refresh(self):
        net_ok, net_msg = check_network()
        self.net_dot.config(fg=GREEN if net_ok else RED)
        self.net_label.config(text=net_msg, fg=GREEN if net_ok else RED)

        for m in self.monitors:
            mid  = m["id"]
            card = self.cards.get(mid)
            if card is None:
                continue
            running = self._is_running(mid)

            if running:
                card["dot"].config(fg=GREEN)
                card["status_lbl"].config(text="Running", fg=GREEN)
                card["btn_action"].config(text="Stop", bg=BTN_STOP)
            else:
                s = self.state[mid]
                crashed = s["proc"] is not None and s["proc"].poll() is not None
                if crashed:
                    ret = s["proc"].poll()
                    s["proc"] = None
                    card["dot"].config(fg=RED)
                    card["status_lbl"].config(text=f"Crashed (exit {ret})", fg=RED)
                    self._log(f"[{mid.upper()}] ✗ Process exited with code {ret}", tag="error")
                else:
                    card["dot"].config(fg=GRAY)
                    card["status_lbl"].config(text="Inactive", fg=GRAY)
                card["btn_action"].config(text="Start", bg=BTN_START)

            if running and not net_ok:
                card["status_lbl"].config(text="Running — ⚠ No network!", fg=YELLOW)

            # Stats line
            try:
                cfg       = json.loads(m["config_file"].read_text())
                seen_n    = _read_seen_count(mid, cfg["seen_file"])
                stats     = _read_stats(mid)
                total     = stats.get("total_notified", 0)
                last      = _time_ago(stats.get("last_found_at"))
                card["stats_lbl"].config(
                    text=f"Seen: {seen_n} IDs · Notified: {total} · Last find: {last}"
                )
            except Exception:
                pass

    def _schedule_refresh(self):
        self._refresh()
        self.after(4000, self._schedule_refresh)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = "default"):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}]  {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _log_link(self, msg: str, tag: str, url: str):
        """Log a line that is clickable and opens url in browser."""
        ts       = time.strftime("%H:%M:%S")
        link_tag = f"link_{self._link_counter}"
        self._link_counter += 1
        self._link_map[link_tag] = url
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}]  {msg} 🔗\n", (tag, link_tag))
        self.log_box.tag_config(link_tag, underline=True)
        self.log_box.tag_bind(link_tag, "<Button-1>",
                              lambda e, u=url: webbrowser.open(u))
        self.log_box.tag_bind(link_tag, "<Enter>",
                              lambda e: self.log_box.config(cursor="hand2"))
        self.log_box.tag_bind(link_tag, "<Leave>",
                              lambda e: self.log_box.config(cursor=""))
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ── History window ────────────────────────────────────────────────────────

    def _show_history(self, mid: str):
        m = next((x for x in self.monitors if x["id"] == mid), None)
        if m is None:
            return
        cfg          = json.loads(m["config_file"].read_text())
        history_path = DATA_DIR / f"history_{mid}.json"
        history      = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title(f"History — {cfg['label']}")
        win.configure(bg=BG)
        win.resizable(True, True)

        hdr = tk.Frame(win, bg=BG)
        hdr.pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(hdr, text=f"Notification history — {cfg['label']}",
                 bg=BG, fg=WHITE, font=("SF Pro Display", 14, "bold")).pack(side="left")

        def _clear():
            if not messagebox.askyesno("Clear history",
                    f"Clear all history for '{cfg['label']}'?", parent=win):
                return
            history_path.write_text("[]")
            win.destroy()
            self._log(f"[{mid.upper()}] 🗑 History cleared", tag="warn")

        FlatButton(hdr, text="🗑 Clear", color=BTN_STOP,
                   command=_clear, padx=8, pady=3,
                   font=("SF Pro", 10)).pack(side="right")

        if not history:
            tk.Label(win, text="No notifications yet.", bg=BG, fg=GRAY,
                     font=("SF Pro", 11)).pack(padx=14, pady=20)
            return

        # Scrollable frame
        container = tk.Frame(win, bg=BG)
        container.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        canvas  = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner   = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollb.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollb.pack(side="right", fill="y")

        # Header row
        for col, txt in enumerate(["Date", "Price", "Title"]):
            tk.Label(inner, text=txt, bg=BG, fg=GRAY,
                     font=("SF Pro", 10, "bold"), width=12 if col < 2 else 40,
                     anchor="w").grid(row=0, column=col, padx=6, pady=2, sticky="w")

        for i, item in enumerate(reversed(history), start=1):
            date  = item.get("notified_at", "")[:16]
            price = f"€{item.get('price', '?')}"
            title = item.get("title", "")
            url   = item.get("url", "")

            tk.Label(inner, text=date,  bg=BG, fg=GRAY, font=("Menlo", 10),
                     anchor="w").grid(row=i, column=0, padx=6, pady=1, sticky="w")
            tk.Label(inner, text=price, bg=BG, fg=GREEN, font=("SF Pro", 10, "bold"),
                     anchor="w").grid(row=i, column=1, padx=6, pady=1, sticky="w")

            title_lbl = tk.Label(inner, text=title, bg=BG, fg=FG,
                                 font=("SF Pro", 10), anchor="w", cursor="hand2",
                                 wraplength=420)
            title_lbl.grid(row=i, column=2, padx=6, pady=1, sticky="w")
            if url:
                title_lbl.config(fg=BLUE, underline=True if not url else False)
                title_lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
                title_lbl.bind("<Enter>", lambda e, l=title_lbl: l.config(fg=WHITE))
                title_lbl.bind("<Leave>", lambda e, l=title_lbl, u=url: l.config(fg=BLUE))

        win.geometry("700x460")

    # ── Charts window ─────────────────────────────────────────────────────────

    def _show_charts(self):
        today = datetime.date.today()
        days  = [(today - datetime.timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
        labels = [d[5:] for d in days]  # MM-DD

        # Collect data
        all_counts = {}
        for m in self.monitors:
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

        PAD_L, PAD_R, PAD_T, PAD_B = 48, 16, 30, 28
        W, H   = 820, 180
        n_days = len(days)

        for m in self.monitors:
            values  = all_counts[m["id"]]
            max_val = max(values) if any(values) else 1

            frame = tk.Frame(win, bg=BG, pady=6)
            frame.pack(fill="x", padx=14)
            tk.Label(frame, text=m["label"], bg=BG, fg=FG,
                     font=("SF Pro", 11, "bold")).pack(anchor="w")

            c = tk.Canvas(frame, width=W, height=H, bg=BG2,
                          highlightthickness=0)
            c.pack()

            draw_w = W - PAD_L - PAD_R
            draw_h = H - PAD_T - PAD_B
            bar_w  = max(4, draw_w // n_days - 2)

            # Y axis lines
            for step in range(0, max_val + 1):
                y = PAD_T + draw_h - int(step / max_val * draw_h)
                c.create_line(PAD_L - 4, y, W - PAD_R, y, fill=BG3)
                c.create_text(PAD_L - 6, y, text=str(step),
                              anchor="e", fill=GRAY, font=("Menlo", 8))

            for i, (val, lbl) in enumerate(zip(values, labels)):
                x_center = PAD_L + int((i + 0.5) * draw_w / n_days)
                bar_h    = int(val / max_val * draw_h) if max_val else 0
                x0 = x_center - bar_w // 2
                x1 = x_center + bar_w // 2
                y0 = PAD_T + draw_h - bar_h
                y1 = PAD_T + draw_h

                color = BLUE if val > 0 else BG3
                c.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

                if val > 0:
                    c.create_text(x_center, y0 - 4, text=str(val),
                                  anchor="s", fill=FG, font=("Menlo", 8))

                # X label every 2 days to avoid crowding
                if i % 2 == 0:
                    c.create_text(x_center, H - PAD_B + 6, text=lbl,
                                  anchor="n", fill=GRAY, font=("Menlo", 8))

        win.geometry(f"848x{(H + 50) * len(self.monitors)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = VintedGUI()
    app.mainloop()
