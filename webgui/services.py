"""Data access + monitor-process management for the NiceGUI front-end.

Keeps all the "talk to the backend" logic in one place so the UI (app.py) stays
declarative. Nothing here imports nicegui — it's pure Python and unit-testable.
"""

from __future__ import annotations

import collections
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

BASE_DIR       = Path(__file__).resolve().parent.parent
CONFIGS_DIR    = BASE_DIR / "configs"
DATA_DIR       = BASE_DIR / "data"
MONITOR_SCRIPT = BASE_DIR / "monitor.py"


# ── Network ─────────────────────────────────────────────────────────────────

def check_network() -> bool:
    for url in ("https://www.google.com", "https://www.apple.com"):
        try:
            urllib.request.urlopen(url, timeout=4)
            return True
        except Exception:
            pass
    for host, port in (("8.8.8.8", 53), ("1.1.1.1", 53)):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            return True
        except OSError:
            pass
    return False


# ── Time / math helpers ─────────────────────────────────────────────────────

def time_ago(ts: str | None) -> str:
    if not ts:
        return "mai"
    try:
        dt = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        diff = int(time.time() - dt)
        if diff < 60:    return "ora"
        if diff < 3600:  return f"{diff // 60}m fa"
        if diff < 86400: return f"{diff // 3600}h fa"
        return f"{diff // 86400}g fa"
    except Exception:
        return ts


def median(values: list) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── Config / per-monitor data ───────────────────────────────────────────────

def load_monitors() -> list[dict]:
    out: list[dict] = []
    for cfg_file in sorted(CONFIGS_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_file.read_text())
            out.append({
                "id":          cfg["id"],
                "label":       cfg["label"],
                "type":        cfg.get("type", "text"),
                "config_file": cfg_file,
            })
        except Exception:
            pass
    out.sort(key=lambda m: (m["type"] != "text", m["label"]))
    return out


def load_config(config_file: Path) -> dict:
    try:
        return json.loads(Path(config_file).read_text())
    except Exception:
        return {}


def _read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def read_stats(mid: str) -> dict:
    return _read_json(DATA_DIR / f"stats_{mid}.json",
                      {"total_notified": 0, "last_found_at": None})


def read_seen_count(seen_file: str) -> int:
    data = _read_json(DATA_DIR / seen_file, [])
    return len(data) if isinstance(data, list) else 0


def read_price_stats(mid: str) -> dict:
    return _read_json(DATA_DIR / f"price_stats_{mid}.json",
                      {"prices": [], "sample_count": 0, "updated_at": None, "anomalies": []})


def read_history(mid: str) -> list:
    return _read_json(DATA_DIR / f"history_{mid}.json", [])


def coverage_state(cfg_id: str) -> dict:
    return _read_json(DATA_DIR / f"coverage_cat_{cfg_id}.json", {})


DEALS_FILE = DATA_DIR / "deals.json"


def read_deals() -> dict:
    """Last deal-scan results written by scripts/deal_detector.py --json."""
    return _read_json(DEALS_FILE, {"generated_at": None, "n_items": 0, "deals": []})


# ── Dashboard query cache ────────────────────────────────────────────────────
# The Categorie view runs a GROUP BY over ~200k rows on every refresh; a short
# TTL cache keeps the UI snappy without meaningfully stale data (the underlying
# monitor writes at scrape-interval granularity anyway).
_dash_cache: dict = {}
_DASH_TTL = 20.0


def dash_cached(key: tuple, compute):
    now = time.time()
    hit = _dash_cache.get(key)
    if hit and now - hit[0] < _DASH_TTL:
        return hit[1]
    val = compute()
    _dash_cache[key] = (now, val)
    return val


def dash_invalidate():
    _dash_cache.clear()


def subcategories() -> list[tuple[str, int]]:
    """[(name, catalog_id)] of sub-categories seen in the shared DB (for the
    new-category form). Thin wrapper over core.discovery."""
    from core.discovery import subcategories_seen
    return subcategories_seen()


def make_category_config(name: str, catalog_ids: list[int], *,
                         order: str = "favourites", min_price: int = 10,
                         max_price: int = 5000, scrape_interval: int = 3) -> tuple[dict, Path]:
    """Build a category-monitor config from form inputs. `catalog_ids` (selected
    leaf sub-categories) scope BOTH the scrape and the sold-check. Pure function."""
    import re
    mid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "categoria"
    cfg = {
        "id": mid, "label": name, "type": "category",
        "catalog_ids": catalog_ids, "soldcheck_catalog_ids": catalog_ids,
        "domains": ["https://www.vinted.it", "https://www.vinted.fr", "https://www.vinted.de"],
        "min_price": min_price, "max_price": max_price,
        "check_interval_minutes": 8, "scrape_interval_minutes": scrape_interval,
        "pages_per_domain": 3, "pages_per_domain_max": 10,
        "request_delay_seconds": 2.0,
        "price_drop_threshold_pct": 15, "price_drop_min_abs": 5,
        "presumed_sold_min_seen": 3, "presumed_sold_http_quota": 20,
        "soldcheck_order": order,
    }
    return cfg, CONFIGS_DIR / f"{mid}.json"


def delete_config(config_file: Path) -> None:
    try:
        Path(config_file).unlink(missing_ok=True)
    except Exception:
        pass


# ── Monitor process management ───────────────────────────────────────────────

def find_pid(pattern: str) -> int | None:
    if HAS_PSUTIL:
        try:
            for p in psutil.process_iter(["pid", "cmdline"]):
                if any(pattern in a for a in (p.info.get("cmdline") or [])):
                    return p.info["pid"]
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


def kill_pid(pid: int) -> None:
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
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


class MonitorManager:
    """Owns the monitor subprocesses + a shared rolling log buffer the UI drains.

    Log lines are pushed (thread-safe) into `log` as dicts
    {ts, mid, text, tag, url}; the UI polls `drain_log()` on a timer instead of
    being called cross-thread."""

    LOG_MAX = 2000

    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}
        self.log: collections.deque = collections.deque(maxlen=self.LOG_MAX)
        self._lock = threading.Lock()
        self._cursor = 0          # how many log entries the UI has consumed
        self._seq = 0
        self._regen_active = False   # guard: one cluster regeneration at a time
        self._regen_proc: subprocess.Popen | None = None
        self._deal_scan_active = False  # guard: one deal scan at a time
        # Image-embedding worker (core.image_worker): one global process, tied to
        # the category monitors. auto_image → auto-start it whenever a category
        # monitor is running (and re-start if it died).
        self._image_proc: subprocess.Popen | None = None
        self.auto_image = True
        # Text-embedding worker (core.text_worker): symmetric to image worker.
        # Pre-computes title embeddings (e5-large) → regen consumes the cache
        # from the DB instead of re-encoding 165k titles every time.
        self._text_proc: subprocess.Popen | None = None
        self.auto_text = True
        # Sold-check worker (core.sold_check_worker): symmetric to img/text
        # workers. Runs the drip independently of the scrape monitor.
        self._sold_proc: subprocess.Popen | None = None
        self.auto_sold = True

    # -- logging --
    def _emit(self, mid: str, text: str, tag: str = "default", url: str | None = None):
        with self._lock:
            self._seq += 1
            self.log.append({"seq": self._seq, "ts": time.strftime("%H:%M:%S"),
                             "mid": mid, "text": text, "tag": tag, "url": url})

    def drain_log(self, after_seq: int) -> list[dict]:
        """Return log entries with seq > after_seq (for incremental UI updates)."""
        with self._lock:
            return [e for e in self.log if e["seq"] > after_seq]

    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    # -- cluster regeneration --
    def regen_active(self) -> bool:
        with self._lock:
            return self._regen_active

    def regenerate_clusters(self, mid: str = "cluster") -> bool:
        """Wipe + recompute clustering for the whole shared DB.

        Runs in a SEPARATE PROCESS (not a thread): the embedding/HDBSCAN pass is
        CPU-bound and would hold the GIL, starving NiceGUI's event loop and
        dropping the browser websocket ("connessione persa"). A subprocess keeps
        the UI fully responsive. We only read its stdout (I/O, releases the GIL)
        and stream it into the live log. The slow work holds no write lock, so
        running monitors keep going.

        Returns False if a regeneration is already in progress."""
        with self._lock:
            if self._regen_active:
                return False
            self._regen_active = True

        from core.category_db import shared_db_path
        db = shared_db_path()
        if not db.exists():
            self._emit(mid, f"✗ DB {db.name} inesistente", "error")
            with self._lock:
                self._regen_active = False
            return False

        self._emit(mid, "🔄 Rigenerazione cluster avviata (processo separato, "
                        "qualche minuto)…", "info")

        def reader():
            # Pause the image worker so the regen alone catches up ALL pending
            # images (no race, no double CLIP). The regen embeds every item with
            # photo_urls and no image_emb before clustering → "in pari con tutti
            # gli URL". Resume the worker afterwards.
            worker_was_on = self.image_worker_running()
            txt_worker_was_on = self.text_worker_running()
            sold_worker_was_on = self.sold_check_worker_running()
            if worker_was_on:
                self._kill_image_worker()
                self._emit(mid, "💤 image worker in sleep durante la rigenerazione (riprende a fine regen)", "muted")
            if txt_worker_was_on:
                self._kill_text_worker()
                self._emit(mid, "💤 text worker in sleep durante la rigenerazione", "muted")
            if sold_worker_was_on:
                self._kill_sold_check_worker()
                self._emit(mid, "💤 sold-check worker in sleep durante la rigenerazione", "muted")
            try:
                argv = [sys.executable, "-u", "-m", "scripts.regenerate_clusters", str(db)]
                # Embedding model: the code default is now e5-large (see
                # core.clustering.MODEL_NAME); the subprocess inherits the env,
                # so set VINTED_EMBED_MODEL=…MiniLM… to fall back to the light one.
                proc = subprocess.Popen(
                    argv, cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._regen_proc = proc
                for raw in proc.stdout:                       # type: ignore[union-attr]
                    line = raw.rstrip()
                    if not line:
                        continue
                    head = line.lstrip()
                    if head.startswith("✓") or head.startswith("done") or "done:" in line:
                        tag = "found"
                    elif head.startswith("✗") or head.startswith(("Traceback", "Error", "  File")):
                        tag = "error"
                    else:
                        tag = "default"
                    self._emit(mid, line, tag)
                code = proc.wait()
                if code == 0:
                    self._emit(mid, "✓ Rigenerazione completata — premi ↻ Refresh per vedere i cluster", "found")
                else:
                    self._emit(mid, f"✗ Rigenerazione terminata con codice {code}", "error")
            except Exception as e:
                self._emit(mid, f"✗ Errore avvio rigenerazione: {e}", "error")
            finally:
                self._regen_proc = None
                with self._lock:
                    self._regen_active = False
                # resume the workers if they were running (and auto-start still on)
                if worker_was_on and self.auto_image:
                    self.start_image_worker()
                    self._emit(mid, "▶ image worker riattivato", "muted")
                if txt_worker_was_on and self.auto_text:
                    self.start_text_worker()
                    self._emit(mid, "▶ text worker riattivato", "muted")
                if sold_worker_was_on and self.auto_sold:
                    self.start_sold_check_worker()
                    self._emit(mid, "▶ sold-check worker riattivato", "muted")

        threading.Thread(target=reader, daemon=True).start()
        return True

    # -- deal scan (detector v2 in a subprocess, results → data/deals.json) --
    def deal_scan_active(self) -> bool:
        with self._lock:
            return self._deal_scan_active

    def run_deal_scan(self, top_n: int = 300) -> bool:
        """Run scripts/deal_detector.py --json in a subprocess (CPU/RAM heavy:
        loads all cached text embeddings once). UI stays responsive; results
        land in data/deals.json. Refuses while a regen or another scan runs."""
        with self._lock:
            if self._deal_scan_active or self._regen_active:
                return False
            self._deal_scan_active = True
        self._emit("DEALS", "💎 Scansione occasioni avviata (detector v2)…", "info")

        def reader():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(BASE_DIR / "scripts" / "deal_detector.py"),
                     "--json", str(DEALS_FILE), "--top", str(top_n)],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for raw in proc.stdout:                    # type: ignore[union-attr]
                    line = raw.rstrip()
                    if line:
                        tag = ("found" if line.startswith("✓")
                               else "error" if line.startswith(("✗", "Traceback"))
                               else "muted")
                        self._emit("DEALS", line, tag)
                code = proc.wait()
                if code == 0:
                    self._emit("DEALS", "✓ Scansione completata — apri la tab 💎 Occasioni",
                               "found")
                else:
                    self._emit("DEALS", f"✗ Scansione terminata con codice {code}", "error")
            except Exception as e:
                self._emit("DEALS", f"✗ Errore scansione: {e}", "error")
            finally:
                with self._lock:
                    self._deal_scan_active = False

        threading.Thread(target=reader, daemon=True).start()
        return True

    # -- process state --
    def is_running(self, mid: str) -> bool:
        p = self.procs.get(mid)
        if p is not None and p.poll() is None:
            return True
        return find_pid(f"{mid}.json") is not None

    def crashed_code(self, mid: str) -> int | None:
        p = self.procs.get(mid)
        if p is not None and p.poll() not in (None, 0):
            return p.poll()
        return None

    def _argv(self, m: dict, once: bool = False) -> list[str]:
        argv = [sys.executable, "-u"]
        if m.get("type") == "category":
            argv += ["-m", "core.category_monitor"]
        else:
            argv += [str(MONITOR_SCRIPT)]
        argv += ["--config", str(m["config_file"])]
        if once:
            argv += ["--once"]
        return argv

    def _reader(self, mid: str, proc: subprocess.Popen):
        label = mid.upper()
        try:
            for raw in proc.stdout:                      # type: ignore[union-attr]
                line = raw.rstrip()
                if not line:
                    continue
                url = None
                text = line
                if " | https://" in line:
                    text, url = line.rsplit(" | ", 1)
                    url = url.strip()
                ll = line.lower()
                if any(x in ll for x in ("new listings found", "→", "telegram sent",
                                         "drip", "sold", "🔁")) or line.lstrip().startswith("✓"):
                    tag = "found"
                elif "no new" in ll or "no internet" in ll:
                    tag = "muted"
                elif line.lstrip().startswith("✗") or any(
                        x in ll for x in ("error", "errore", "traceback", "blocked", "⏸")):
                    tag = "error"
                else:
                    tag = "default"
                self._emit(label, text, tag, url)
        except Exception:
            pass

    def start(self, m: dict):
        mid = m["id"]
        if self.is_running(mid):
            return
        try:
            proc = subprocess.Popen(
                self._argv(m), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR))
            self.procs[mid] = proc
            self._emit(mid.upper(), f"▶ avviato (PID {proc.pid})", "info")
            threading.Thread(target=self._reader, args=(mid, proc), daemon=True).start()
        except Exception as e:
            self._emit(mid.upper(), f"✗ avvio fallito: {e}", "error")
        # A category monitor produces photo_urls → make sure the image worker
        # is running to vectorize them.
        if m.get("type") == "category":
            self.sync_image_worker(True)
            self.sync_text_worker(True)

    # -- image-embedding worker --
    def image_worker_running(self) -> bool:
        p = self._image_proc
        if p is not None and p.poll() is None:
            return True
        return find_pid("image_worker") is not None

    def start_image_worker(self):
        if self.image_worker_running():
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "core.image_worker"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR))
            self._image_proc = proc
            self._emit("IMG-WORKER", f"▶ avviato (PID {proc.pid})", "info")
            threading.Thread(target=self._reader, args=("img-worker", proc), daemon=True).start()
        except Exception as e:
            self._emit("IMG-WORKER", f"✗ avvio fallito: {e}", "error")

    def _kill_image_worker(self):
        """Stop the worker process WITHOUT disabling auto-start (internal pause,
        e.g. during a regenerate so it doesn't race the regen's catch-up)."""
        p = self._image_proc
        if p is not None:
            kill_pid(p.pid)
            self._image_proc = None
        pid = find_pid("image_worker")
        if pid:
            kill_pid(pid)

    def stop_image_worker(self):
        self.auto_image = False          # explicit stop overrides auto-start
        self._kill_image_worker()
        self._emit("IMG-WORKER", "■ fermato", "muted")

    def sync_image_worker(self, category_active: bool):
        """Auto-start the worker whenever a category monitor is active (and it
        isn't already running). Called on monitor start + periodically by the UI,
        so it re-activates every time a category monitor comes back up.

        NEVER (re)start it during a cluster regeneration: the regen alone catches
        up all pending images, and a second CLIP process would race it + double
        the RAM. The regen's `finally` resumes the worker when it's done."""
        if self._regen_active or (BASE_DIR / "data" / ".workers_paused").exists():
            return
        if self.auto_image and category_active and not self.image_worker_running():
            self.start_image_worker()

    # -- text-embedding worker (symmetric to image worker) --
    def text_worker_running(self) -> bool:
        p = self._text_proc
        if p is not None and p.poll() is None:
            return True
        return find_pid("text_worker") is not None

    def start_text_worker(self):
        if self.text_worker_running():
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "core.text_worker"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR))
            self._text_proc = proc
            self._emit("TXT-WORKER", f"▶ avviato (PID {proc.pid})", "info")
            threading.Thread(target=self._reader, args=("txt-worker", proc), daemon=True).start()
        except Exception as e:
            self._emit("TXT-WORKER", f"✗ avvio fallito: {e}", "error")

    def _kill_text_worker(self):
        """Stop the worker process WITHOUT disabling auto-start (internal pause,
        e.g. during a regenerate)."""
        p = self._text_proc
        if p is not None:
            kill_pid(p.pid)
            self._text_proc = None
        pid = find_pid("text_worker")
        if pid:
            kill_pid(pid)

    def stop_text_worker(self):
        self.auto_text = False
        self._kill_text_worker()
        self._emit("TXT-WORKER", "■ fermato", "muted")

    def sync_text_worker(self, category_active: bool):
        """Same semantics as `sync_image_worker`: auto-start when a category
        monitor is active, never restart during a regen."""
        if self._regen_active or (BASE_DIR / "data" / ".workers_paused").exists():
            return
        if self.auto_text and category_active and not self.text_worker_running():
            self.start_text_worker()

    # -- sold-check worker (separate process, runs independently of monitor) --
    def sold_check_worker_running(self) -> bool:
        p = self._sold_proc
        if p is not None and p.poll() is None:
            return True
        return find_pid("sold_check_worker") is not None

    def start_sold_check_worker(self):
        if self.sold_check_worker_running():
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "core.sold_check_worker"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR))
            self._sold_proc = proc
            self._emit("SOLD-CHECK", f"▶ avviato (PID {proc.pid})", "info")
            threading.Thread(target=self._reader, args=("sold-check", proc), daemon=True).start()
        except Exception as e:
            self._emit("SOLD-CHECK", f"✗ avvio fallito: {e}", "error")

    def _kill_sold_check_worker(self):
        p = self._sold_proc
        if p is not None:
            kill_pid(p.pid)
            self._sold_proc = None
        pid = find_pid("sold_check_worker")
        if pid:
            kill_pid(pid)

    def stop_sold_check_worker(self):
        self.auto_sold = False
        self._kill_sold_check_worker()
        self._emit("SOLD-CHECK", "■ fermato", "muted")

    def sync_sold_check_worker(self):
        """Auto-start the sold-check worker. Unlike img/text workers it does
        NOT depend on the scrape monitor being active — the user can keep the
        drip alive even with scrape off."""
        if self._regen_active or (BASE_DIR / "data" / ".workers_paused").exists():
            return
        if self.auto_sold and not self.sold_check_worker_running():
            self.start_sold_check_worker()

    def test(self, m: dict):
        mid = m["id"]
        self._emit(mid.upper(), "▶¹ test (un ciclo)…", "info")
        try:
            proc = subprocess.Popen(
                self._argv(m, once=True), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE_DIR))
            threading.Thread(target=self._reader, args=(mid, proc), daemon=True).start()
        except Exception as e:
            self._emit(mid.upper(), f"✗ test fallito: {e}", "error")

    def stop(self, mid: str):
        p = self.procs.pop(mid, None)
        stopped = False
        if p is not None:
            kill_pid(p.pid)
            stopped = True
        ext = find_pid(f"{mid}.json")
        if ext is not None:
            kill_pid(ext)
            stopped = True
        if stopped:
            self._emit(mid.upper(), "■ fermato", "muted")

    def restart(self, m: dict):
        self.stop(m["id"])
        time.sleep(0.4)
        self.start(m)
