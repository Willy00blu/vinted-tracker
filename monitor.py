#!/usr/bin/env python3
"""
Vinted monitor runner — generic, config-driven.

This script is launched by the GUI (gui.py) for each active monitor.
It can also be run manually:

    python monitor.py --config configs/monitor_5k.json

One config file = one type of item to track.
Add new configs in the configs/ folder to monitor anything you want.
"""

import argparse
import json
import time
from pathlib import Path

from core.network  import wifi_aware_sleep
from core.storage  import load_seen, save_seen, load_price_db, load_stats, bump_stats, append_history
from core.scraper  import search_all
from core.notifier import send_email

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
SETTINGS_FILE = BASE_DIR / "settings.json"


def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        raise FileNotFoundError(
            "settings.json not found.\n"
            "Copy settings.example.json → settings.json and fill in your credentials."
        )
    return json.loads(SETTINGS_FILE.read_text())


def run(cfg: dict, once: bool = False):
    settings   = load_settings()
    label      = cfg["label"]
    interval   = cfg["check_interval_minutes"] * 60
    seen_path    = DATA_DIR / cfg["seen_file"]
    stats_path   = DATA_DIR / f"stats_{cfg['id']}.json"
    history_path = DATA_DIR / f"history_{cfg['id']}.json"

    # Load optional price reference DB (used to calculate discounts)
    price_db = None
    if cfg.get("price_db_file"):
        db_path = DATA_DIR / cfg["price_db_file"]
        if db_path.exists():
            price_db = load_price_db(db_path)

    print(f"{label} — started", flush=True)
    if not once:
        print(f"Checking every {cfg['check_interval_minutes']} min, max €{cfg['max_price']}\n", flush=True)

    seen = load_seen(seen_path)

    # First run: populate seen IDs without sending notifications, but save to history
    if not seen and not once:
        print("First run: loading existing listings without notifying...", flush=True)
        found = search_all(cfg, price_db)
        now   = time.strftime("%Y-%m-%d %H:%M:%S")
        items = list(found.values())
        for it in items:
            it["notified_at"] = now
        if items:
            append_history(history_path, items)
        seen = set(str(k) for k in found.keys())
        save_seen(seen_path, seen)
        print(f"  {len(seen)} listings saved. New ones will be notified from now on.\n", flush=True)

    def _cycle():
        nonlocal seen, price_db
        print(f"[{time.strftime('%H:%M:%S')}] Scanning for new listings — {label}...", flush=True)

        # Reload price DB at every cycle so it can be updated without restarting
        if cfg.get("price_db_file"):
            db_path = DATA_DIR / cfg["price_db_file"]
            if db_path.exists():
                price_db = load_price_db(db_path)

        found     = search_all(cfg, price_db)
        new_items = [v for k, v in found.items() if str(k) not in seen]

        if new_items:
            if price_db:
                new_items.sort(key=lambda x: x.get("discount_pct", 0), reverse=True)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {len(new_items)} new listings found!", flush=True)
            for it in new_items:
                lbl    = it.get("label", "")
                disc   = it.get("discount_str", "")
                prefix = f"{lbl} " if lbl  else ""
                suffix = f" {disc}" if disc else ""
                print(f"  → {prefix}[{it['price']}€] {it['title']}{suffix} | {it['url']}", flush=True)
                it["notified_at"] = now
            if not once:
                send_email(new_items, cfg, settings)
                append_history(history_path, new_items)
            seen.update(str(k) for k in found.keys())
            save_seen(seen_path, seen)
            bump_stats(stats_path, len(new_items))
        else:
            print("  No new listings.", flush=True)

    if once:
        _cycle()
        return

    while True:
        _cycle()
        wifi_aware_sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vinted item monitor")
    parser.add_argument(
        "--config", required=True,
        help="Path to a monitor config JSON (e.g. configs/monitor_5k.json)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scan cycle and exit (no email, no loop)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    cfg = json.loads(config_path.read_text())
    run(cfg, once=args.once)
