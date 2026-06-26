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
from core.storage  import (
    load_seen, save_seen, load_price_db, bump_stats, append_history,
    load_price_stats, update_price_stats, compute_median, record_anomaly,
    update_tracked_and_detect_drops,
)
from core.scraper  import search_all
from core.notifier import send_notification

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
    seen_path        = DATA_DIR / cfg["seen_file"]
    stats_path       = DATA_DIR / f"stats_{cfg['id']}.json"
    history_path     = DATA_DIR / f"history_{cfg['id']}.json"
    price_stats_path = DATA_DIR / f"price_stats_{cfg['id']}.json"
    tracked_path     = DATA_DIR / f"tracked_{cfg['id']}.json"

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
        found, all_prices = search_all(cfg, price_db)
        if all_prices:
            update_price_stats(price_stats_path, all_prices)
        # Also establish the tracked-price baseline for drop detection
        update_tracked_and_detect_drops(tracked_path, found, 0, 0)
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

        found, all_prices = search_all(cfg, price_db)

        # Update market price stats (rolling window of all keyword-matching prices)
        if all_prices and not once:
            update_price_stats(price_stats_path, all_prices)

        # ── Price-drop detection on already-seen items ────────────────────────
        # Piggybacks on the same `found` dict: items still in the feed get
        # re-priced and compared to their stored baseline.
        drop_threshold_pct = cfg.get("price_drop_threshold_pct", 15)
        drop_min_abs       = cfg.get("price_drop_min_abs", 5)
        drops: list = []
        if not once:
            drops = update_tracked_and_detect_drops(
                tracked_path, found, drop_threshold_pct, drop_min_abs,
            )

        new_items = [v for k, v in found.items() if str(k) not in seen]

        # Anomaly detection on new items
        threshold_pct = cfg.get("anomaly_threshold_pct", 0)
        min_samples   = cfg.get("anomaly_min_samples", 20)
        if threshold_pct and new_items:
            ps = load_price_stats(price_stats_path)
            if len(ps["prices"]) >= min_samples:
                median = compute_median(ps["prices"])
                if median:
                    cutoff = median * (1 - threshold_pct / 100)
                    for it in new_items:
                        if it["price"] < cutoff:
                            disc = round((1 - it["price"] / median) * 100)
                            it["anomaly_pct"]    = disc
                            it["anomaly_median"] = median
                            it["anomaly_str"]    = f"🚨 -{disc}% sotto mediana (€{int(median)})"
                            if not once:
                                record_anomaly(price_stats_path, it, median)

        if new_items:
            if any(i.get("anomaly_pct") for i in new_items):
                new_items.sort(key=lambda x: x.get("anomaly_pct", 0), reverse=True)
            elif cfg.get("min_favourites", 0):
                new_items.sort(key=lambda x: x.get("favourite_count", 0), reverse=True)
            elif price_db:
                new_items.sort(key=lambda x: x.get("discount_pct", 0), reverse=True)

        drops.sort(key=lambda x: x.get("drop_pct", 0), reverse=True)

        now = time.strftime("%Y-%m-%d %H:%M:%S")

        if new_items:
            print(f"  {len(new_items)} new listings found!", flush=True)
            for it in new_items:
                lbl    = it.get("label", "")
                disc   = it.get("discount_str", "")
                fav    = it.get("favourite_count", 0)
                anom   = it.get("anomaly_str", "")
                prefix = f"{lbl} " if lbl  else ""
                suffix = f" {disc}" if disc else ""
                fav_str  = f" ♥{fav}" if fav else ""
                anom_str = f" {anom}" if anom else ""
                print(f"  → {prefix}[{it['price']}€]{fav_str}{anom_str} {it['title']}{suffix} | {it['url']}", flush=True)
                it["notified_at"] = now

        if drops:
            print(f"  {len(drops)} price drops detected!", flush=True)
            for d in drops:
                fav     = d.get("favourite_count", 0)
                fav_str = f" ♥{fav}" if fav else ""
                print(f"  ↓ [{d['price']}€]{fav_str} {d['drop_str']} {d['title']} | {d['url']}", flush=True)
                d["notified_at"] = now

        if not new_items and not drops:
            print("  No new listings or price drops.", flush=True)

        if (new_items or drops) and not once:
            send_notification(new_items, cfg, settings, drops=drops)
            append_history(history_path, new_items + drops)
            if new_items:
                seen.update(str(k) for k in found.keys())
                save_seen(seen_path, seen)
            bump_stats(stats_path, len(new_items) + len(drops))

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
