"""
Network connectivity checks and WiFi-aware sleep.

is_online()         — returns True if internet is reachable
wifi_aware_sleep()  — like time.sleep(), but wakes up early when
                      WiFi was lost and then restored, so a scraping
                      cycle can run immediately after reconnection.
"""

import socket
import time
import urllib.request


def is_online() -> bool:
    """Return True if the internet is reachable."""
    # Try HTTP first (most reliable on macOS)
    for url in ("https://www.google.com", "https://www.apple.com"):
        try:
            urllib.request.urlopen(url, timeout=4)
            return True
        except Exception:
            pass
    # Fallback: raw TCP to public DNS servers
    for host, port in [("8.8.8.8", 53), ("1.1.1.1", 53)]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            return True
        except OSError:
            pass
    return False


def wifi_aware_sleep(seconds: int):
    """
    Sleep for `seconds`, but exit early if WiFi was lost and then restored.

    This ensures a scraping cycle runs immediately after reconnection
    instead of waiting for the full interval to expire.
    """
    net_lost = False
    elapsed  = 0
    poll     = 15  # check every 15 seconds

    while elapsed < seconds:
        time.sleep(min(poll, seconds - elapsed))
        elapsed += poll

        if not is_online():
            if not net_lost:
                print(f"[{time.strftime('%H:%M:%S')}] No internet — waiting for connection...", flush=True)
            net_lost = True
        elif net_lost:
            print(f"[{time.strftime('%H:%M:%S')}] Connection restored — running immediate scan", flush=True)
            return
