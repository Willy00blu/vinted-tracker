"""
Telegram notifications.

send_notification() — sends a Telegram message listing new items and price drops.
Credentials are read from settings.json:
    - telegram_bot_token  (create a bot via @BotFather)
    - telegram_chat_id    (your numeric chat id — ask @userinfobot)
"""

from __future__ import annotations

import html

import httpx


# Telegram caps sendMessage at 4096 chars (text + entities). Leave room for
# the header wrappers we add when splitting.
TG_MAX_LEN = 4000


def _esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def _fmt_new(it: dict) -> str:
    label   = _esc(it.get("label", ""))
    disc    = _esc(it.get("discount_str", ""))
    fav     = it.get("favourite_count", 0)
    anom    = _esc(it.get("anomaly_str", ""))
    title   = _esc(it.get("title", ""))
    price   = it.get("price", 0)
    url     = _esc(it.get("url", ""))
    prefix   = f"{label} — " if label else ""
    suffix   = f"  {disc}"   if disc  else ""
    fav_str  = f"  ♥{fav}"   if fav   else ""
    anom_str = f"  {anom}"   if anom  else ""
    return (
        f"{prefix}<b>€{price}</b>{fav_str}{anom_str}  "
        f"<a href=\"{url}\">{title}</a>{suffix}"
    )


def _fmt_drop(d: dict) -> str:
    fav     = d.get("favourite_count", 0)
    fav_str = f"  ♥{fav}" if fav else ""
    title   = _esc(d.get("title", ""))
    url     = _esc(d.get("url", ""))
    drop    = _esc(d.get("drop_str", ""))
    return f"{drop}{fav_str}  <a href=\"{url}\">{title}</a>"


def _chunks(lines: list[str], header: str) -> list[str]:
    """Pack lines into Telegram messages of ≤TG_MAX_LEN chars.
    First chunk gets the header; later chunks get a continuation marker so
    the reader can tell they belong together."""
    out:   list[str] = []
    cur:   list[str] = [header]
    cur_len          = len(header)
    for line in lines:
        n = len(line) + 1
        if cur_len + n > TG_MAX_LEN:
            out.append("\n".join(cur))
            cont = f"… <i>(continua, parte {len(out) + 1})</i>"
            cur     = [cont]
            cur_len = len(cont)
        cur.append(line)
        cur_len += n
    out.append("\n".join(cur))
    return out


def _send(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":                  chat_id,
        "text":                     text,
        "parse_mode":               "HTML",
        "disable_web_page_preview": True,
    }
    with httpx.Client(timeout=15) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()


def send_notification(items: list, cfg: dict, settings: dict,
                      drops: list | None = None) -> None:
    """
    Send a Telegram notification bundling new items and price drops.

    Args:
        items:    list of new item dicts (id, title, price, url, optional fields)
        cfg:      monitor config (provides label for the header)
        settings: credentials dict with keys telegram_bot_token, telegram_chat_id
        drops:    optional list of price-drop event dicts
    """
    drops = drops or []
    if not items and not drops:
        return

    token   = settings.get("telegram_bot_token")
    chat_id = settings.get("telegram_chat_id")
    if not token or not chat_id:
        # Telegram not configured → skip notification but DON'T crash the
        # monitor. Tracking (seen IDs, history, stats) still proceeds; the
        # user just won't get a push until they fill in settings.json.
        print("  ⚠ Telegram non configurato (telegram_bot_token / "
              "telegram_chat_id mancanti) — notifica saltata, monitor continua",
              flush=True)
        return

    label = _esc(cfg.get("label", ""))
    if drops and items:
        header = f"🔔 <b>{len(items)} nuovi + {len(drops)} ribassati</b> — {label}"
    elif drops:
        header = f"📉 <b>{len(drops)} prezzo/i ridotto/i</b> — {label}"
    else:
        header = f"🔔 <b>{len(items)} nuovo/i</b> — {label}"

    lines: list[str] = []
    if drops:
        lines.append("\n📉 <b>PREZZI RIDOTTI</b>")
        for d in drops:
            lines.append(_fmt_drop(d))
    if items:
        lines.append("\n🆕 <b>NUOVI ANNUNCI</b>")
        for it in items:
            lines.append(_fmt_new(it))

    try:
        for chunk in _chunks(lines, header):
            _send(token, chat_id, chunk)
    except Exception as e:
        # Network error, bad token, Telegram down, etc. — never crash the
        # monitor over a failed push. The items are still tracked.
        print(f"  ⚠ invio Telegram fallito ({type(e).__name__}: {e}) — "
              f"monitor continua", flush=True)
        return

    tot   = len(items) + len(drops)
    extra = f" ({len(drops)} drop)" if drops else ""
    print(f"  → Telegram sent with {tot} entries{extra}", flush=True)
