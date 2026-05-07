"""
Email notifications.

send_email() — sends an email listing all newly found items.
Credentials are read from settings.json (never hardcoded).
"""

import smtplib
from email.mime.text import MIMEText


def send_email(items: list, cfg: dict, settings: dict):
    """
    Send an email notification for a list of newly found items.

    Args:
        items:    list of item dicts (id, title, price, url, optional discount fields)
        cfg:      monitor config dict (provides email_subject and email_intro)
        settings: credentials dict with keys smtp_user, smtp_pass, email_to
    """
    lines = []
    for it in items:
        label  = it.get("label", "")
        disc   = it.get("discount_str", "")
        prefix = f"{label} — " if label else ""
        suffix = f"  {disc}"   if disc  else ""
        lines.append(f"{prefix}[{it['price']}€] {it['title']}{suffix}")
        lines.append(f"  {it['url']}\n")

    body    = cfg["email_intro"] + "\n\n" + "\n".join(lines)
    subject = cfg["email_subject"].format(count=len(items))

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = settings["smtp_user"]
    msg["To"]      = settings["email_to"]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(settings["smtp_user"], settings["smtp_pass"])
        server.sendmail(settings["smtp_user"], settings["email_to"], msg.as_string())

    print(f"  → Email sent with {len(items)} new listings", flush=True)
