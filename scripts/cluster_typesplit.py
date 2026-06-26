"""Test a TYPE-split: inside a cluster, separate items by product type detected
from the title (game / console / controller / accessory ...). Targets the
'PS4 console+games' ecosystem clusters. Reuses cached e5 embeddings.

    python3 scripts/cluster_typesplit.py
"""
import sys, collections, statistics as st, re, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import os
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
from core.category_db import shared_db_path, connect
import core.clustering as cl
from scripts.cluster_experiment import _tokenize
from scripts.cluster_ab2 import base_partition, metrics, CACHE

# product-type lexicon (multilingual) — order = priority
TYPE_WORDS = {
    "game": ["jeu", "jeux", "game", "games", "gioco", "giochi", "juego",
             "juegos", "spiel", "spiele", "jogo", "videogioco", "videogame"],
    "controller": ["manette", "controller", "mando", "joystick", "joypad",
                   "dualshock", "dualsense", "joycon", "joy-con"],
    "console": ["console", "consolle", "konsole"],
    "case": ["cover", "custodia", "funda", "coque", "hulle", "case", "etui"],
    "charger": ["chargeur", "caricabatterie", "cargador", "charger", "ladekabel"],
    "headset": ["casque", "headset", "cuffie", "auriculares", "ecouteurs"],
    "cable": ["cavo", "cable", "kabel", "hdmi"],
}
TYPEMAP = {}
for t, ws in TYPE_WORDS.items():
    for w in ws:
        TYPEMAP[w] = t


def _detect_type(title):
    t = unicodedata.normalize("NFD", (title or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    toks = re.sub(r"[^a-z0-9]", " ", t).split()
    for tok in toks:
        if tok in TYPEMAP:
            return TYPEMAP[tok]
    return None


def type_split(mapping, items, min_type=2):
    """Split a cluster when it holds >=2 distinct detected types (each >=min_type
    items). Typed items get 'key · type'; untyped keep the base key."""
    by = collections.defaultdict(list)
    for it in items:
        by[mapping[it["id"]]].append(it)
    out = dict(mapping)
    n_split = 0
    for k, v in by.items():
        if len(v) < 4:
            continue
        types = collections.Counter(_detect_type(it["title"]) for it in v)
        strong = [t for t, c in types.items() if t and c >= min_type]
        if len(strong) < 2:
            continue
        n_split += 1
        for it in v:
            tt = _detect_type(it["title"])
            if tt in strong:
                out[it["id"]] = f"{k} · {tt}"
            # untyped or rare-type → keep base key k
    return out, n_split


def main():
    d = np.load(CACHE, allow_pickle=True)
    emb = {int(i): vv for i, vv in zip(d["ids"], d["emb"])}
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id,title,brand,price,category FROM items "
            "WHERE removed=0 AND title IS NOT NULL")]
    items = [it for it in items if it["id"] in emb]
    print(f"items: {len(items)}\n")

    with connect(Path("/tmp/typesplit.db")) as conn:
        base = base_partition(items, emb, "catbrand")
        m1 = cl.refine_clusters(items, dict(base), conn, log_fn=lambda *a: None, emb_cache=emb)
        metrics(m1, items, "#1 cat+brand (baseline)")
        mts, ns = type_split(m1, items)
        metrics(mts, items, f"#1 + type-split (split {ns} clusters)")

    # eyeball: a PS4-ish ecosystem cluster, before vs after
    by1 = collections.defaultdict(list)
    for it in items:
        by1[m1[it["id"]]].append(it)
    target = max((kv for kv in by1.items() if "ps4" in kv[0].lower() or "ps3" in kv[0].lower()
                  or "xbox" in kv[0].lower() or "switch" in kv[0].lower()),
                 key=lambda kv: len(kv[1]), default=(None, []))
    if target[0]:
        k, v = target
        print(f"\n=== PRIMA: [{k[:46]}] n={len(v)} ===")
        for it in v[:10]:
            print(f"   €{it['price'] or 0:>4} [{_detect_type(it['title']) or '—':>10}] {(it['title'] or '')[:40]}")
        byts = collections.defaultdict(list)
        for it in v:
            byts[mts[it["id"]]].append(it)
        print(f"\n=== DOPO type-split → {len(byts)} sotto-cluster ===")
        for sk, sv in sorted(byts.items(), key=lambda kv: -len(kv[1])):
            pr = [it["price"] for it in sv if it["price"]]
            med = st.median(pr) if pr else 0
            print(f"   [{sk[-22:]}] n={len(sv)} med€{med:.0f}")


if __name__ == "__main__":
    main()
