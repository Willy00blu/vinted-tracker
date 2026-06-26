"""
Synthetic clustering test: does adding the Vinted category to the embedding
text actually separate items that the bare title/brand collapses together?

We hand-craft a labelled dataset inspired by the REAL contamination seen in
the production DB (the "tappi spechietti / loss of vision" junk cluster, which
mixed hair tools + controllers + mice + amps + games — all DIFFERENT Vinted
categories). Each row has a ground-truth `group` (what SHOULD cluster together).

We run cluster_hdbscan twice — WITHOUT category vs WITH category in the embed
text — and measure:
  - purity:        avg fraction of each cluster that is its majority group
  - n_clusters:    how many clusters formed
  - contamination: pairs of items from DIFFERENT groups landing in the SAME
                   cluster (lower = better separation)

Run: python3 scripts/synthetic_cluster_test.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.clustering  import cluster_hdbscan  # noqa: E402
from core.category_db import SCHEMA           # noqa: E402


# (title, brand, category, ground_truth_group)
# Categories are the real Vinted leaf names we'd extract from the breadcrumb.
DATA = [
    # ── hair styling tools (multilingual — the embedding SHOULD merge these) ──
    ("Piastra capelli",        "Rowenta",   "hair styling tools", "hair"),
    ("Lisseur cheveux",        "",          "hair styling tools", "hair"),
    ("Lockenstab",             "",          "hair styling tools", "hair"),
    ("Secador de pelo",        "Rowenta",   "hair styling tools", "hair"),
    ("Plancha de pelo",        "Cecotec",   "hair styling tools", "hair"),
    ("Fer à boucler",          "BaByliss",  "hair styling tools", "hair"),
    ("Glätteisen",             "Remington", "hair styling tools", "hair"),
    # ── game controllers (were contaminating the hair cluster) ──
    ("Manette Scuf",           "Scuf",      "controllers",        "controller"),
    ("Manette PS4 sans fil",   "Sony",      "controllers",        "controller"),
    ("Controller Xbox Series", "Microsoft", "controllers",        "controller"),
    ("Mando PS5 DualSense",    "Sony",      "controllers",        "controller"),
    # ── computer mice ──
    ("Mionix Castor",          "Mionix",    "mice",               "mouse"),
    ("Mouse Logitech MX",      "Logitech",  "mice",               "mouse"),
    ("Souris gaming Razer",    "Razer",     "mice",               "mouse"),
    # ── hi-fi amplifiers ──
    ("Ampli hifi NAD",         "NAD",       "amplifiers",         "amp"),
    ("Amplificatore Yamaha",   "Yamaha",    "amplifiers",         "amp"),
    # ── headphones ──
    ("Cuffie Dyson",           "Dyson",     "headphones",         "headphone"),
    ("Écouteurs sans fil",     "Sony",      "headphones",         "headphone"),
    ("AirPods Pro 2",          "Apple",     "headphones",         "headphone"),
    ("Auriculares bluetooth",  "JBL",       "headphones",         "headphone"),
    # ── video games (were contaminating too) ──
    ("VR Worlds PS4",          "Sony",      "video games",        "game"),
    ("Wario Ware Switch",      "Nintendo",  "video games",        "game"),
    ("Ninja Gaiden PS4",       "Koei",      "video games",        "game"),
    ("FIFA 23 PS5",            "EA",        "video games",        "game"),
    # ── mobile phones (trivial — distinct, should cluster by model) ──
    ("iPhone 13 Pro 128gb",    "Apple",     "mobile phones",      "iphone13"),
    ("iPhone 13 Pro Max",      "Apple",     "mobile phones",      "iphone13"),
    ("iPhone 13 256gb",        "Apple",     "mobile phones",      "iphone13"),
    ("iPhone 15 Pro",          "Apple",     "mobile phones",      "iphone15"),
    ("iPhone 15 Pro Max blue", "Apple",     "mobile phones",      "iphone15"),
    # ── consoles (trivial distinct) ──
    ("PlayStation 5 Slim",     "Sony",      "consoles",           "ps5"),
    ("PS5 Digital Edition",    "Sony",      "consoles",           "ps5"),
    ("Nintendo Switch OLED",   "Nintendo",  "consoles",           "switch"),
    ("Nintendo Switch Lite",   "Nintendo",  "consoles",           "switch"),
    # ── cameras ──
    ("Nikon Coolpix S3100",    "Nikon",     "cameras",            "camera"),
    ("Canon EOS 650D",         "Canon",     "cameras",            "camera"),
    ("Polaroid Go Gen 2",      "Polaroid",  "cameras",            "camera"),
    # ── singletons / generic (should stay alone or be noise) ──
    ("Cavetto Alexa",          "Amazon",    "other accessories",  "_single1"),
    ("UV Resin Kit",           "",          "other accessories",  "_single2"),
    ("Zaino laptop",           "",          "bags",               "_single3"),
]


def run(use_category: bool) -> dict:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    items = []
    for i, (title, brand, cat, grp) in enumerate(DATA):
        items.append({
            "id":       i,
            "title":    title,
            "brand":    brand,
            "category": cat if use_category else None,
        })
    mapping = cluster_hdbscan(items, conn=conn, min_cluster_size=2,
                              log_fn=lambda s: None)
    conn.close()
    return mapping


def run_partitioned() -> dict:
    """Partition by category, cluster (brand+title, NO category in text) within
    each partition, then namespace the cluster keys by category so different
    categories can never share a cluster."""
    by_cat: dict[str, list[dict]] = {}
    for i, (title, brand, cat, grp) in enumerate(DATA):
        by_cat.setdefault(cat, []).append(
            {"id": i, "title": title, "brand": brand, "category": None})
    mapping: dict[int, str] = {}
    for cat, items in by_cat.items():
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        m = cluster_hdbscan(items, conn=conn, min_cluster_size=2,
                            log_fn=lambda s: None)
        conn.close()
        for iid, ck in m.items():
            mapping[iid] = f"{cat}/{ck}"   # namespace by category
    return mapping


def evaluate(mapping: dict, label: str):
    groups = {i: DATA[i][3] for i in range(len(DATA))}
    # cluster → list of groups
    by_cluster: dict[str, list[str]] = {}
    for i, ck in mapping.items():
        by_cluster.setdefault(ck, []).append(groups[i])

    # Purity: avg over clusters of (majority group count / cluster size)
    purities = []
    for ck, grps in by_cluster.items():
        c = Counter(grps)
        purities.append(c.most_common(1)[0][1] / len(grps))
    purity = sum(purities) / len(purities) if purities else 0

    # Contamination: pairs in same cluster but different ground-truth group
    contam = 0
    for i, j in combinations(mapping.keys(), 2):
        if mapping[i] == mapping[j] and groups[i] != groups[j]:
            contam += 1

    # Fragmentation: same group split across clusters (bad for the hair group)
    group_clusters: dict[str, set] = {}
    for i, ck in mapping.items():
        group_clusters.setdefault(groups[i], set()).add(ck)
    fragmented = sum(1 for g, cks in group_clusters.items()
                     if not g.startswith("_single") and len(cks) > 1)

    print(f"\n── {label} ──")
    print(f"  clusters formed:      {len(by_cluster)}")
    print(f"  purity:               {purity:.3f}  (1.0 = perfetto)")
    print(f"  cross-group contam.:  {contam} coppie sbagliate (meno = meglio)")
    print(f"  groups fragmented:    {fragmented}")
    # Show the hair cluster specifically
    hair_items = [i for i in mapping if groups[i] == "hair"]
    if hair_items:
        hair_clusters = Counter(mapping[i] for i in hair_items)
        print(f"  hair tools → {len(hair_clusters)} cluster(s): "
              f"{dict(hair_clusters)}")
        # Anything non-hair sharing the hair cluster?
        hair_cluster_keys = set(mapping[i] for i in hair_items)
        intruders = [DATA[i][0] for i in mapping
                     if mapping[i] in hair_cluster_keys and groups[i] != "hair"]
        print(f"  intrusi nel cluster hair: {intruders if intruders else 'nessuno ✓'}")
    return purity, contam


def main():
    print(f"Synthetic dataset: {len(DATA)} item, "
          f"{len(set(d[3] for d in DATA))} gruppi ground-truth")
    m_no  = run(use_category=False)
    evaluate(m_no, "SENZA categoria (brand + titolo)")
    m_yes = run(use_category=True)
    evaluate(m_yes, "CON categoria nel testo (categoria + brand + titolo)")
    m_part = run_partitioned()
    evaluate(m_part, "PARTIZIONATO per categoria (cluster brand+titolo dentro ogni cat.)")


if __name__ == "__main__":
    main()
