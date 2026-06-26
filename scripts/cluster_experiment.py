"""Offline experiment: evaluate candidate clustering keys against the real
shared DB WITHOUT writing anything. Lets us iterate on a deterministic
brand+model canonical key and compare metrics to the current state.

Run:  python3 scripts/cluster_experiment.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
import collections
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db import shared_db_path, connect


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# product-type synonyms across IT/FR/ES/EN/DE/PT -> one canonical token
TYPE_SYNONYMS = {
    "game": ["game", "games", "jeu", "jeux", "gioco", "giochi", "juego",
             "juegos", "spiel", "spiele", "videogioco", "videogame",
             "videogames", "jogo", "jogos"],
    "console": ["console", "consolle", "konsole"],
    "headphones": ["headphones", "headphone", "casque", "casques", "ecouteur",
                   "ecouteurs", "auriculares", "auricolari", "cuffie", "cuffia",
                   "kopfhorer", "earphones", "earphone", "earbuds", "headset",
                   "auricolare", "fones"],
    "speaker": ["speaker", "speakers", "enceinte", "enceintes", "altavoz",
                "altavoces", "altoparlante", "cassa", "casse", "lautsprecher",
                "soundbar", "colonna"],
    "charger": ["charger", "chargeur", "cargador", "caricabatterie",
                "caricatore", "ladegerat", "caricabatteria", "carregador"],
    "cable": ["cable", "cables", "cavo", "cavi", "kabel", "cordon"],
    "case": ["case", "cover", "custodia", "funda", "coque", "hulle", "etui",
             "capa"],
    "phone": ["phone", "smartphone", "telephone", "telefono", "movil",
              "handy", "cellulare", "telemovel", "telefonino"],
    "laptop": ["laptop", "portatile", "portable", "ordinateur", "notebook"],
    "tablet": ["tablet", "tablette", "tableta"],
    "tv": ["tv", "televisore", "televisor", "television", "televiseur",
           "fernseher"],
    "camera": ["camera", "fotocamera", "camara", "kamera", "reflex", "appareil"],
    "watch": ["watch", "montre", "orologio", "reloj", "uhr", "smartwatch"],
    "printer": ["printer", "imprimante", "stampante", "impresora", "drucker"],
    "mouse": ["mouse", "souris", "raton"],
    "keyboard": ["keyboard", "clavier", "tastiera", "teclado", "tastatur"],
    "battery": ["battery", "batterie", "batteria", "bateria", "akku", "pile",
                "pilas", "pila"],
    "drone": ["drone", "drones", "drohne"],
    "controller": ["controller", "manette", "mando", "joystick", "joypad",
                   "controllo"],
    "monitor": ["monitor", "ecran", "schermo", "pantalla", "bildschirm"],
    "ram": ["ram", "ddr", "ddr3", "ddr4", "ddr5", "sodimm", "dimm"],
    "ssd": ["ssd", "hdd", "nvme"],
    "router": ["router", "modem", "routeur"],
}
TYPE_MAP: dict[str, str] = {}
for canon, syns in TYPE_SYNONYMS.items():
    for s in syns:
        TYPE_MAP[s] = canon

KNOWN_BRANDS = {
    "apple", "iphone", "ipad", "macbook", "airpods", "samsung", "sony",
    "nintendo", "playstation", "ps2", "ps3", "ps4", "ps5", "psp", "xbox",
    "microsoft", "hp", "dell", "lenovo", "asus", "acer", "canon", "nikon",
    "jbl", "bose", "philips", "logitech", "huawei", "xiaomi", "redmi", "lg",
    "panasonic", "gopro", "dyson", "remington", "rowenta", "bosch", "makita",
    "dewalt", "nvidia", "amd", "intel", "razer", "sandisk", "kingston",
    "seagate", "toshiba", "garmin", "fitbit", "amazon", "google", "oneplus",
    "oppo", "realme", "motorola", "nokia", "tcl", "hisense", "marshall",
    "beats", "anker", "tplink", "netgear", "epson", "brother", "kodak",
    "polaroid", "fujifilm", "ghd", "babyliss", "wahl", "cecotec",
}
# brand aliases / normalisation
BRAND_ALIAS = {
    "iphone": "apple", "ipad": "apple", "macbook": "apple", "airpods": "apple",
    "imac": "apple", "redmi": "xiaomi", "ps2": "playstation",
    "ps3": "playstation", "ps4": "playstation", "ps5": "playstation",
    "psp": "playstation", "ps1": "playstation",
}

NOISE = {
    # marketing / condition
    "nuovo", "nuova", "usato", "usata", "perfetto", "perfetta", "ottimo",
    "ottima", "originale", "vintage", "raro", "spedizione", "gratis", "prezzo",
    "trattabile", "scatola", "imballo", "accessori", "completo", "garanzia",
    "ricondizionato", "sigillato", "come", "per", "del", "della", "delle",
    "dei", "degli", "rotto", "rotta", "new", "used", "perfect", "original",
    "with", "without", "free", "shipping", "boxed", "sealed", "refurbished",
    "warranty", "neuf", "neuve", "comme", "etat", "parfait", "tres", "bon",
    "avec", "sans", "neu", "gebraucht", "wie", "gut", "guter", "zustand",
    "ohne", "mit", "and", "et", "und", "lot", "set", "kit", "the", "di", "da",
    "in", "il", "la", "le", "un", "una", "uno", "con", "des", "les", "pour",
    "und", "von", "para", "como", "muy", "buen", "estado", "nuevo", "usado",
    "marca", "senza", "altro", "altri", "vari", "varie", "pezzi", "pezzo",
    "modello", "model", "color", "colore", "couleur", "nero", "bianco",
    "black", "white", "blu", "blue", "rosso", "red", "noir", "blanc",
}


# Model qualifiers worth keeping (language-independent), unlike descriptive
# words. They distinguish models at the same price tier (13 vs 13 Pro).
MODEL_WORDS = {"pro", "max", "plus", "mini", "ultra", "lite", "air", "oled",
               "se", "xl", "gb", "tb", "go", "fe", "neo", "edge", "active"}


def _tokenize(title: str, brand: str):
    t = strip_accents((title or "").lower())
    t = re.sub(r"[^a-z0-9+]", " ", t)
    tokens = [tok for tok in t.split() if tok]
    b = strip_accents((brand or "").lower()).strip()
    b = re.sub(r"[^a-z0-9]", "", b)
    if b in ("senzamarca", "nomark", "none", "sansmarque", "altro", ""):
        b = ""
    if not b:
        b = next((tok for tok in tokens if tok in KNOWN_BRANDS), "")
    b = BRAND_ALIAS.get(b, b)
    return tokens, b


def strong_key(title: str, brand: str) -> str | None:
    """Deterministic, language-independent key for items with a clear signal:
    brand + canonical product-type + model numbers/qualifiers (sorted, so word
    order & language don't matter). Returns None when the signal is too weak
    (no brand, or no type AND no model number) → caller routes to embeddings."""
    tokens, b = _tokenize(title, brand)
    if not b:
        return None
    types, nums, quals = [], [], []
    for tok in tokens:
        if tok == b or BRAND_ALIAS.get(tok) == b:
            continue
        if tok in TYPE_MAP:
            ct = TYPE_MAP[tok]
            if ct not in types:
                types.append(ct)
        elif tok in MODEL_WORDS:
            quals.append(tok)
        elif re.fullmatch(r"[a-z]*\d[a-z0-9]*", tok) and len(tok) <= 7 and tok not in NOISE:
            nums.append(tok)
    if not types and not nums:
        return None
    parts = [b] + sorted(set(types))[:1] + sorted(set(nums))[:2] + sorted(set(quals))[:2]
    return " ".join(parts)[:60]


def strong_key_v2(title: str, brand: str) -> str | None:
    """Like strong_key but ALSO highlights the model when there is no model
    NUMBER: it appends the single most distinctive word (longest informative
    token, usually a proper noun like a game name or a line name 'switch').
    This splits brand+type dumping grounds (all nintendo games) by model while
    staying language-independent for proper nouns."""
    tokens, b = _tokenize(title, brand)
    if not b:
        return None
    types, nums, quals, words = [], [], [], []
    for tok in tokens:
        if tok == b or BRAND_ALIAS.get(tok) == b or tok in NOISE:
            continue
        if tok in TYPE_MAP:
            ct = TYPE_MAP[tok]
            if ct not in types:
                types.append(ct)
        elif tok in MODEL_WORDS:
            quals.append(tok)
        elif re.fullmatch(r"[a-z]*\d[a-z0-9]*", tok) and len(tok) <= 7:
            nums.append(tok)
        elif len(tok) >= 3:
            words.append(tok)
    if not types and not nums and not words:
        return None
    if not types and not nums:
        # only descriptive words + brand: weak (cross-language risk) → embeddings
        return None
    parts = [b] + sorted(set(types))[:1] + sorted(set(nums))[:2] + sorted(set(quals))[:2]
    if not nums and words:
        # no number to pin the model → add the most distinctive word
        disc = sorted(set(words), key=lambda w: (-len(w), w))[0]
        parts.append(disc)
    return " ".join(parts)[:60]


def canonical_key(title: str, brand: str, *, n_models: int = 2) -> str:
    """brand + canonical product-type + sorted model tokens (order-independent)."""
    t = strip_accents((title or "").lower())
    t = re.sub(r"[^a-z0-9+]", " ", t)
    tokens = [tok for tok in t.split() if tok]

    b = strip_accents((brand or "").lower()).strip()
    b = re.sub(r"[^a-z0-9]", "", b)
    if b in ("senzamarca", "nomark", "none", "sansmarque", "altro", ""):
        b = ""
    if not b:
        b = next((tok for tok in tokens if tok in KNOWN_BRANDS), "")
    b = BRAND_ALIAS.get(b, b)

    types: list[str] = []
    models: list[str] = []
    for tok in tokens:
        if tok in NOISE or tok == b:
            continue
        if tok in BRAND_ALIAS and BRAND_ALIAS[tok] == b:
            continue
        if tok in TYPE_MAP:
            ct = TYPE_MAP[tok]
            if ct not in types:
                types.append(ct)
        elif re.fullmatch(r"[a-z]*\d[a-z0-9]*", tok) and len(tok) <= 7:
            # model-number-like token: 13, a14, 20v, i5, rtx3080
            models.append(tok)
        elif len(tok) >= 3 and tok not in KNOWN_BRANDS:
            models.append(tok)

    # de-dupe preserving order, then take the most "distinctive" few, SORTED
    seen = set()
    uniq_models = [m for m in models if not (m in seen or seen.add(m))]
    # prefer model-number tokens (contain a digit) first
    nums = [m for m in uniq_models if re.search(r"\d", m)]
    words = [m for m in uniq_models if not re.search(r"\d", m)]
    chosen = sorted(nums)[:n_models] + sorted(words)[:max(0, n_models - len(nums[:n_models]) + 1)]
    chosen = sorted(set(chosen))

    parts = [p for p in [b] + sorted(set(types))[:1] + chosen if p]
    key = " ".join(parts).strip()
    if not key:
        # nothing usable: fall back to first 3 informative tokens
        info = [tok for tok in tokens if tok not in NOISE and len(tok) >= 3]
        key = " ".join(sorted(info[:3])) or "unknown"
    return key[:60]


def report(by: dict, name: str, n_items: int):
    sizes = [len(v) for v in by.values()]
    nclus = len(sizes)
    sings = sum(1 for s in sizes if s == 1)
    useful = sum(s for s in sizes if s >= 2)
    big = max(by.items(), key=lambda kv: len(kv[1]))
    cvs, incoh = [], 0
    for k, v in by.items():
        if len(v) < 5:
            continue
        pr = [it["price"] for it in v if it["price"]]
        if len(pr) > 1 and st.mean(pr):
            cv = st.pstdev(pr) / st.mean(pr)
            cvs.append(cv)
            if cv > 0.9:
                incoh += 1
    print(f"\n=== {name} ===")
    print(f"  cluster: {nclus}  singleton: {sings} ({100*sings/max(1,nclus):.0f}% cl, "
          f"{100*sings/n_items:.0f}% item)")
    print(f"  item in cluster utili(>=2): {useful} ({100*useful/n_items:.0f}%)")
    print(f"  max cluster: {len(big[1])} ('{big[0][:40]}')")
    print(f"  CV mediano (n>=5): {st.median(cvs) if cvs else 0:.2f}  "
          f"incoerenti CV>0.9: {incoh}/{len(cvs)} ({100*incoh/max(1,len(cvs)):.0f}%)")


def hybrid_keys(items, threshold=0.80, log=print, keyfn=strong_key):
    """Strong lexical key where possible; embedding clustering for the rest,
    seeded so weak items can attach to existing strong clusters."""
    import numpy as np
    from core.clustering import _get_model, _embed_text

    keys: dict[int, str] = {}
    strong_groups: dict[str, list[dict]] = collections.defaultdict(list)
    weak: list[dict] = []
    for it in items:
        k = keyfn(it["title"], it["brand"])
        if k:
            strong_groups[k].append(it)
            keys[it["id"]] = k
        else:
            weak.append(it)
    log(f"  strong-key groups: {len(strong_groups)}  weak items: {len(weak)}")

    # Seed centroids: embed a medoid-ish vector per strong group (mean of its
    # member embeddings). Compute by embedding all strong items once.
    model = _get_model()
    strong_items = [it for g in strong_groups.values() for it in g]
    if strong_items:
        se = model.encode([_embed_text(it) for it in strong_items],
                          show_progress_bar=False, convert_to_numpy=True,
                          batch_size=256)
        se = se / np.maximum(np.linalg.norm(se, axis=1, keepdims=True), 1e-12)
        idx = 0
        cent: dict[str, np.ndarray] = {}
        cnt: dict[str, int] = {}
        for k, g in strong_groups.items():
            vecs = se[idx:idx+len(g)]; idx += len(g)
            cent[k] = vecs.mean(axis=0)
            cnt[k] = len(g)
    else:
        cent, cnt = {}, {}

    # Incremental embedding clustering for weak items, attaching to strong
    # centroids when close enough, else forming new embedding clusters.
    if weak:
        we = model.encode([_embed_text(it) for it in weak],
                          show_progress_bar=False, convert_to_numpy=True,
                          batch_size=256)
        we = we / np.maximum(np.linalg.norm(we, axis=1, keepdims=True), 1e-12)
        cent_keys = list(cent.keys())
        cent_mat = np.array([cent[k] for k in cent_keys]) if cent_keys else None
        for it, emb in zip(weak, we):
            best_k, best_s = None, -1.0
            if cent_mat is not None:
                sims = cent_mat @ emb
                j = int(sims.argmax())
                best_s = float(sims[j]); best_k = cent_keys[j]
            if best_k is not None and best_s >= threshold:
                keys[it["id"]] = best_k
                # update centroid (running mean)
                c = cent[best_k]; n = cnt[best_k]
                cent[best_k] = (c * n + emb) / (n + 1); cnt[best_k] = n + 1
            else:
                nk = "emb:" + _embed_short(it["title"])
                base = nk; n = 0
                while nk in cent:
                    n += 1; nk = f"{base}#{n}"
                keys[it["id"]] = nk
                cent[nk] = emb; cnt[nk] = 1
                cent_keys.append(nk)
                cent_mat = np.vstack([cent_mat, emb]) if cent_mat is not None else emb.reshape(1, -1)
    return keys


def coherent_coverage(by: dict, n_items: int) -> float:
    """Fraction of items that live in a USABLE cluster: size>=2 AND coherent
    (size<5 treated coherent; else CV<=0.9). The single number to maximise."""
    good = 0
    for v in by.values():
        if len(v) < 2:
            continue
        pr = [it["price"] for it in v if it["price"]]
        if len(v) < 5:
            good += len(v)
        elif len(pr) > 1 and st.mean(pr) and st.pstdev(pr) / st.mean(pr) <= 0.9:
            good += len(v)
    return 100 * good / n_items


def final_keys(items, threshold=0.80, split_min=30, log=print, keyfn=strong_key_v2):
    """The full pipeline:
      1. strong_key_v2 → brand+type+model groups (model highlighted).
      2. embedding sub-split of any strong group larger than split_min
         (breaks PS2/Switch ecosystems into console/games/accessories).
      3. weak items → embedding clustering, attaching to strong centroids.
    """
    import numpy as np
    from core.clustering import (_get_model, _embed_text,
                                 _make_embedding_cluster_key, _get_hdbscan)

    keys: dict[int, str] = {}
    strong: dict[str, list[dict]] = collections.defaultdict(list)
    weak: list[dict] = []
    for it in items:
        k = keyfn(it["title"], it["brand"])
        (strong[k].append(it) if k else weak.append(it))
    log(f"  strong groups:{len(strong)} weak:{len(weak)}")

    model = _get_model()

    def embed(lst):
        e = model.encode([_embed_text(it) for it in lst], show_progress_bar=False,
                         convert_to_numpy=True, batch_size=256)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)

    # 1+2: assign strong keys, sub-splitting big groups
    used = set()
    cent: dict[str, "np.ndarray"] = {}
    cnt: dict[str, int] = {}
    H = _get_hdbscan().HDBSCAN
    big_items = [it for k, g in strong.items() if len(g) > split_min for it in g]
    big_emb = embed(big_items) if big_items else None
    bi = 0
    for k, g in strong.items():
        if len(g) <= split_min:
            for it in g:
                keys[it["id"]] = k
            continue
        emb = big_emb[bi:bi+len(g)]; bi += len(g)
        labels = H(min_cluster_size=2, metric="euclidean",
                   cluster_selection_method="leaf", core_dist_n_jobs=1).fit_predict(emb)
        bylab = collections.defaultdict(list)
        for i, l in enumerate(labels):
            bylab[int(l)].append(i)
        if len(bylab) <= 1:               # no useful split → keep whole
            for it in g:
                keys[it["id"]] = k
            continue
        sub_cent = []
        for l, idxs in bylab.items():
            if l == -1:
                continue
            v = emb[idxs].mean(axis=0)
            medoid = g[idxs[int(np.argmin(np.linalg.norm(emb[idxs]-v, axis=1)))]]
            tok = _make_embedding_cluster_key(medoid["title"]).split()
            sub = f"{k} · {tok[0] if tok else l}"
            base = sub; c = 0
            while sub in used:
                c += 1; sub = f"{base}#{c}"
            used.add(sub)
            vn = v / np.maximum(np.linalg.norm(v), 1e-12)
            sub_cent.append((sub, vn))
            for i in idxs:
                keys[g[i]["id"]] = sub
        # noise inside big group → nearest sub or keep parent key
        cm = np.array([c for _, c in sub_cent]) if sub_cent else None
        ck = [s for s, _ in sub_cent]
        for i, l in enumerate(labels):
            if l != -1:
                continue
            if cm is not None:
                sims = cm @ emb[i]; j = int(sims.argmax())
                if sims[j] >= threshold:
                    keys[g[i]["id"]] = ck[j]; continue
            keys[g[i]["id"]] = k

    # rebuild centroids over ALL assigned-so-far keys for weak attachment
    assigned = [it for it in items if it["id"] in keys]
    ae = embed(assigned)
    for it, e in zip(assigned, ae):
        k = keys[it["id"]]
        if k in cent:
            cent[k] = (cent[k]*cnt[k] + e)/(cnt[k]+1); cnt[k] += 1
        else:
            cent[k] = e; cnt[k] = 1

    # 3: weak items → attach or new
    if weak:
        we = embed(weak)
        ck = list(cent.keys())
        cm = np.array([cent[k] for k in ck]) if ck else None
        for it, e in zip(weak, we):
            best_k, best_s = None, -1.0
            if cm is not None:
                sims = cm @ e; j = int(sims.argmax()); best_s = float(sims[j]); best_k = ck[j]
            if best_k is not None and best_s >= threshold:
                keys[it["id"]] = best_k
                cent[best_k] = (cent[best_k]*cnt[best_k] + e)/(cnt[best_k]+1); cnt[best_k] += 1
            else:
                nk = "emb:" + _embed_short(it["title"]); base = nk; c = 0
                while nk in cent:
                    c += 1; nk = f"{base}#{c}"
                keys[it["id"]] = nk; cent[nk] = e; cnt[nk] = 1
                ck.append(nk); cm = np.vstack([cm, e]) if cm is not None else e.reshape(1, -1)
    return keys


def price_split(by: dict, min_size=8, cv_thresh=0.8, max_bands=4,
                gap_log=0.5, quantile_fallback=True):
    """Split incoherent clusters (CV>thresh) into price bands. First tries
    largest log-gap cuts (a €80 console vs €15 games); if no clear gap but the
    spread is still high, falls back to quantile bands so even continuous
    dumping grounds become price-coherent."""
    import math
    out = {}
    for k, v in by.items():
        pr = [it["price"] for it in v if it["price"]]
        if len(v) < min_size or len(pr) < 4 or not st.mean(pr):
            out[k] = v; continue
        cv = st.pstdev(pr) / st.mean(pr)
        if cv <= cv_thresh:
            out[k] = v; continue
        sortable = sorted([it for it in v if it["price"]], key=lambda it: it["price"])
        noprice = [it for it in v if not it["price"]]
        lp = [math.log(it["price"]) for it in sortable]
        gaps = sorted(((lp[i+1]-lp[i], i) for i in range(len(lp)-1)), reverse=True)
        cuts = sorted(i for g, i in gaps[:max_bands-1] if g > gap_log)
        if not cuts and quantile_fallback:
            # no clean gap but spread is wide → cut into ~log-even bands
            nb = min(max_bands, max(2, round(cv / 0.5)))
            step = len(sortable) / nb
            cuts = sorted({int(step*(b+1))-1 for b in range(nb-1)})
            cuts = [c for c in cuts if 0 <= c < len(sortable)-1]
        if not cuts:
            out[k] = v; continue
        bands, prev = [], 0
        for ci in cuts + [len(sortable)-1]:
            if ci >= prev:
                bands.append(sortable[prev:ci+1]); prev = ci+1
        for band in bands:
            if band:
                lo = int(band[0]["price"]); hi = int(band[-1]["price"])
                out[f"{k} [{lo}-{hi}€]"] = band
        if noprice:
            out.setdefault(f"{k} [?]", []).extend(noprice)
    return out


def _embed_short(title: str) -> str:
    t = strip_accents((title or "").lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    toks = [x for x in t.split() if x not in NOISE and len(x) >= 3]
    return " ".join(sorted(toks[:3])) or "x"


if __name__ == "__main__":
    with connect(shared_db_path()) as c:
        items = [dict(r) for r in c.execute(
            "SELECT id, title, brand, price, category, cluster_key "
            "FROM items WHERE removed=0 AND title IS NOT NULL")]
    n = len(items)
    print(f"items: {n}")

    by_cur = collections.defaultdict(list)
    for it in items:
        by_cur[it["cluster_key"]].append(it)
    report(by_cur, "ATTUALE (DB embedding+HDBSCAN)", n)

    # strong-key only, measured over the WHOLE set (None -> its own bucket key)
    by_strong = collections.defaultdict(list)
    for it in items:
        by_strong[strong_key(it["title"], it["brand"]) or f"__weak_{it['id']}"].append(it)
    report(by_strong, "SOLO chiave forte (deboli isolati)", n)

    # full hybrid
    print("\n[hybrid] embedding pass…")
    hk = hybrid_keys(items, threshold=0.80)
    by_h = collections.defaultdict(list)
    for it in items:
        by_h[hk[it["id"]]].append(it)
    report(by_h, "IBRIDO (chiave forte + embedding sul resto, thr=0.80)", n)
