"""
Title clustering for category monitors.

Two-stage approach:

  1. Embedding-based incremental clustering. We embed each title with a
     multilingual sentence-transformer, compare it (cosine) to existing
     centroids stored in the DB. If best similarity is >= SIMILARITY_THRESHOLD
     we assign the existing cluster and update the centroid via moving-average.
     Otherwise we create a new cluster. This keeps cluster_ids STABLE across
     scrape cycles, which is the hard part of clustering streaming data.

  2. Naive fallback (brand + first informative tokens) used when:
       - sentence-transformers isn't installed, OR
       - the caller doesn't pass a DB connection.
     Stable by construction.

History: a third stage of ~25 hand-written regex rules used to live on top
(catching "iphone 13 pro" / "ps5" / etc.). It was removed after an A/B test
(scripts/clustering_ab_test.py, 2026-05-19) showed embedding-only produced
+20% more useful clusters with the same singleton rate, and better
price-coherence (PS5 controllers vs PS5 consoles cluster separately, instead
of being collapsed by the regex). It was also pointless for non-tech
categories like Moda / Casa / Sport.
"""

from __future__ import annotations

import datetime
import os
import re
import threading
import unicodedata

# Embedding model is configurable via env. Default: multilingual-e5-large — the
# higher-quality model (A/B on the real 56k DB: +16pp coverage, +17pp price
# coherence at equal brand-purity vs MiniLM). The small/fast MiniLM is kept as
# the lightweight ALTERNATIVE, selectable via:
#   VINTED_EMBED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
# e5 models need a "query: " prefix on inputs (handled in _embed_text). Since
# clustering now runs only in the (subprocess) regenerate, e5's size/speed cost
# is paid there, not in the always-on monitor.
MODEL_NAME = os.environ.get("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
_NEEDS_E5_PREFIX = "e5" in MODEL_NAME.lower()

# Embedding cache versioning. Bump when:
#   - the model name changes (e5-large → other)
#   - the noise stopword list changes (affects _embed_text output)
#   - the _embed_text logic changes (brand prepend, normalization)
# The text_worker / image_worker re-embed gradually whenever DB rows have a
# different version than these constants. Mixing vectors from different
# models in the same DB silently breaks clustering, hence the discipline.
TEXT_EMB_VERSION  = "v10-e5large-sprint8-20260531"   # sprint-8: CPU/tech noise expansion (ghz/mhz/cpu/core/processore/lga/ddr/...) + clock-frequency strip pattern (\d+.\d+ GHz). Fixes CPU cluster contamination (intel i3/i5/i7 + Xeon + Pentium tutti mischiati).

# Numerical price feature: concat log(price) z-scored (within partition) to the
# title embedding before HDBSCAN. Items with similar prices stay closer in the
# augmented space → cleaner product-vs-accessory separation.
# λ = 0 disables. λ ≈ 1/sqrt(1024) ≈ 0.031 gives roughly equal weight per dim
# (e5-large is 1024-d normalized, so each dim has stdev ~0.031). λ in
# [0.05, 0.20] is the typical useful range. Persisted centroids stay text-only
# (1024-d) for back-compat with `match_cluster_batch` and the GUI display.
PRICE_FEATURE_LAMBDA = 0.30   # 0 = off — Halved (was 0.60) after observing
                              #   spurious "iphone 11 64gb#3" fragmentation:
                              #   in brand-homogeneous partitions (Apple), λ=0.60
                              #   was creating density valleys WITHIN the same
                              #   product (€95-105 vs €105-120) — the feature
                              #   should separate DIFFERENT products, not price
                              #   tiers of identical ones.
# CV-gated activation: apply the price feature ONLY when the partition's
# price coefficient-of-variation falls in this window.
#   CV < CV_MIN: partition already coherent on price → augmenting is wasted
#     (and would risk over-splitting iPhone-Pro-only-style clusters by
#      small storage/condition price differences).
#   CV > CV_MAX: partition is hopelessly heterogeneous (e.g. all "games"
#     across all platforms and qualities). The price axis then dominates
#     the embedding and clusters items by price tier regardless of what
#     they actually are → "all cheap games €6-€37" giant blob, garbage
#     for anomaly detection even if statistically clean (high price-coh).
PRICE_FEATURE_CV_MIN = 0.50   # was 0.30: tightened so that brand-homogeneous
                              # partitions (which always have CV>0.30 because
                              # they mix iPhone/iPad/MacBook) DON'T trigger λ
                              # unless they're truly multi-product (CV>0.50).
PRICE_FEATURE_CV_MAX = 0.80
IMAGE_EMB_VERSION = "v1-clip-vit-l-14-multiphoto"


def _hf_cache_dir(model_name: str) -> str:
    """HF hub cache folder for a model name ('org/name' or a bare ST name)."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))


# If the SELECTED model is already cached, run the HF Hub fully offline. This
# skips the per-start "check for updates" HTTP call (the noisy "unauthenticated
# requests to the HF Hub" warning, also rate-limited) and speeds startup. Only
# when cached, so a fresh model can still download on first use. Honour any
# explicit user setting. Must run BEFORE importing sentence_transformers.
if "HF_HUB_OFFLINE" not in os.environ and os.path.isdir(_hf_cache_dir(MODEL_NAME)):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Embeddings are an optional dependency. Without them we fall back to the
# naive token-based key silently.
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None  # type: ignore

# HDBSCAN is also optional and ONLY used by the "Rigenera cluster" path.
# We do NOT import it at module load: importing hdbscan pulls in joblib's
# loky backend which spawns helper processes / semaphores — pointless cost
# (and a source of leaked-semaphore warnings on hard kills) for the in-cycle
# monitor, which only ever uses the incremental cluster_titles_batch.
# HDBSCAN_AVAILABLE is probed cheaply via importlib without importing it.
import importlib.util as _ilu
HDBSCAN_AVAILABLE = _ilu.find_spec("hdbscan") is not None
_hdbscan_lib = None  # lazily populated by _get_hdbscan()


def _get_hdbscan():
    """Import hdbscan on first actual use (regenerate path only)."""
    global _hdbscan_lib
    if _hdbscan_lib is None:
        import hdbscan as _h
        _hdbscan_lib = _h
    return _hdbscan_lib

# Translation is optional and only used to make cluster display labels uniform
# in English. The clustering itself happens in vector space via the multilingual
# sentence-transformer, so translation is purely cosmetic. If transformers isn't
# available or the model can't load, labels stay in their original language.
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    AutoTokenizer = None          # type: ignore
    AutoModelForSeq2SeqLM = None  # type: ignore


# ─── Naive fallback ─────────────────────────────────────────────────────────

_NOISE = {
    # Italian — condition/sale-noise (verified against the live corpus 2026-05-28)
    "nuovo", "nuova", "usato", "usata", "perfetto", "perfetta", "perfette",
    "ottimo", "ottime", "ottima", "buono", "buona", "buone", "buonissime",
    "originale", "vintage", "raro", "rara", "spedizione", "gratis", "prezzo",
    "trattabile", "con", "senza", "scatola", "scatoletta", "imballo",
    "accessori", "completo", "completa", "garanzia", "ricondizionato",
    "sigillato", "come", "per", "del", "della", "delle", "dei", "degli",
    "rotto", "rotta", "condizioni", "condizione", "stato", "veloce",
    "perditempo", "occasione", "vendo", "vende", "vendere", "venduto", "uso",
    "preciso", "descrizione",
    "modello", "tipo", "pari", "il", "la", "lo", "gli",
    # English
    "new", "used", "perfect", "original", "with", "without", "free", "fast",
    "for", "from", "shipping", "boxed", "sealed", "refurbished", "warranty",
    "condition", "conditions", "mint", "ships", "returns", "selling", "sell",
    "excellent", "complete",
    # French
    "neuf", "neuve", "comme", "etat", "état", "parfait", "parfaite", "très",
    "bon", "bonne", "avec", "sans", "jamais", "utilisé", "utilisée", "vendu",
    "vente", "vendre", "expédition", "expedition", "rapide", "occasion",
    "pour", "complet", "complète", "fonctionne", "lire",
    "vends", "vend", "cause", "prix", "en", "la", "le", "les", "des", "du",
    "sur", "il", "au", "aux", "tout", "toute", "toutes", "tous",
    # German
    "neu", "neuwertig", "gebraucht", "wie", "gut", "guter", "zustand",
    "ohne", "mit", "versand", "schnell", "garantie", "sehr",
    "ovp", "für", "von", "die", "das", "der", "ein", "eine", "auf", "im", "am",
    "verkaufe", "verkaufen", "biete", "preis",
    # Spanish
    "nueva", "nuevo", "nuevos", "nuevas", "como", "perfecto", "perfecta",
    "perfectos", "perfectas", "estado", "envío", "envio", "rapido", "rápido",
    "jamás", "jamas", "muy", "usado", "usada", "usados", "usadas",
    "para", "sin", "el", "los", "las", "todo", "toda", "todos", "todas",
    # Dutch
    "nieuw", "ongebruikt", "zo", "goed", "als", "verzending", "snel", "staat",
    "compleet",
    "met", "voor", "van", "te", "koop",
    # Portuguese
    "novo", "nova", "novos", "novas", "perfeito", "perfeita", "perfeitos",
    "perfeitas", "envio",
    # Generic / connectors
    "and", "et", "und", "lot", "set", "kit", "the",

    # ─── COLORI (tutte le lingue) ──────────────────────────────────────
    # Italian
    "nero", "nera", "neri", "nere", "bianco", "bianca", "bianchi", "bianche",
    "rosso", "rossa", "rossi", "rosse", "blu", "azzurro", "azzurra", "verde",
    "verdi", "giallo", "gialla", "gialli", "gialle", "grigio", "grigia",
    "grigi", "grigie", "marrone", "marroni", "viola", "rosa", "arancione",
    "arancio", "oro", "argento", "dorato", "dorata", "argentato",
    # French
    "noir", "noire", "noirs", "noires", "blanc", "blanche", "blancs",
    "blanches", "rouge", "rouges", "bleu", "bleue", "bleus", "bleues", "vert",
    "verte", "verts", "vertes", "jaune", "jaunes", "gris", "grise", "marron",
    "violet", "violette", "rose", "roses", "orange", "doré", "dorée", "argenté",
    "argentée",
    # English
    "black", "white", "red", "blue", "green", "yellow", "grey", "gray",
    "brown", "purple", "pink", "orange", "gold", "silver", "golden",
    # German
    "schwarz", "weiß", "weiss", "rot", "blau", "grün", "gelb", "grau",
    "braun", "lila", "violett", "orangefarben", "gold", "silber", "golden",
    # Spanish
    "negro", "negra", "negros", "negras", "blanco", "blanca", "blancos",
    "blancas", "rojo", "roja", "rojos", "rojas", "azul", "azules", "verde",
    "verdes", "amarillo", "amarilla", "gris", "marrón", "morado", "rosa",
    "naranja", "dorado", "dorada", "plata", "plateado", "plateada",
    # Portuguese
    "preto", "preta", "pretos", "pretas", "branco", "branca", "brancos",
    "brancas", "vermelho", "vermelha", "vermelhos", "vermelhas", "azul",
    "azuis", "verde", "verdes", "amarelo", "amarela", "cinza", "cinzento",
    "castanho", "rosa", "laranja", "dourado", "dourada", "prata", "prateado",
    # Dutch
    "zwart", "wit", "rood", "blauw", "groen", "geel", "grijs", "bruin",
    "paars", "roze", "oranje", "goud", "zilver",

    # ─── AGGETTIVI generici (non strettamente prodotto) ────────────────
    # condizione / funzionalità
    "funzionante", "funzionale", "funcional", "funcionando", "functioning",
    "working", "operational", "fonctionnel", "fonctionnelle",
    # qualità/stato
    "bello", "bella", "belli", "belle", "beau", "belle", "beautiful",
    "lovely", "fantastico", "fantastica", "fantástico", "great", "amazing",
    "wonderful", "schön", "schöne", "schoen", "bonito", "bonita", "linda",
    # caratteristiche tecniche generiche (NON modelli)
    "digital", "digitale", "digitali", "digitaal", "digitales", "digitalt",
    "electrique", "électrique", "electrico", "eléctrico", "elétrico",
    "elétrica", "elektrische", "electric", "elettrico", "elettrica",
    "externo", "externa", "externe", "external", "esterno", "esterna",
    "interno", "interna", "interne", "internal",
    "originel", "originelle", "original", "originali", "originale", "originaal",
    # Connettori extra multilingua
    "of", "by", "to", "into", "onto", "over", "under",
    "da", "di", "del", "dello", "delle", "dei", "degli", "alla", "alle",
    "ai", "agli", "al", "allo", "nel", "nella", "nelle", "nei", "negli",
    "sui", "sugli", "sulla", "sulle", "sul", "sullo",
    "à", "aux", "du", "des", "des", "par", "vers", "chez",
    "vom", "zur", "zum", "zu",
    "ao", "à", "às", "do", "dos", "das", "no", "na", "nos", "nas",
    # frasi vendita comuni
    "vends", "vendido", "vendida", "vendidos", "vendidas",
    "occasion", "occasions", "ocasión", "ocasion",

    # ─── INTRUDERS trovati nei cluster IDEAL (analisi 2026-05-29) ──────
    # preposizioni e connettori che ancora sfuggono
    "de",        # FR/IT/ES/PT — 162 cluster IDEAL "contaminati"
    "in",        # EN/IT — 66 cluster
    "non",       # FR/IT negation
    "sous",      # FR "sotto"
    # edizioni / versioni / generazioni
    "edition", "édition", "ediz", "edizione", "edizioni",
    "edición", "edição", "ediciones", "edições", "ausgabe", "editie",
    "version", "versione", "versioni", "versión", "versions", "versão",
    "génération", "generation", "generations", "generazione",
    "generazioni", "generación", "geração", "gen",
    # packaging
    "boîte", "boite", "boîtier", "scatola", "scatole", "caja", "cajas",
    "doos", "caixa", "blister", "ovp", "boxed",
    "officiel", "officielle", "officiale", "oficial",
    # generici tecnici (non specifici di un modello)
    "ordinateur", "computer", "windows", "android", "ios",
    "filaire", "fil", "wired", "kabel", "cabo", "cable", "câble",
    "cavo", "cavi",
    "numérique", "numerique",
    "retro", "retrò",
    "auto", "automatic", "automatique", "automatico", "automático",
    "manuale", "manual", "manuel",

    # ─── SPRINT 2 NOISE Tier A — data-driven (freq ≥ 100 on 192k corpus) ─
    # Marketing/qualifiers boilerplate
    "smart",  # smart watch/tv/lock already canonicalized — token is filler
    "collection", "collector", "limited", "limitée", "limitada",
    "limitato", "limitata", "rare", "compatible", "compatibile", "premium",
    "active", "design",
    # Adjectives (IT/FR/ES boilerplate)
    "bellissima", "bellissimo", "bellissimi", "bellissime",
    "professionale", "professionnel", "professional",
    "inteligente", "connecté", "connectée",
    "ancien", "ancienne",
    # Color descriptors leftover ("colore X" prefix filler)
    "color", "colore", "couleur",
    # Brand/meta words
    "marque", "brand", "marca",
    # Region/condition slang (collezionismo)
    "loose", "cib", "tbe",
    # Currency leftovers (end-of-title)
    "eur", "euros",
    # Testing/working state
    "testato", "tested", "testé",
    # Promo
    "promo", "promotion", "promozione", "anniversary",
    # Generic qualifiers
    "universal", "universel", "universale",
    # Demographic
    "kids", "enfant", "enfants", "bambini", "bambino",
    # Description meta
    "description",
    # "included" (already inkl. in DE list, but FR/EN/ES/IT missing)
    "inclus", "incluse", "incluido", "included", "inkl",
    # Parts/pieces ambiguous
    "pezzi", "pièces", "piezas", "pieces", "stuks",
    # Misc adjectives FR/DE
    "vide", "fixe", "neue", "neuer", "neues",
    # Sealed packaging (already partial)
    "precintado", "scellé", "scellée",
    # Plurals of neuf/nuovo (singular already in list)
    "neufs", "neuves", "nuove", "nuovi",
    # Model/modello as meta-words (the actual model number is the signal)
    "model", "modèle", "modelo",
    # Rétro adjective (accented)
    "rétro",
    # Special/anniversary editions
    "special",
    # Bag/case generics (low signal — already have "case" from _TRANSLATE)
    "tapis", "bag", "sacoche", "sac", "tasche", "borsa", "pochette",
    # Residual conjunctions
    "ou", "or", "pas", "ed", "ne",

    # ─── SPRINT 5 NOISE: condition/sale phrases (user-reported, regen Z) ──
    # FR: "très bon état", "en parfait état", "à vendre"
    "tres", "bons", "bonnes",          # bon/bonne already in NOISE
    "états", "etats",                  # état already in NOISE
    "vendre",                          # à vendre
    "venta", "venda",                  # ES/PT "for sale"
    # IT plurals + state words
    "ottimi", "buoni",                 # ottimo/buono singolare already in NOISE
    "conservato", "conservata", "conservati", "conservate",
    "tenuto", "tenuta", "tenuti", "tenute",
    "intatto", "intatta", "intatti", "intatte",
    "integro", "integra", "integri", "integre",
    # FR état descriptors
    "entretenu", "entretenue",         # "very well maintained"
    # Complete (multi-lingue, plural)
    "komplett", "completos", "completas",
    # Functioning (residual)
    "funcionando", "funcional",

    # ─── SPRINT 7 NOISE (architect intervento #1) ─────────────────────
    # Region/format codes — never reliable disambiguators on Vinted
    "pal", "ntsc", "ntsc-j", "ntscj", "ntsc-u", "pal-eu",
    # Generic FR/IT residual
    "ordinateur", "argent", "argenté", "argentee", "argentée",

    # ─── SPRINT 8 NOISE — CPU/tech generics (user-reported 2026-05-31) ─
    # "Intel Core i5-6600K 3.5GHz LGA1151" — most tokens are filler that
    # contaminates the embedding. Keep brand (intel) + model code (i5 6600k).
    "ghz", "mhz", "gh", "mh",       # clock frequency unit (residual after strip)
    "cpu", "core", "cores",         # too generic for Intel/AMD listings
    "processore", "processeur", "processor", "procesador",
    "lga", "socket", "chipset",
    "ddr", "ddr3", "ddr4", "ddr5",  # RAM type — secondary to capacity
    "rpm",                          # HDD spindle speed
    "watt", "watts", "w",
    "volt", "volts", "v",
    # Generic adjectives for tech listings
    "vintage", "retro-style",
}
_NOISE = {w.lower() for w in _NOISE}

# ─── MULTILINGUAL TOKEN TRANSLATION ─────────────────────────────────────────
# Maps language-specific product-type tokens to a canonical English token so
# clusters of the same product across listings in different languages
# converge (e.g. "jeu nintendo switch" + "giochi nintendo switch" → both end
# up with "game nintendo switch"). Data-driven: built from the live corpus
# 174k titles (2026-05-28), top 15 product families covering ~55k tokens.
# All entries are SAFE one-way mappings: a single canonical per family,
# never ambiguous (e.g. "portable" excluded — FR uses it for both laptop
# and phone; "fil" excluded — generic "wire"; "radio"/"console"/"drone"
# already cross-lingual identical, no mapping needed).
_TRANSLATE = {
    # GAME — videogames
    "jeu":"game","jeux":"game","gioco":"game","giochi":"game","games":"game",
    "spiel":"game","spiele":"game","juego":"game","juegos":"game",
    # CONTROLLER
    "manette":"controller","manettes":"controller","controllers":"controller",
    "gamepad":"controller","gamepads":"controller","mando":"controller",
    "mandos":"controller",
    # CAMERA — "appareil" is borderline (FR "appareil photo") but in the
    # corpus almost always means camera; safe to map.
    "kamera":"camera","fotocamera":"camera","cámara":"camera","camara":"camera",
    "caméra":"camera","appareil":"camera","appareils":"camera",
    # HEADPHONES
    "casque":"headphones","casques":"headphones","cuffie":"headphones",
    "écouteurs":"headphones","ecouteurs":"headphones","kopfhörer":"headphones",
    "kopfhoerer":"headphones","auriculares":"headphones","auricolari":"headphones",
    "cascos":"headphones","headphone":"headphones",
    # PHONE
    "téléphone":"phone","telephone":"phone","telefono":"phone","telefon":"phone",
    "teléfono":"phone","telefoon":"phone","handy":"phone","handys":"phone",
    # CASE / COVER
    "coque":"case","coques":"case","cover":"case","custodia":"case",
    "custodie":"case","hülle":"case","huelle":"case","hülen":"case",
    "funda":"case","fundas":"case","hoesje":"case","hoesjes":"case",
    "capa":"case","capas":"case","étui":"case","etui":"case",
    # CHARGER
    "chargeur":"charger","chargeurs":"charger","caricabatterie":"charger",
    "caricatore":"charger","caricatori":"charger","ladegerät":"charger",
    "ladekabel":"charger","cargador":"charger","cargadores":"charger",
    "oplader":"charger",
    # LAPTOP — "portable" excluded (FR ambiguous: téléphone portable vs
    # ordinateur portable). The unambiguous ones:
    "portatile":"laptop","portatil":"laptop","portátil":"laptop",
    "laptops":"laptop",
    # TABLET
    "tablette":"tablet","tableta":"tablet","tablets":"tablet","tabletas":"tablet",
    # MOUSE — "maus" DE means mouse, "ratón"/"raton" ES means mouse
    "souris":"mouse","maus":"mouse","ratón":"mouse","raton":"mouse",
    "mouses":"mouse",
    # KEYBOARD
    "clavier":"keyboard","claviers":"keyboard","tastiera":"keyboard",
    "tastiere":"keyboard","tastatur":"keyboard","tastaturen":"keyboard",
    "teclado":"keyboard","teclados":"keyboard","toetsenbord":"keyboard",
    "keyboards":"keyboard",
    # SPEAKER
    "enceinte":"speaker","enceintes":"speaker","altavoz":"speaker",
    "altavoces":"speaker","lautsprecher":"speaker","altoparlante":"speaker",
    "altoparlanti":"speaker","speakers":"speaker",
    # SCREEN / DISPLAY
    "écran":"screen","schermo":"screen","pantalla":"screen","scherm":"screen",
    "bildschirm":"screen",
    # WATCH (also smartwatch — "watch" already in English, "smart watch"
    # mapped via "watch" token)
    "montre":"watch","montres":"watch","orologio":"watch","reloj":"watch",
    "uhr":"watch",
    # PRINTER
    "imprimante":"printer","stampante":"printer","impresora":"printer",
    "drucker":"printer",
    # ─── TYPO FIXES (sprint-1) ────────────────────────────────────────
    "nitendo":"nintendo",   # 136 occurrences in 184k titles

    # ─── SPRINT 2: 12 NEW FAMILIES (data-driven, freq ≥ 200) ──────────
    # BATTERY (1754 occ): unifies FR/IT/DE/ES/PT
    "batterie":"battery","batteria":"battery","bateria":"battery",
    "akku":"battery","batería":"battery",
    # CABLE (1692 occ): one of the biggest unified families
    "cavo":"cable","cavi":"cable","câble":"cable","câbles":"cable",
    "cabo":"cable","kabel":"cable",
    # CARTRIDGE (1419 occ): toner/inkjet/game cartridges
    "cartuccia":"cartridge","cartucce":"cartridge","cartouche":"cartridge",
    "cartouches":"cartridge",
    # ADAPTER (869 occ)
    "adaptateur":"adapter","adattatore":"adapter","adaptador":"adapter",
    # REMOTE (528 occ)
    "télécommande":"remote","telecomando":"remote","fernbedienung":"remote",
    # DISK (954 occ): CD/DVD/HDD ambiguity acceptable (all storage)
    "disque":"disk","disco":"disk",
    # PROJECTOR (490 occ)
    "proiettore":"projector","projecteur":"projector","proyector":"projector",
    "beamer":"projector",
    # READER (931 occ): lettore CD/DVD/MP3/ebook
    "lettore":"reader","lecteur":"reader","lector":"reader","leitor":"reader",
    # LAMP (469 occ)
    "lampe":"lamp","lampada":"lamp","lámpara":"lamp",
    # RAZOR (322 occ)
    "rasoir":"razor","rasoio":"razor",
    # FAN (ventilatore, 247 occ)
    "ventilatore":"fan","ventilateur":"fan","ventilador":"fan",
    # DRYER (456 occ): asciugacapelli (excluded "sèche" — too ambiguous)
    "asciugacapelli":"dryer","secador":"dryer","föhn":"dryer",
    # STRAIGHTENER (1500 occ): piastra/lisseur/glätteisen/plancha
    "piastra":"straightener","lisseur":"straightener",
    "glätteisen":"straightener","plancha":"straightener",
    # CURLER (466 occ): arricciacapelli/boucleur/lockenstab
    "arricciacapelli":"curler","boucleur":"curler","lockenstab":"curler",
    # BRUSH (1097 occ): brosse/spazzola/cepillo (dental/hair)
    "brosse":"brush","spazzola":"brush","cepillo":"brush",
    # CAR (~850 occ, excludes "auto" — ambiguous with "automatico")
    "voiture":"car","coche":"car","macchina":"car","carro":"car",
}

# ─── BRAND ALIASES — sub-brands only ────────────────────────────────────────
# Conservative: only documented sub-brands (Xiaomi family). Honor used to be
# Huawei's sub-brand but became independent in 2020 → NOT aliased. Beats is
# Apple-owned but kept its own branding → NOT aliased.
_BRAND_ALIASES = {
    "redmi":  "Xiaomi",
    "poco":   "Xiaomi",
    "roidmi": "Xiaomi",   # Xiaomi-ecosystem brand
}


def _normalize_brand(brand: str | None) -> str | None:
    """Map a sub-brand to its parent. Used by both `_embed_text` and partition."""
    if not brand:
        return brand
    key = brand.strip().lower()
    return _BRAND_ALIASES.get(key, brand)

_STRIP_PATTERNS = [
    re.compile(r"\b(?:nuovo|nuova)\s+con\s+(?:cartellino|etichetta).*$", re.I),
    # Multi-language "like new" qualifier — but only when at end of title,
    # otherwise we'd strip useful specs ("Pc Gamer comme neuf - RTX i5 16go SSD").
    re.compile(r"\b(?:wie|comme|like|come)\s+(?:neu|neuf|nieuw|nuov[oa]|new)\s*$",
               re.I),
    # End-of-title currency suffix: "iPhone 12 250€" → strip. Multi-symbol.
    re.compile(r"\b\d+\s*(?:eur|euros?|euro|€|\$|£|usd|gbp|chf)\s*$", re.I),
    # CPU clock-frequency: "3.5 GHz", "3.30GHz", "2.40 GHz" → entire token
    # stripped. The decimal number after subword splitting ("3 30 ghz") was
    # producing fake model-number tokens (30, 40, 60) that contaminated CPU
    # cluster embeddings (Intel i5-6600 / i5-6500 / i7-4770 all mashed into
    # one "intel core ghz" cluster). Match BEFORE the general subword split.
    re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:ghz|mhz|gh|mh)\b", re.I),
    # Generic punctuation strip — keep last so decimal points get to the
    # clock-frequency rule first.
    re.compile(r"[^\w\s+]+"),
]


# Split only when the digit run is ≥2 AND the letter prefix is ≥2 chars.
# The 2-char prefix requirement preserves SINGLE-LETTER model codes:
#   Samsung Galaxy S23 → keeps "s23" (was wrongly split to "s 23" → all
#   Galaxy S-series mashed together with the orphan "s" token).
#   Sony A7, Canon M50, Olympus E-M5, iPhone X → all preserved.
# Multi-char prefixes still split: iphone15→iphone 15, ddr4→stays
# (digit_run<2), 256gb→256 gb, 1080p→1080 p.
_SUBWORD_LETTER_DIGIT = re.compile(r"([a-zàâéèêëïîôûùçñü]{2,})(\d{2,})")
_SUBWORD_DIGIT_LETTER = re.compile(r"(\d{2,})([a-zàâéèêëïîôûùçñü])")
# Memory/storage normalization. GB/GO (FR Octets)/TB/TO (FR Téraoctets)/MB/MO
# all converge to canonical "gb"/"tb"/"mb". 1To FR was the main miss before.
_MEMORY_UNITS = re.compile(r"\b(\d+)\s*(gb|go|tb|to|mb|mo)\b")


def _normalize(s: str) -> str:
    s = (s or "").lower()
    for p in _STRIP_PATTERNS:
        s = p.sub(" ", s)
    # Subword split: "iphone15" → "iphone 15", "256gb" → "256 gb", "8gbram"
    # → "8 gbram" (then the GBRAM gets cleaned in further passes / NOISE).
    # Helps when sellers don't put spaces around variant numbers.
    s = _SUBWORD_LETTER_DIGIT.sub(r"\1 \2", s)
    s = _SUBWORD_DIGIT_LETTER.sub(r"\1 \2", s)
    # Canonicalize memory/storage tokens: "128GO" (FR) / "128gb" (EN/IT) /
    # "128 GB" (with space) all converge to "128 gb" so iPhone variants
    # cluster correctly by capacity. GO is the dominant FR abbreviation; we
    # always emit "gb" as the canonical form.
    def _mem(m):
        n, u = m.group(1), m.group(2)
        if u in ("gb", "go"):
            canon = "gb"
        elif u in ("tb", "to"):
            canon = "tb"
        elif u in ("mb", "mo"):
            canon = "mb"
        else:
            canon = u
        return f"{n} {canon}"
    s = _MEMORY_UNITS.sub(_mem, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _embed_text(item: dict, exclude_brand: str | None = None) -> str:
    """Text fed to the sentence embedder: brand + title, with multilingual
    condition/sale-noise tokens stripped (see `_NOISE`).

    A bare 1-word title like "Piastra" is ambiguous and collapses near other
    generic short titles (embedding hubness); prefixing the brand ("Rowenta
    Piastra") anchors it to its product family. We do NOT prepend the Vinted
    category here — testing showed it over-dominates and merges distinct
    products within a category (iPhone 13 + 15). Category is used instead as a
    hard PARTITION (see cluster_partitioned), which keeps cross-category items
    apart while letting the title still separate products within a category.

    NOISE stripping (2026-05-27): "iphone 11 ottime condizioni spedizione gratis"
    → "iphone 11". Sellers add boilerplate condition/shipping/sale phrases in
    many languages; those tokens are mostly identical across listings and just
    inject noise into the embedding (everyone says "neuf"/"como nuevo"/"like
    new"). Filtering them out before encoding makes vectors of the same product
    converge — title-only "iphone 11" listings now cluster cleanly with the
    "iphone 11 perfect condition shipping fast" variants.

    Brand-token kept verbatim (e.g. "apple iphone 11") — the brand IS signal,
    EXCEPT when `exclude_brand` is set. Inside a brand-homogeneous partition
    every item shares the same brand token, so it contributes the same noise
    to every embedding and hides the discriminating tokens (CPU vs SSD vs GPU
    for "Intel"). Passing `exclude_brand="Intel"` strips "intel" from the text
    so the residual tokens drive the clustering (Proposal #1, 2026-05-27).

    If the cleaned title is empty (very rare, ultra-short noisy titles), we
    fall back to the un-cleaned title so we never produce an empty vector.

    For e5 models the input must be prefixed with "query: " (they're trained
    that way); we add it here so every embedding site stays consistent."""
    title = item.get("title") or ""
    norm  = _normalize(title)
    # Brand canonical form (alias sub-brands → parent: Redmi/Poco → Xiaomi).
    raw_brand = (item.get("brand") or "").strip().lower()
    # Sprint-2: skip junk-brand prepending. "Senza marca" / "Altro" / "No brand"
    # are placeholder values from the seller form, not real brands — prepending
    # them dilutes the embedding with garbage tokens that appear in 100+ items.
    _JUNK_BRANDS_LOW = {"senza marca", "senzamarca", "altro", "altri",
                        "sans marque", "ohne marke", "sin marca", "sem marca",
                        "no brand", "none", "geen merk", "marca propia"}
    if raw_brand in _JUNK_BRANDS_LOW:
        raw_brand = ""
    raw_brand_canon = (_normalize_brand(raw_brand) or "").lower()
    brand = raw_brand_canon if raw_brand_canon else raw_brand
    # Word-boundary check (not substring): "Apple" should still prepend when
    # title="appletv 4k" (apple is in 'appletv' substring but not as a token).
    if brand and not re.search(rf"\b{re.escape(brand)}\b", norm):
        norm = _normalize(f"{brand} {title}")
    # Strip noise tokens. Brand-tokens are preserved (signal). NOTE: _normalize
    # does NOT length-filter tokens — single-char tokens survive unless they're
    # in _NOISE. Kept deliberately (single-letter model codes: Sony A, Canon M).
    brand_tokens = set(brand.split()) if brand else set()
    # Intra-partition: also strip the partition's brand token(s).
    skip_tokens: set[str] = set()
    if exclude_brand:
        canon_excl = _normalize_brand(exclude_brand) or exclude_brand
        skip_tokens = {t for t in _normalize(canon_excl).split() if t}
    out_tokens: list[str] = []
    for t in norm.split():
        if t in skip_tokens:
            continue
        if t in _NOISE and t not in brand_tokens:
            continue
        # Multilingual canonicalization: jeu/giochi/games → game, etc.
        out_tokens.append(_TRANSLATE.get(t, t))
    # Also alias sub-brand tokens that appear in the title (e.g. "Redmi Note"
    # → "xiaomi note") so cluster keys with translated brand merge correctly.
    out_tokens = [
        _BRAND_ALIASES.get(t, t).lower() if t in _BRAND_ALIASES else t
        for t in out_tokens
    ]
    cleaned = " ".join(out_tokens).strip()
    # If cleanup ate everything (title was pure noise like "Vendo ottime
    # condizioni nuovo"), return empty string — callers (embed_items /
    # regen) detect this and SKIP the item (it goes to 'unknown' cluster).
    # Previously we fell back to the un-cleaned title, but that re-introduced
    # noise into clustering and inflated bogus clusters of "noise-only" titles.
    if not cleaned:
        return ""
    return f"query: {cleaned}" if _NEEDS_E5_PREFIX else cleaned


def _naive_cluster_key(title: str, brand: str | None) -> str:
    norm = _normalize(title)
    b    = (brand or "").lower().strip()
    if not norm:
        return "unknown"

    if b and b in norm:
        tail   = norm.split(b, 1)[1].strip()
        tokens = [t for t in tail.split() if len(t) >= 2 and t not in _NOISE]
        prefix = b
    else:
        tokens = [t for t in norm.split() if len(t) >= 2 and t not in _NOISE]
        prefix = ""

    model_tokens = tokens[:3]
    if not model_tokens:
        return (prefix or norm[:40]).strip()
    key = f"{prefix} {' '.join(model_tokens)}".strip()
    return key[:60]


# ─── Embedding clustering (incremental, centroid-persistent) ────────────────

# Multilingual MiniLM — 384-dim embeddings, ~120MB download, fast on CPU.
# MODEL_NAME is defined near the top (env-configurable). Default is
# intfloat/multilingual-e5-large; the lighter paraphrase-multilingual-MiniLM is
# the alternative via VINTED_EMBED_MODEL. Both cover IT/FR/DE/EN/ES.

# Cosine threshold to merge a new item into an existing centroid. Calibrated
# empirically on a 2832-item Tecnologia DB via threshold sweep (0.70-0.95):
#   0.70  → 172 utili ma "bluetooth speaker" cluster contamina con karaoke/autoradio
#   0.78  → buoni cluster, qualche misto (smart lock in cluster bluetooth)
#   0.82  → SWEET SPOT: "monitor lg" / "manette wii" / "écouteurs pro" puliti,
#           solo "bluetooth audio" rimane un po' aggregato ma è categoria coerente
#   0.85  → cluster molto puliti ma "monitor" si frammenta in singleton
#   0.90+ → quasi tutto singleton, embedding inutile
SIMILARITY_THRESHOLD = 0.82

# ALWAYS-ON incremental path fixes. #1: true running-mean centroid update
# instead of the EMA(α=0.5) `(old+emb)/2` that overweights the most recent
# item and lets centroids drift between regenerations. #7: vectorized centroid
# match (one matmul) instead of a Python double-loop over all centroids per
# item. VALIDATED 2026-06-29 (scripts/test_incremental.py, 60k items, 7.4k
# clusters): assignment accuracy 49.6% (ema) → 62.7% (runmean), centroid
# drift-cos 0.975 → 0.997 → enabled by default.
INCREMENTAL_RUNNING_MEAN = True
INCREMENTAL_VECTOR_MATCH = True

# Band-aware incremental match: price-band siblings of the same product
# ("<base> [€a-b]", "<base> #highN"…) have near-identical TEXT centroids, so a
# cosine-only matcher routes new items among them by coin flip — a game priced
# €20 landing in the console band looks like a −85% deal (false positive on
# live inflow). When the text-matched key has band siblings and the item has a
# price, re-route to the sibling with the closest log-price median (persisted
# per centroid in cluster_centroids.price_median). No-op without siblings or
# medians (legacy DBs) — safe to keep always on.
BAND_AWARE_MATCH = True
_PRICE_BAND_TAG = re.compile(r" \[€[^\]]*\]| #high\d+| #low\d+| \[\?\]")


def _band_base_index(medians: dict) -> dict:
    """{base_key: [(cluster_key, price_median), …]} for bases with ≥2 bands.
    `medians` maps cluster_key → price_median (may contain None values)."""
    from collections import defaultdict
    idx = defaultdict(list)
    for k, med in medians.items():
        idx[_PRICE_BAND_TAG.sub("", k)].append((k, med))
    return {b: sibs for b, sibs in idx.items() if len(sibs) >= 2}


def _band_route(best_key: str, price, base_index: dict) -> str:
    """Re-route a text-matched key among its price-band siblings by log-price
    adjacency. Returns `best_key` unchanged when the item has no price, the key
    has no siblings, or no sibling has a stored median."""
    if not BAND_AWARE_MATCH or not price or price <= 0:
        return best_key
    sibs = base_index.get(_PRICE_BAND_TAG.sub("", best_key))
    if not sibs:
        return best_key
    import math
    lp = math.log(price)
    best, bd = best_key, None
    for k, med in sibs:
        if med and med > 0:
            d = abs(lp - math.log(med))
            if bd is None or d < bd:
                bd, best = d, k
    return best

# ─── Cluster refinement (regenerate path) ───────────────────────────────────
# After the density pass we run two refinements, validated offline against the
# real 40k-item shared DB (scripts/cluster_experiment.py + cluster_eval.py):
#   1. Lexical-gated singleton rescue — absorb a leftover singleton into the
#      nearest cluster on embedding sim alone if very close (HI), or at a looser
#      sim (LO) ONLY when brand (and any shared model number) also agree. This
#      recovers true duplicates HDBSCAN left as noise WITHOUT the brand-garbage
#      that pure-embedding absorption introduces (brand-purity 81%→85%).
#   2. Price-band split — break price-incoherent clusters (a €80 console + its
#      €15 games) into price bands so each has a meaningful median.
# Net effect vs raw HDBSCAN: coverage 61→68%, price-coherent 47→65%,
# brand-purity 83→85%, with product cohesion essentially unchanged.
# Rescue gates are MODEL-SPECIFIC: e5 cosine sims sit on a much higher scale
# than MiniLM (singleton→centroid median ~0.91 vs ~0.75), so it needs higher
# thresholds. Calibrated via scripts/e5_calibrate.py on the real 56k DB: at
# HI=0.94 e5 matches MiniLM's brand-purity (94%) while lifting coverage 68→84%
# and price-coherence 64→81%.
if _NEEDS_E5_PREFIX:
    RESCUE_GATE_HI = 0.94
    RESCUE_GATE_LO = 0.88
else:
    RESCUE_GATE_HI = 0.86
    RESCUE_GATE_LO = 0.72
# Price-split is deliberately CONSERVATIVE so it never masks a deal. It splits a
# cluster into two price tiers ONLY when ALL of these hold:
#   - the cluster is price-incoherent (CV > PRICE_SPLIT_CV), and
#   - there's a real multiplicative GAP (≥ e^GAP_LOG ≈ 2x) between two sorted
#     prices, with BOTH sides substantial (≥ MIN_BAND items) — so a single cheap
#     outlier (= the deal) is never shaved off into its own band, and
#   - the two sides ALSO differ in EMBEDDING (cosine < MAX_COS) — i.e. they're
#     different products. Same product with price variance (new vs used) →
#     near-identical titles → high cosine → NOT split, so an underpriced item
#     still stands out against its product's median.
# Calibration (scripts/calibrate_split.py, real DB) showed price+embedding can't
# reliably tell "different product" from "a deal" inside a cluster (cosine of
# diff-product vs same-type tiers overlap: 0.972 vs 0.977; best F1 only 0.38).
# So price-splitting is kept ULTRA-conservative — it fires only on the most
# extreme, confident cases (~10 clusters out of 14k) and never risks masking a
# deal. Detecting underpriced items is done instead by a statistical anomaly
# flag, not by splitting. See the calibration for these operating points.
PRICE_SPLIT_CV = 0.9
PRICE_SPLIT_MIN = 8           # min cluster size to even consider
PRICE_SPLIT_GAP_LOG = 0.9     # ln(ratio); 0.9 ≈ a 2.5x price chasm
PRICE_SPLIT_MIN_BAND = 3      # each tier must have ≥3 items (never shave a deal)
PRICE_SPLIT_MAX_COS = 0.97    # split only when the two tiers are clearly different

# ─── Image two-stage refine (regenerate path) ───────────────────────────────
# Validated (scripts/refine_sweep.py, random_tech_test.py): TEXT clusters first,
# then IMAGE (L-14, multi-photo mean) splits ONLY the already price-incoherent
# clusters (CV>gate) into visual sub-products (console vs games). Gated → never
# touches clean clusters (clean_split=0, deal-safe). price-coherence ↑, coverage
# preserved. The heavy CLIP lives ONLY in the regenerate subprocess.
IMAGE_MODEL_NAME      = os.environ.get("VINTED_IMAGE_MODEL", "clip-ViT-L-14")
IMAGE_REFINE_MIN      = 10    # sprint-7: was 6. Under 10 items, image
                              # sub-clustering produces more noise (CLIP-only
                              # closeness from same seller / similar pose)
                              # than real product structure.
IMAGE_REFINE_MIN_SUB  = 3     # sprint-7: was 2. Sub-clusters of size 2 are
                              # statistically unstable — 2066 cluster #imgN
                              # with jaccard=0 on titles found by scout.
# NEW (2026-05-26 v2): image refine no longer gated on a raw price-CV. It now
# fires on clusters where Jenks2 ATTEMPTED a split but didn't go through
# (because below its full threshold, or blocked by title-emb-cos like the
# Nintendo Switch case). The criterion is "Jenks2-style bimodal signal
# present but sub-threshold":
IMAGE_REFINE_NEAR_MISS_GVF   = 0.40   # sprint-7 (architect #2): re-enable
IMAGE_REFINE_NEAR_MISS_RATIO = 1.5    # near-miss gate (was 0 = disabled).
                                       # image_refine no longer fires on every
# After HDBSCAN finds visual sub-clusters, validate by checking that the
# image-centroids of the sub-clusters are actually different. Mirrors the
# title-emb cosine guard in Jenks2 but in the visual axis.
#
# 1.0 = guard DISABLED (we trust HDBSCAN density alone — splits only happen
# when real visual sub-structure is found, so the guard was double-counting).
# 0.92 = previous tight guard (rejected too many mappazzoni residui — see
# scripts/mappazzoni_diag.py output 2026-05-27).
IMAGE_REFINE_MAX_IMG_COS     = 0.92   # sprint-7 (architect #8): tightened
                                       # from 0.96 → 0.92. With min_sub=3,
                                       # near-miss gate active, and jaccard
                                       # gate, this finally rejects sub-clusters
                                       # that differ on noise (lighting/pose)
                                       # rather than product.
# Price-validation guard (alternative/complementary to img-cos): accept the
# visual split ONLY when the sub-clusters have substantially DIFFERENT
# median prices. Same-product variants (e.g. iPhone 12 black vs red) look
# different to CLIP but cost the same → REJECT. Different products (iPhone
# 12 vs case for iPhone 12) → big price gap → ACCEPT. Pure-visual splits
# (no price info) fall back to image_refine without this guard.
# Value 1.0 = DISABLED, 2.0 = sub-clusters must differ ≥2× on median.
IMAGE_REFINE_MIN_PRICE_RATIO = 1.3   # sprint-1 (was 1.0): reject visual splits
                                      # where sub-clusters have nearly equal median
                                      # prices (likely same product, color variants).
                                      # Old A/B at 2.0 blocked too many; 1.3 strikes
                                      # the balance between over-split and under-split.
IMAGE_MAX_PHOTOS      = 4     # photos per item to average

_KNOWN_BRANDS = {
    "apple", "samsung", "sony", "nintendo", "playstation", "xbox", "microsoft",
    "hp", "dell", "lenovo", "asus", "acer", "canon", "nikon", "jbl", "bose",
    "philips", "logitech", "huawei", "xiaomi", "redmi", "lg", "panasonic",
    "gopro", "dyson", "remington", "rowenta", "bosch", "makita", "dewalt",
    "nvidia", "amd", "intel", "razer", "sandisk", "kingston", "seagate",
    "toshiba", "garmin", "fitbit", "amazon", "google", "oneplus", "oppo",
    "realme", "motorola", "nokia", "tcl", "hisense", "marshall", "beats",
    "anker", "epson", "brother", "kodak", "polaroid", "fujifilm", "babyliss",
    "wahl", "cecotec", "iphone", "ipad", "macbook", "airpods",
}
_BRAND_ALIAS = {
    "iphone": "apple", "ipad": "apple", "macbook": "apple", "airpods": "apple",
    "imac": "apple", "redmi": "xiaomi", "ps2": "playstation",
    "ps3": "playstation", "ps4": "playstation", "ps5": "playstation",
    "psp": "playstation", "ps1": "playstation",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _lex_sig(title: str, brand: str) -> tuple[str, frozenset]:
    """(canonical brand, set of model-number tokens) used to gate rescue."""
    t = _strip_accents((title or "").lower())
    t = re.sub(r"[^a-z0-9+]", " ", t)
    tokens = t.split()
    b = re.sub(r"[^a-z0-9]", "", _strip_accents((brand or "").lower()))
    if b in ("senzamarca", "nomark", "none", "sansmarque", "altro", ""):
        b = ""
    if not b:
        b = next((tok for tok in tokens if tok in _KNOWN_BRANDS), "")
    b = _BRAND_ALIAS.get(b, b)
    nums = frozenset(
        tok for tok in tokens
        if re.fullmatch(r"[a-z]*\d[a-z0-9]*", tok) and len(tok) <= 7
        and tok not in _NOISE
    )
    return b, nums


_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load the sentence transformer (once per process)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


# ─── Translation pipeline (multi-language → English) ────────────────────────
#
# Used only to generate English display_label for newly-created clusters
# (cosmetic). transformers 5.x removed the "translation" task name from the
# pipeline factory, so we use AutoTokenizer + AutoModelForSeq2SeqLM directly
# and call .generate() ourselves.
#
# Model is ~300MB; downloaded on first use. If it fails to load (no network,
# disk full, etc.), translation is permanently disabled for the process.

TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-mul-en"
_tokenizer = None
_seq2seq_model = None
_translator_lock = threading.Lock()
_translator_failed = False


def _get_translator():
    """Return (tokenizer, model) or None if unavailable/failed."""
    global _tokenizer, _seq2seq_model, _translator_failed
    if not TRANSLATION_AVAILABLE or _translator_failed:
        return None
    if _tokenizer is None:
        with _translator_lock:
            if _tokenizer is None and not _translator_failed:
                try:
                    _tokenizer     = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
                    _seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL)
                except Exception as e:
                    print(f"  ⚠ translation model unavailable: {e}", flush=True)
                    _translator_failed = True
                    return None
    return _tokenizer, _seq2seq_model


def translate_to_english(text: str) -> str | None:
    """Translate `text` to English. Returns None on failure / unavailability."""
    if not text or not text.strip():
        return None
    tr = _get_translator()
    if tr is None:
        return None
    tokenizer, model = tr
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=80)
        outputs = model.generate(**inputs, max_length=80, num_beams=1)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if out and out.strip():
            return out.strip().lower()
    except Exception:
        pass
    return None


def _emb_to_blob(emb) -> bytes:
    return emb.astype("float32").tobytes()


def _blob_to_emb(blob: bytes):
    return np.frombuffer(blob, dtype="float32")


def _cosine(a, b) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_FAMILY_TOKENS_FOR_KEY = {
    "tablet", "laptop", "phone", "watch", "camera", "headphones",
    "controller", "game", "cable", "charger", "keyboard", "mouse",
    "case", "battery", "speaker", "screen", "printer", "reader",
    "remote", "adapter", "cartridge", "disk", "projector", "lamp",
    "razor", "fan", "dryer", "straightener", "curler", "brush",
    "console",
}


def _make_embedding_cluster_key(title: str) -> str:
    """Generate a human-readable cluster_key for a newly-discovered cluster.
    Top 4 informative tokens of the normalized title, capped at 60 chars.

    Sprint-7 (architect #1): demote family tokens (tablet/phone/console/…) to
    suffix position so the cluster name shows the SPECIFIC product first.
    Was producing labels like "tablet redmi pad se" → now "redmi pad se tablet".
    Helps both readability and downstream sibling-merge (which strips suffixes)."""
    norm = _normalize(title)
    # Apply _TRANSLATE to canonicalize family tokens (chargeur→charger, jeu→
    # game, piastra→straightener) so the demote logic sees them correctly.
    tokens = [_TRANSLATE.get(t, t) for t in norm.split()
              if len(t) >= 2 and t not in _NOISE]
    if not tokens:
        return (norm[:60] or "unknown")
    specific = [t for t in tokens if t not in _FAMILY_TOKENS_FOR_KEY]
    family   = [t for t in tokens if t in _FAMILY_TOKENS_FOR_KEY]
    # Sprint-7 fix: specific[:5] (was :3) — preserves "max", "256", "gb",
    # "ultra", "plus", etc. so iPhone 15 Pro Max ≠ iPhone 15 Pro in the key.
    # Without this, sibling-merge fused 15 Pro + 15 Pro Max (same base name).
    if specific:
        ordered = specific[:5] + family[:1]
    else:
        ordered = tokens[:5]
    return " ".join(ordered)[:80]   # also raised cap from 60 → 80 chars


def _display_label(cluster_key: str, category: str | None) -> str | None:
    """Cluster display name: "english_category · cluster_key" when the Vinted
    category is known (free & accurate, from the breadcrumb slug), else None
    so the GUI falls back to the raw cluster_key. Replaces the unreliable
    opus-mt translation that mangled short product terms (piastra→'window')."""
    cat = (category or "").strip()
    if cat:
        return f"{cat} · {cluster_key}"
    return None


def cluster_titles_batch(items: list[dict], conn=None) -> dict[int, str]:
    """
    Batch-cluster a list of items, returning {item_id: cluster_key}.

    Resolution order per item:
      1. Embedding clustering (requires sentence-transformers + conn):
           - Encode the title once (batched for the whole call).
           - Compare to all existing centroids in the DB (cosine).
           - If max-sim ≥ SIMILARITY_THRESHOLD → assign that cluster_key,
             update centroid (moving average) and item_count.
           - Else → create a new centroid + cluster_key from the title,
             with an English display_label (best-effort translation).
      2. Naive fallback (brand + 2-3 informative tokens) — used when
         embeddings are unavailable or no DB connection is provided.

    The centroid table is created by category_db.connect()'s schema.
    """
    result: dict[int, str] = {}
    needs_embed: list[dict] = []

    for it in items:
        if EMBEDDINGS_AVAILABLE and conn is not None:
            needs_embed.append(it)
        else:
            result[it["id"]] = _naive_cluster_key(it.get("title", ""),
                                                  it.get("brand"))

    if not needs_embed:
        return result

    # Load existing centroids (one query, then keep in memory), plus item_count
    # (running-mean weighting, #1) and price_median (band-aware routing).
    # Legacy DBs may miss price_median → fall back to a NULL-median select.
    centroids: dict[str, "np.ndarray"] = {}
    counts: dict[str, int] = {}
    medians: dict[str, float | None] = {}
    try:
        rows_c = list(conn.execute(
            "SELECT cluster_key, embedding, item_count, price_median "
            "FROM cluster_centroids"))
    except Exception:
        rows_c = [(r[0], r[1], r[2], None) for r in conn.execute(
            "SELECT cluster_key, embedding, item_count FROM cluster_centroids")]
    for row in rows_c:
        try:
            centroids[row[0]] = _blob_to_emb(row[1])
            counts[row[0]] = int(row[2] or 1)
            medians[row[0]] = row[3]
        except Exception:
            continue
    band_index = _band_base_index(medians) if BAND_AWARE_MATCH else {}

    # Encode all titles in one batch (much faster than one-by-one).
    # Embed brand+title (see _embed_text) for better discrimination.
    model = _get_model()
    titles_norm = [_embed_text(it) for it in needs_embed]
    embeddings = model.encode(
        titles_norm, show_progress_bar=False, convert_to_numpy=True,
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    upd_existing: list[tuple] = []
    ins_new:      list[tuple] = []

    # #7: vectorize the centroid match (one matmul vs a Python double-loop over
    # ~40k centroids per item). Built once and grown as new clusters appear.
    _ckeys = list(centroids.keys())
    if INCREMENTAL_VECTOR_MATCH and _ckeys:
        _cmat = np.array([centroids[k] for k in _ckeys], dtype=np.float32)
        _cmat /= np.maximum(np.linalg.norm(_cmat, axis=1, keepdims=True), 1e-12)
    else:
        _cmat = None
    # Clusters created during THIS batch (the Python double-loop saw these via
    # the growing dict; the vectorized path tracks them in a small side list so
    # an item can still join a near-duplicate created a few items earlier).
    _new_keys: list[str] = []
    _new_vecs: list = []

    for it, emb in zip(needs_embed, embeddings):
        best_key: str | None = None
        best_sim = -1.0
        if INCREMENTAL_VECTOR_MATCH:
            en = emb / max(float(np.linalg.norm(emb)), 1e-12)
            if _cmat is not None:
                sims = _cmat @ en
                j = int(sims.argmax())
                best_sim = float(sims[j]); best_key = _ckeys[j]
            for nk, nv in zip(_new_keys, _new_vecs):
                s = float(nv @ en)
                if s > best_sim:
                    best_sim = s; best_key = nk
        else:
            for ck, c_emb in centroids.items():
                sim = _cosine(emb, c_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_key = ck

        if best_key is not None and best_sim >= SIMILARITY_THRESHOLD:
            # Band-aware re-route among price-band siblings (no-op otherwise).
            routed = _band_route(best_key, it.get("price"), band_index)
            if routed != best_key and routed in centroids:
                best_key = routed
            # Merge into existing cluster — centroid update.
            old = centroids[best_key]
            if INCREMENTAL_RUNNING_MEAN:
                # True running mean: new = (old*n + emb)/(n+1). Converges to the
                # cluster mean instead of the EMA(α=0.5) that the moving average
                # below implements (which overweights the most recent item).
                n = counts.get(best_key, 1)
                new_centroid = (old * n + emb) / (n + 1)
                counts[best_key] = n + 1
            else:
                new_centroid = (old + emb) / 2.0
            centroids[best_key] = new_centroid
            result[it["id"]] = best_key
            upd_existing.append((_emb_to_blob(new_centroid), now, best_key))
        else:
            # New cluster — generate a unique human-readable key
            base = _make_embedding_cluster_key(it.get("title", ""))
            new_key = base
            n = 0
            while new_key in centroids:
                n += 1
                new_key = f"{base}#{n}"
            centroids[new_key] = emb
            if INCREMENTAL_RUNNING_MEAN:
                counts[new_key] = 1
            if INCREMENTAL_VECTOR_MATCH:
                _new_keys.append(new_key)
                _new_vecs.append(emb / max(float(np.linalg.norm(emb)), 1e-12))
            result[it["id"]] = new_key
            # Display label = "english_category · cluster_key" when the item's
            # Vinted category is known (extracted during sold-check), else NULL
            # so the GUI falls back to the raw cluster_key.
            display = _display_label(new_key, it.get("category"))
            price_med = float(it["price"]) if (it.get("price") and it["price"] > 0) else None
            ins_new.append((new_key, _emb_to_blob(emb), now, display, price_med))

    # Persist centroid updates / inserts in batch. We catch and re-raise
    # with extra context because executemany's default error message swallows
    # the params, making it impossible to tell whether the problem is a
    # locked DB, a stale schema, or a malformed param tuple.
    if upd_existing:
        try:
            conn.executemany(
                """
                UPDATE cluster_centroids
                   SET embedding       = ?,
                       item_count      = item_count + 1,
                       last_updated_at = ?
                 WHERE cluster_key     = ?
                """,
                upd_existing,
            )
        except Exception as e:
            print(f"  ⚠ cluster centroid UPDATE failed "
                  f"({len(upd_existing)} rows): {type(e).__name__}: {e}",
                  flush=True)
            raise
    if ins_new:
        try:
            conn.executemany(
                """
                INSERT INTO cluster_centroids
                    (cluster_key, embedding, item_count, last_updated_at,
                     display_label, price_median)
                VALUES (?, ?, 1, ?, ?, ?)
                """,
                ins_new,
            )
        except Exception as e:
            print(f"  ⚠ cluster centroid INSERT failed "
                  f"({len(ins_new)} rows): {type(e).__name__}: {e}",
                  flush=True)
            raise

    return result


def cluster_hdbscan(items: list[dict], conn,
                    min_cluster_size: int = 2,
                    log_fn=print,
                    ) -> dict[int, str]:
    """
    Density-based clustering with HDBSCAN — the algorithm the regenerate
    button SHOULD use when "more data should produce finer splits".

    Workflow:
      1. Embed every title with the multilingual MiniLM (one batch call).
      2. L2-normalize the vectors so euclidean distance ≡ cosine distance.
      3. Run HDBSCAN with `min_cluster_size=N` — clusters with fewer items
         become "noise" (label -1) and get the naive_cluster_key fallback.
      4. For each density cluster, pick the MEDOID (item closest to the
         cluster mean) and use its title to mint a human-readable
         cluster_key. Centroid = mean of member vectors.
      5. Wipe and re-insert `cluster_centroids` so the in-cycle incremental
         path can later merge new items into these clusters.

    Pros vs the incremental cluster_titles_batch:
      - Density-aware → with 200 iPhones it can split 13/Pro/Pro Max into
        separate clusters; with 5 it leaves them lumped.
      - Outlier detection (noise label) → low-quality titles don't pollute
        centroids by accident.

    Cons:
      - Cluster_keys are NOT stable between runs (medoid depends on the
        current dataset).
      - Slower than the incremental pass (HDBSCAN on 10k items ≈ 20-40s
        of pure CPU, plus the embedding pass).
    """
    if not HDBSCAN_AVAILABLE:
        log_fn("  ⚠ HDBSCAN not installed — falling back to incremental")
        return cluster_titles_batch(items, conn=conn)
    mapping, rows = _hdbscan_compute(items, "", min_cluster_size, log_fn)
    _persist_centroids(conn, rows, wipe=True)
    log_fn(f"  persisted {len(rows)} centroids")
    return mapping


def _hdbscan_compute(items: list[dict], namespace: str,
                     min_cluster_size: int, log_fn, emb_cache=None,
                     n_jobs: int = -1,
                     ) -> tuple[dict[int, str], list[tuple]]:
    """Core HDBSCAN clustering of ONE group of items — NO DB writes.

    Returns (mapping, centroid_rows). cluster_keys are prefixed with
    `namespace` (e.g. "hair styling tools/") so partitioned callers keep
    categories disjoint. centroid_rows are ready for INSERT.

    `emb_cache` ({id: raw_embedding}) lets the caller embed every item once and
    reuse it here + in refine_clusters (avoids embedding the corpus twice).
    """
    if not items:
        return {}, []

    def _naive_all() -> dict[int, str]:
        return {it["id"]: namespace + _naive_cluster_key(it.get("title", ""),
                                                         it.get("brand"))
                for it in items}

    # Embeddings unavailable, or too few points for HDBSCAN (needs
    # >min_samples) → naive fallback, no centroids.
    if not EMBEDDINGS_AVAILABLE or len(items) <= min_cluster_size:
        return _naive_all(), []

    if emb_cache is not None:
        embs = np.array([emb_cache[it["id"]] for it in items])
    else:
        model = _get_model()
        embs = model.encode([_embed_text(it) for it in items],
                            show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.maximum(norms, 1e-12)

    # PRICE FEATURE AUGMENTATION (only for HDBSCAN clustering; centroids
    # persisted later are still the pure-text mean_vec for back-compat).
    # Items with prices closer to their cluster mediana stick together; items
    # with extreme prices drift, which helps HDBSCAN naturally separate
    # accessories/bundles/edizioni from the main product.
    embs_for_cluster = embs_n
    if PRICE_FEATURE_LAMBDA > 0 and len(items) >= 4:
        prices = np.array([float(it.get("price") or 0) for it in items], dtype=np.float32)
        valid = prices > 0
        if valid.sum() >= 4:
            # CV GATE: only apply when the partition's price distribution is
            # in a "useful" range. CV<MIN: cluster already homogeneous, λ
            # would over-split for tiny price diffs. CV>MAX: partition too
            # heterogeneous, λ creates "all cheap items" garbage blobs.
            valid_prices = prices[valid]
            cv_part = (valid_prices.std() / valid_prices.mean()
                       if valid_prices.mean() > 0 else 0)
            if PRICE_FEATURE_CV_MIN <= cv_part <= PRICE_FEATURE_CV_MAX:
                log_p = np.zeros_like(prices)
                log_p[valid] = np.log1p(prices[valid])
                mu = log_p[valid].mean(); sd = log_p[valid].std()
                if sd > 0:
                    z = np.zeros_like(prices)
                    z[valid] = (log_p[valid] - mu) / sd
                    # No-price items stay at z=0 (centro) so they don't drift.
                    price_dim = (z * PRICE_FEATURE_LAMBDA).reshape(-1, 1).astype(np.float32)
                    aug = np.concatenate([embs_n, price_dim], axis=1)
                    # Re-normalize L2 so cosine semantics remain consistent.
                    an = np.linalg.norm(aug, axis=1, keepdims=True)
                    embs_for_cluster = aug / np.maximum(an, 1e-12)

    # Sprint-3 adaptive params: min_samples scales with partition size for
    # more conservative density estimation on large partitions (reduces noise
    # label rate and produces purer cluster cores). Algorithm boruvka_balltree
    # + approx_min_span_tree shaves 30-50% wallclock on partitions >500 items
    # without measurable quality loss (BallTree beats KDTree above ~60 dims).
    n_part = len(embs_for_cluster)
    if n_part < 20:
        min_samples = 1
    elif n_part < 200:
        min_samples = 2
    else:
        min_samples = 3   # large/mega partitions: density estimate stable enough
    clusterer = _get_hdbscan().HDBSCAN(
        min_cluster_size      = min_cluster_size,
        min_samples           = min_samples,
        metric                = "euclidean",
        algorithm             = "boruvka_balltree",
        approx_min_span_tree  = True,
        cluster_selection_method = HDBSCAN_SELECTION,
        core_dist_n_jobs      = n_jobs,
    )
    labels = clusterer.fit_predict(embs_for_cluster)

    result: dict[int, str] = {}
    by_label: dict[int, dict] = {}
    noise_items:    list[dict] = []
    noise_embs_idx: list[int]  = []
    for idx, (it, label, emb) in enumerate(zip(items, labels, embs_n)):
        if label == -1:
            noise_items.append(it)
            noise_embs_idx.append(idx)
            continue
        b = by_label.setdefault(
            int(label), {"ids": [], "embs": [], "titles": [], "cats": [],
                         "prices": []})
        b["ids"].append(it["id"])
        b["embs"].append(emb)
        b["titles"].append(it.get("title", ""))
        b["cats"].append((it.get("category") or "").strip())
        if it.get("price") and it["price"] > 0:
            b["prices"].append(float(it["price"]))

    from collections import Counter as _Counter
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    used_keys: set[str] = set()
    centroid_rows: list[tuple] = []
    centroid_vecs: list[tuple[str, "np.ndarray"]] = []

    for _label, data in by_label.items():
        embs_arr  = np.array(data["embs"])
        mean_vec  = embs_arr.mean(axis=0)
        n_centroid = mean_vec / max(float(np.linalg.norm(mean_vec)), 1e-12)
        dists     = np.linalg.norm(embs_arr - mean_vec, axis=1)
        medoid_t  = data["titles"][int(np.argmin(dists))]
        readable  = _make_embedding_cluster_key(medoid_t)   # no namespace
        base      = namespace + readable
        key, n    = base, 0
        while key in used_keys:
            n  += 1
            key = f"{base}#{n}"
        used_keys.add(key)
        for item_id in data["ids"]:
            result[item_id] = key
        cat_counts = _Counter(c for c in data["cats"] if c)
        # Only adopt a category label when it actually REPRESENTS the cluster:
        # at least 30% of the items carry it. Otherwise a single mis-categorized
        # listing (e.g. a seller who puts an iPhone in "mobile phone cases")
        # would mislabel an entire 72-item iPhone cluster. Below quorum → no
        # category in the display label (falls back to the product key alone).
        top_cat = None
        if cat_counts:
            best_cat, best_n = cat_counts.most_common(1)[0]
            if best_n / max(1, len(data["ids"])) >= 0.30:
                top_cat = best_cat
        # Display label uses the readable (un-namespaced) product name so we
        # don't repeat the category (e.g. "games · jeu ps4", not
        # "games · games/jeu ps4").
        pr = sorted(data["prices"])
        price_med = pr[len(pr) // 2] if pr else None
        centroid_rows.append((
            key, _emb_to_blob(mean_vec.astype("float32")),
            len(data["ids"]), now_ts, _display_label(readable, top_cat),
            price_med,
        ))
        centroid_vecs.append((key, n_centroid))

    # Hybrid rescue of noise items (within this group only).
    n_rescued = 0
    if centroid_vecs and noise_items:
        for it, emb_idx in zip(noise_items, noise_embs_idx):
            emb = embs_n[emb_idx]
            best_key, best_sim = None, -1.0
            for key, c_vec in centroid_vecs:
                sim = float(np.dot(emb, c_vec))
                if sim > best_sim:
                    best_sim, best_key = sim, key
            if best_key is not None and best_sim >= SIMILARITY_THRESHOLD:
                result[it["id"]] = best_key
                n_rescued += 1
            else:
                result[it["id"]] = namespace + _naive_cluster_key(
                    it.get("title", ""), it.get("brand"))
    else:
        for it in noise_items:
            result[it["id"]] = namespace + _naive_cluster_key(
                it.get("title", ""), it.get("brand"))

    # Recount item_count after rescue so on-disk counts are accurate.
    if n_rescued:
        counts = _Counter(result.values())
        centroid_rows = [
            (k, blob, counts.get(k, dc), ts, lab, med)
            for k, blob, dc, ts, lab, med in centroid_rows
        ]
    return result, centroid_rows


def _persist_centroids(conn, centroid_rows: list[tuple], wipe: bool) -> None:
    """Write centroid rows to the DB. `wipe` clears the table first (use once
    per full rebuild; pass False to append additional partitions)."""
    if conn is None:
        return
    if wipe:
        conn.execute("DELETE FROM cluster_centroids")
    if centroid_rows:
        conn.executemany(
            """
            INSERT INTO cluster_centroids
                (cluster_key, embedding, item_count, last_updated_at,
                 display_label, price_median)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            centroid_rows,
        )


def embed_items(items: list[dict], log_fn=lambda *a: None,
                chunk_size: int = 4000, conn=None) -> dict:
    """Embed every item ONCE → {id: raw_embedding}.

    READS FROM DB CACHE FIRST. Items already have `text_emb`/`text_emb_version`
    pre-computed by the continuous text_worker → loaded as-is. Only items with
    a missing OR stale (different version) cache get re-encoded here. So the
    regen embedding step shrinks from ~15-20 min to seconds once the worker
    has caught up.

    If `conn` is provided, fresh embeddings are written back to the DB
    (text_emb + text_emb_version) so subsequent regens find them cached.
    Without this, every regen re-encodes the same items from scratch.

    Chunked + progress on the items we actually need to re-encode."""
    if not EMBEDDINGS_AVAILABLE or not items:
        return {}
    import time
    out: dict = {}
    todo: list[dict] = []
    expected_v_bytes = TEXT_EMB_VERSION.encode("utf-8")
    for it in items:
        blob = it.get("text_emb")
        ver = it.get("text_emb_version")
        # The caller may cast text_emb_version AS BLOB to defend against
        # corrupt rows → we accept both str and bytes here.
        if isinstance(ver, bytes):
            ver_ok = (ver == expected_v_bytes)
        else:
            ver_ok = (ver == TEXT_EMB_VERSION)
        if blob is not None and ver_ok:
            try:
                out[it["id"]] = _blob_to_emb(blob)
                continue
            except Exception:
                pass
        todo.append(it)
    cached = len(items) - len(todo)
    if cached:
        log_fn(f"  embedding: {cached}/{len(items)} dal cache DB (text_emb_version={TEXT_EMB_VERSION})")
    if not todo:
        return out
    model = _get_model()
    texts_raw = [_embed_text(it) for it in todo]
    # Filter out items whose title was 100% noise → empty after preprocessing.
    # Those items are skipped (no embedding, no cluster) → they fall to the
    # 'unknown' bucket in cluster_partitioned / refine. Prevents an artificial
    # cluster of empty-pre-processed listings ("query:" alone) that would
    # otherwise pollute centroids.
    keep_mask = [t != ("query: " if _NEEDS_E5_PREFIX else "") and t != "" for t in texts_raw]
    skipped_empty = sum(1 for k in keep_mask if not k)
    if skipped_empty:
        log_fn(f"  embedding: {skipped_empty} item con titolo 100% noise scartati "
               f"(non avranno embedding → cluster_key='unknown')")
    todo  = [it for it, k in zip(todo, keep_mask) if k]
    texts = [t  for t,  k in zip(texts_raw, keep_mask) if k]
    n = len(texts)
    t0 = time.time()
    for i in range(0, n, chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        chunk_items = todo[i:i + chunk_size]
        embs = model.encode(chunk_texts, show_progress_bar=False,
                            convert_to_numpy=True, batch_size=256)
        for it, e in zip(chunk_items, embs):
            out[it["id"]] = e
        # Persist fresh embeddings so next regen finds them in cache. Done in
        # chunks (committed each batch) so even an aborted regen leaves some
        # progress for the worker / next run.
        if conn is not None:
            try:
                conn.executemany(
                    "UPDATE items SET text_emb = ?, text_emb_version = ?, "
                    "text_tried_at = datetime('now') WHERE id = ?",
                    [(_emb_to_blob(e.astype("float32")), TEXT_EMB_VERSION, it["id"])
                     for it, e in zip(chunk_items, embs)],
                )
                conn.commit()
            except Exception as ex:
                log_fn(f"  ⚠ persist fresh text_emb failed: {type(ex).__name__}: {ex}")
        done = i + len(chunk_texts)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (n - done) / max(rate, 1.0) if done < n else 0
        log_fn(f"  embedding {done}/{n} fresh ({100*done/n:.0f}%, "
               f"{elapsed:.0f}s elapsed, ETA ~{eta:.0f}s, {rate:.0f} item/s)")
    return out


def eval_metrics(items: list[dict], mapping: dict[int, str],
                 emb_of: dict) -> dict:
    """Quality metrics for cluster evaluation.

    BASIC (vs scripts/eval_db.py, computed from already-embedded corpus):
       cohesion, price_coh, brand_pure, coverage, score=coh×price_coh

    EXTENDED (added to expose blob clusters that gameable the basic SCORE):
       n_ideal:    cluster size≥10, CV<0.30, brand_pure≥95% → anomaly-ready
       n_blob:     cluster size≥200, CV<0.50, brand_pure<70% → semantic garbage
                   (statistically clean but composed of unrelated products
                    just sharing a price tier — see "all cheap games" trap)
       n_giant:    cluster size>200 (any composition)
       anom_ready: % of items living in an IDEAL cluster (THE KPI for the
                   downstream deal detector — clusters not IDEAL provide a
                   noisy baseline and false positives)
       qscore:     cohesion × price_coh × (brand_pure/100) × (1 - blob_share)
                   composite penalizing blob clusters proportionally to the
                   share of items trapped in them.
    """
    import statistics as _st
    import math as _math
    from collections import defaultdict as _dd, Counter as _Counter
    # Junk-brand filter: lowercase tokens to ignore when computing brand-purity
    # (these are placeholder "no-brand" values selected by sellers, not real brands).
    _JUNK_BRANDS = {"", "altro", "altri", "senza marca", "senzamarca", "nomark",
                    "no brand", "none", "sans marque", "ohne marke", "sin marca",
                    "sem marca", "geen merk", "marca propia", "altri brand"}
    def _canon_brand(b: str | None) -> str:
        """Canonicalize brand for eval: lowercase, normalize via alias map,
        return '' for junk brands so they don't count toward purity denominator."""
        if not b:
            return ""
        bl = b.strip().lower()
        if bl in _JUNK_BRANDS:
            return ""
        nb = _normalize_brand(b)
        return (nb or b).strip().lower()
    n = len(items)
    by: dict[str, list[dict]] = _dd(list)
    for it in items:
        by[mapping.get(it["id"], "unknown")].append(it)
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    cov = sum(len(v) for v in multi.values())
    cn = cd = 0; pok = 0; bp_num = bp_den = 0
    # extended counters
    n_ideal = 0; ideal_items = 0
    n_blob = 0; blob_items = 0
    n_giant = 0
    for k, v in multi.items():
        ids = [it["id"] for it in v if it["id"] in emb_of]
        if not ids:
            continue
        vecs = np.array([emb_of[i] for i in ids])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-12)
        cen = vecs.mean(0); cen /= max(float(np.linalg.norm(cen)), 1e-12)
        cn += float((vecs @ cen).sum()); cd += len(ids)
        pr = [it["price"] for it in v if it.get("price")]
        # CV on log(price) — aligned with stage 2 price-feature space and the
        # lognormal nature of marketplace prices (a cluster spanning 200-800€
        # has CV~0.40 raw but is a clean unimodal lognormal population).
        cv_cluster = 0.0
        if len(pr) >= 2:
            log_pr = [_math.log(p) for p in pr if p > 0]
            if len(log_pr) >= 2:
                mu = _st.mean(log_pr)
                if mu != 0:
                    cv_cluster = _st.pstdev(log_pr) / abs(mu)
            # Backward-compat metric: keep raw price-coh gate at CV<=0.9
            if _st.mean(pr) > 0 and _st.pstdev(pr) / _st.mean(pr) <= 0.9:
                pok += len(v)
        # Brand-purity: canonicalize via _normalize_brand + filter junk brands
        brs = [_canon_brand(it.get("brand")) for it in v]
        brs = [b for b in brs if b]
        bp_cluster = 0.0
        if brs:
            dom_n = _Counter(brs).most_common(1)[0][1]
            bp_num += dom_n
            bp_den += len(brs)
            bp_cluster = dom_n / len(brs)
        else:
            # No real brands: treat as 'unknown' brand — neither pure nor impure.
            # If size >=2 and no brand info, we exclude from BP but still allow IDEAL
            # only if the size criterion is met (handled below).
            bp_cluster = 1.0  # don't penalize brand-less clusters
        # IDEAL: size≥10, log-CV<0.30, brand_pure≥95% — perfect anomaly baseline
        # (cv_cluster >= 0 also includes the constant-price case CV=0 which is
        # the highest possible price coherence)
        if len(v) >= 10 and cv_cluster >= 0 and cv_cluster < 0.30 and bp_cluster >= 0.95:
            n_ideal += 1
            ideal_items += len(v)
        # BLOB: large but statistically clean while semantically heterogeneous.
        # NOTE: align with GIANT (size>=200, not size>200) — bug fix.
        if len(v) >= 200 and cv_cluster >= 0 and cv_cluster < 0.50 and bp_cluster < 0.70:
            n_blob += 1
            blob_items += len(v)
        if len(v) >= 200:
            n_giant += 1
    sizes = [len(v) for v in by.values()]
    sing = sum(1 for s in sizes if s == 1)
    cohesion = cn / max(1, cd)
    price_coh = 100 * pok / max(1, n)
    brand_pure = 100 * bp_num / max(1, bp_den)
    score = cohesion * price_coh
    blob_share = blob_items / max(1, n)
    qscore = score * (brand_pure / 100.0) * (1.0 - blob_share)
    anom_ready = 100 * ideal_items / max(1, n)
    return {
        "items":         n,
        "cluster":       len(sizes),
        "multi":         len(multi),
        "singleton_pct": 100 * sing / max(1, len(sizes)),
        "coverage":      100 * cov / max(1, n),
        "cohesion":      cohesion,
        "price_coh":     price_coh,
        "brand_pure":    brand_pure,
        "score":         score,
        # ── extended quality metrics ─────────────────────────────────
        "n_ideal":       n_ideal,
        "ideal_items":   ideal_items,
        "anom_ready":    anom_ready,
        "n_blob":        n_blob,
        "blob_items":    blob_items,
        "n_giant":       n_giant,
        "qscore":        qscore,
    }


# ── Partition sharding for HDBSCAN parallelism ──────────────────────────────
# The single largest partition (typically the brand/family-less __nobrand__
# residual) runs HDBSCAN alone in one worker → it's the serial wall-clock
# bottleneck (super-linear in N). When enabled, partitions bigger than
# PARTITION_SHARD_CAP are pre-split with a cheap MiniBatchKMeans into chunks of
# ~PARTITION_SHARD_TARGET items, which then run in PARALLEL across the pool.
# Items inside a KMeans chunk are embedding-neighbours, so a real product
# cluster mostly stays within one chunk (the cost is rare boundary fragments).
# Default ON since 2026-06-29: cuts the full-DB partition from ~38 min to
# ~7 min with no quality loss (A/B: sharded default-config OBJ 0.274 vs
# unsharded production 0.179 on the same corpus, scripts/confirm_full.py).
PARALLEL_SHARD_LARGE   = True
PARTITION_SHARD_CAP    = 6000
PARTITION_SHARD_TARGET = 4000
HDBSCAN_SELECTION      = "eom"   # "eom" | "leaf" (leaf → finer/smaller clusters)
PARTITION_FAMILY_FOR_BRANDED = False  # partition branded items by cat|brand|family
                                      # so device+accessory can't co-cluster at the
                                      # source (see loop in cluster_partitioned)
DISABLE_POOL           = False   # run heavy partitions in the main thread (no
                                 # ProcessPoolExecutor) — avoids spawn-reimport
                                 # deadlocks when called from scripts that import
                                 # heavy modules; fine with sharding (small parts)


def _shard_large_partitions(by_part: dict, emb_cache, log_fn) -> dict:
    if not PARALLEL_SHARD_LARGE or emb_cache is None:
        return by_part
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception:
        return by_part
    out: dict[str, list] = {}
    n_sharded = 0
    for part, pit in by_part.items():
        n = len(pit)
        if n <= PARTITION_SHARD_CAP:
            out[part] = pit; continue
        K = max(2, -(-n // PARTITION_SHARD_TARGET))   # ceil division
        X = np.array([emb_cache[it["id"]] for it in pit], dtype=np.float32)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        km = MiniBatchKMeans(n_clusters=K, n_init=3, batch_size=2048,
                             random_state=0).fit(X)
        shards: dict[int, list] = {}
        for it, lab in zip(pit, km.labels_):
            shards.setdefault(int(lab), []).append(it)
        for lab, grp in shards.items():
            out[f"{part}#k{lab}"] = grp
        n_sharded += 1
        log_fn(f"  shard: '{part}' ({n} items) → {len(shards)} sotto-shard "
               f"per parallelizzare HDBSCAN")
    return out


def cluster_partitioned(items: list[dict], conn,
                        min_cluster_size: int = 2,
                        log_fn=print, emb_cache=None) -> dict[int, str]:
    """Partition items hierarchically (category → brand), then HDBSCAN within
    each partition.

    Partition key: the Vinted category when known (from the sold-check HTML
    breadcrumb), else the brand, else a shared no-brand bucket. cluster_keys are
    namespaced "<partition>/<key>" so two partitions can never collide.

    Partitioning by brand matters for BOTH quality and speed: category is only
    ~13% populated, so without it ~87% of items land in one giant partition
    where HDBSCAN is slow (super-linear, single core) AND mixes brands. Brand
    splits it into many small partitions → much faster, and products stay within
    their brand. The title still distinguishes products WITHIN a partition.
    """
    if not HDBSCAN_AVAILABLE:
        log_fn("  ⚠ HDBSCAN not installed — falling back to incremental")
        return cluster_titles_batch(items, conn=conn)

    by_part: dict[str, list[dict]] = {}
    # Sprint-3: family hint shards the bloated __nobrand__ partition (46k
    # items in single thread = the wall-clock bottleneck). Items with the
    # same family token in title (already canonicalized by _TRANSLATE) go
    # to the same shard so cluster quality is preserved.
    _FAMILY_PRIORITY = (  # ordered: more specific first
        "case", "charger", "cable", "battery", "screen", "keyboard", "mouse",
        "headphones", "speaker", "remote", "adapter", "cartridge", "lamp",
        "razor", "fan", "dryer", "straightener", "curler", "brush",
        "controller", "console", "watch", "camera", "phone", "tablet",
        "laptop", "printer", "projector", "reader", "disk", "game",
    )
    _FAMILY_SET = set(_FAMILY_PRIORITY)
    # Sprint-7 #6: product-line lookahead — when known product lines appear in
    # title, they pin the family more reliably than generic tokens (e.g.
    # "redmi pad" → tablet, "redmi note" → phone, even though both contain "redmi").
    _PRODUCT_LINE_TO_FAMILY = [
        (("redmi", "pad"),    "tablet"),
        (("redmi", "note"),   "phone"),
        (("galaxy", "tab"),   "tablet"),
        (("galaxy", "s"),     "phone"),
        (("galaxy", "a"),     "phone"),
        (("galaxy", "buds"),  "headphones"),
        (("galaxy", "watch"), "watch"),
        (("ipad",),           "tablet"),
        (("iphone",),         "phone"),
        (("macbook",),        "laptop"),
        (("airpods",),        "headphones"),
        (("apple", "watch"),  "watch"),
        (("xperia",),         "phone"),
        (("pixel",),          "phone"),
    ]
    def _family_of(it: dict) -> str | None:
        toks = _normalize(it.get("title") or "").split()
        toks_set_raw = set(toks)
        # First: product-line lookahead (most specific)
        for line, fam in _PRODUCT_LINE_TO_FAMILY:
            if all(t in toks_set_raw for t in line):
                return fam
        # Fallback: family-token via _TRANSLATE
        toks_set = set(_TRANSLATE.get(t, t) for t in toks)
        for fam in _FAMILY_PRIORITY:
            if fam in toks_set:
                return fam
        return None

    for it in items:
        cat = (it.get("category") or "").strip()
        raw_brand, _ = _lex_sig(it.get("title", ""), it.get("brand"))
        # Canonicalize sub-brand → parent (Redmi/Poco → Xiaomi) so items from
        # different commercial labels of the same family share the partition.
        brand = (_normalize_brand(raw_brand) or raw_brand or "").lower()
        # category → brand: when the category is known (from the sold-check
        # breadcrumb), partition by category AND brand together so a "Consoles"
        # category can't merge a Sony with a Microsoft. Falls back to brand, then
        # a shared no-brand bucket. (A/B on the real DB: brand-purity 94→98%.)
        if cat and brand:
            part = f"{cat}|{brand}"
        elif cat:
            part = cat
        else:
            # Sprint-3: shard __nobrand__ by family hint when available.
            # This turns the 46k single-shard bottleneck into 10-15 shards
            # of 2-5k items each, all parallelizable in the heavy pool.
            fam = _family_of(it) if not brand else None
            if brand:
                part = brand
            elif fam:
                part = f"__nobrand__|{fam}"
            else:
                part = "__nobrand__"
        # Experimental (PARTITION_FAMILY_FOR_BRANDED): extend the key with the
        # family hint for BRANDED items too, so HDBSCAN can't merge a device
        # with its accessories at the source (nintendo|game vs nintendo).
        # Risk: the same product with/without the family word in its title
        # lands in different partitions (fragmentation) — judged by benchmark.
        if PARTITION_FAMILY_FOR_BRANDED and brand:
            fam_b = _family_of(it)
            if fam_b:
                part = f"{part}|{fam_b}"
        by_part.setdefault(part, []).append(it)

    # Optional: pre-shard oversized partitions so the biggest one parallelizes
    # across the pool instead of running alone (serial bottleneck).
    by_part = _shard_large_partitions(by_part, emb_cache, log_fn)

    n_part = len(by_part)
    log_fn(f"  partitioning {len(items)} items into {n_part} partitions (categoria→brand)…")
    import time as _t
    import os as _os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = _t.time()
    mapping: dict[int, str] = {}
    all_rows: list[tuple] = []
    used_keys: set[str] = set()      # global de-dup across partitions

    # Split partitions in two tracks:
    #   - HEAVY (size >= HEAVY_MIN): farm out to a process pool. Each worker
    #     gets ONLY the sub-cache for its partition (not the whole 645MB
    #     dict). Inside each worker HDBSCAN runs single-threaded
    #     (core_dist_n_jobs=1) to avoid oversubscription on the shared CPU.
    #   - LIGHT (size < HEAVY_MIN): processed in the main thread sequentially.
    #     They're each so quick (<<1s) that the pickling overhead would dwarf
    #     the work.
    HEAVY_MIN = 50
    parts_sorted = sorted(by_part.items(), key=lambda kv: -len(kv[1]))
    heavy = [(p, items) for p, items in parts_sorted if len(items) >= HEAVY_MIN]
    light = [(p, items) for p, items in parts_sorted if len(items) < HEAVY_MIN]
    n_workers = max(1, min(len(heavy), (_os.cpu_count() or 4) // 2))
    log_fn(f"  → {len(heavy)} heavy partition (size>={HEAVY_MIN}) in pool da {n_workers} worker, "
           f"{len(light)} light in main thread")

    def _process_results(part_label: str, part_items: list[dict],
                          m_local: dict, rows_local: list[tuple]):
        """De-dup keys across partitions and merge into the global state."""
        rename: dict[str, str] = {}
        fixed_rows = []
        for row in rows_local:
            k = row[0]
            nk, c = k, 0
            while nk in used_keys:
                c += 1
                nk = f"{k}~{c}"
            used_keys.add(nk)
            if nk != k:
                rename[k] = nk
            fixed_rows.append((nk,) + tuple(row[1:]))
        if rename:
            m_local = {iid: rename.get(k, k) for iid, k in m_local.items()}
        mapping.update(m_local)
        all_rows.extend(fixed_rows)

    # 1) HEAVY partitions in parallel
    n_done = 0
    if heavy and DISABLE_POOL:
        # Sequential main-thread path (no ProcessPoolExecutor).
        for part, part_items in heavy:
            ns = "" if part.startswith("__nobrand__") else f"{part}/"
            m_local, rows_local = _hdbscan_compute(
                part_items, ns, min_cluster_size, lambda *a, **k: None,
                emb_cache, n_jobs=1)
            _process_results(part, part_items, m_local, rows_local)
            n_done += 1
        heavy = []   # skip the pool block below
    if heavy:
        jobs = []   # (part, ns, part_items, sub_cache)
        for part, part_items in heavy:
            ns = "" if part.startswith("__nobrand__") else f"{part}/"
            # Only ship the embeddings the worker needs (cuts pickle size
            # from ~645MB to a few MB per partition).
            sub_cache = ({it["id"]: emb_cache[it["id"]] for it in part_items}
                         if emb_cache is not None else None)
            jobs.append((part, ns, part_items, sub_cache))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_meta = {
                ex.submit(_hdbscan_worker, ns, part_items, min_cluster_size, sub_cache):
                    (part, part_items)
                for part, ns, part_items, sub_cache in jobs
            }
            for fut in as_completed(future_to_meta):
                part, part_items = future_to_meta[fut]
                try:
                    m_local, rows_local = fut.result()
                except Exception as e:
                    log_fn(f"  ⚠ heavy partition '{part}' fallita: {e}; "
                           f"fallback in main thread")
                    ns = "" if part.startswith("__nobrand__") else f"{part}/"
                    m_local, rows_local = _hdbscan_compute(
                        part_items, ns, min_cluster_size, log_fn, emb_cache)
                _process_results(part, part_items, m_local, rows_local)
                n_done += 1
                if n_done % max(1, len(heavy) // 5) == 0 or n_done == len(heavy):
                    log_fn(f"  heavy {n_done}/{len(heavy)} done · "
                           f"{_t.time()-t0:.0f}s elapsed · {len(all_rows)} clusters so far")

    # 2) LIGHT partitions in the main thread (cheap, no pickling)
    for part, part_items in light:
        ns = "" if part.startswith("__nobrand__") else f"{part}/"
        m_local, rows_local = _hdbscan_compute(
            part_items, ns, min_cluster_size, log_fn, emb_cache, n_jobs=1)
        _process_results(part, part_items, m_local, rows_local)

    _persist_centroids(conn, all_rows, wipe=True)
    log_fn(f"  persisted {len(all_rows)} centroids across {n_part} partitions "
           f"({_t.time()-t0:.0f}s totale)")
    return mapping


def _hdbscan_worker(namespace: str, part_items: list[dict],
                    min_cluster_size: int, sub_cache):
    """Module-level worker for ProcessPoolExecutor. Picklable.
    Uses core_dist_n_jobs=1 to avoid oversubscription when multiple workers
    run in parallel (each on a separate CPU)."""
    return _hdbscan_compute(part_items, namespace, min_cluster_size,
                             log_fn=lambda *a, **kw: None,
                             emb_cache=sub_cache, n_jobs=1)


def _price_band_split(groups: dict, emb_of, log_fn) -> dict:
    """Split a price-incoherent cluster into two price tiers ONLY at a genuine
    chasm that ALSO separates two different products. `groups` is {key:[items]}
    with numeric 'price'; `emb_of` maps id→L2-normalized embedding.

    A two-tier split is accepted only when (see PRICE_SPLIT_* for the rationale):
      - cluster CV > PRICE_SPLIT_CV, and
      - the largest sorted-price gap with ≥MIN_BAND items on each side exceeds
        e^GAP_LOG (~2x), and
      - the two sides differ in embedding (cosine < MAX_COS) → different product.
    Same product with price variance (new/used) → near-identical titles → high
    cosine → NOT split, so an underpriced item still stands out as a deal. No
    quantile fallback (that would shave continuous distributions = mask deals)."""
    import math
    import statistics as _st
    out: dict[str, list] = {}
    n_split = 0
    for k, v in groups.items():
        pr = [it["price"] for it in v if it.get("price")]
        if (len(v) < PRICE_SPLIT_MIN or len(pr) < 2 * PRICE_SPLIT_MIN_BAND
                or not _st.mean(pr) or _st.pstdev(pr) / _st.mean(pr) <= PRICE_SPLIT_CV):
            out[k] = v
            continue
        priced = sorted((it for it in v if it.get("price")), key=lambda it: it["price"])
        noprice = [it for it in v if not it.get("price")]
        lp = [math.log(it["price"]) for it in priced]
        # largest multiplicative gap with both sides >= MIN_BAND items
        best = None
        for i in range(PRICE_SPLIT_MIN_BAND - 1, len(priced) - PRICE_SPLIT_MIN_BAND):
            g = lp[i + 1] - lp[i]
            if g > PRICE_SPLIT_GAP_LOG and (best is None or g > best[0]):
                best = (g, i)
        if best is None:
            out[k] = v
            continue
        i = best[1]
        A, B = priced[:i + 1], priced[i + 1:]
        # embedding-dissimilarity gate: same product (≈identical titles) → keep
        if emb_of is not None:
            ca = np.mean([emb_of[it["id"]] for it in A], axis=0)
            cb = np.mean([emb_of[it["id"]] for it in B], axis=0)
            ca = ca / max(float(np.linalg.norm(ca)), 1e-12)
            cb = cb / max(float(np.linalg.norm(cb)), 1e-12)
            if float(ca @ cb) > PRICE_SPLIT_MAX_COS:
                out[k] = v
                continue
        n_split += 1
        out[f"{k} [{int(A[0]['price'])}-{int(A[-1]['price'])}€]"] = A
        out[f"{k} [{int(B[0]['price'])}-{int(B[-1]['price'])}€]"] = B
        if noprice:
            out.setdefault(f"{k} [?]", []).extend(noprice)
    if n_split:
        log_fn(f"  price-split: {n_split} cluster a due fasce (gap + prodotto diverso)")
    return out


def _jenks2_stats(prices: list[float], min_band: int = 3):
    """Optimal 1-D 2-break on log(prices) → (split_idx, gvf, ratio) or None.
    Pure stats, no side effects — used both by `_price_split_jenks2` and by
    `image_refine` to decide whether a cluster has a bimodal price signal."""
    import math
    pr = [p for p in prices if p and p > 0]
    if len(pr) < 2 * min_band:
        return None
    lp = sorted(math.log(p) for p in pr)
    mu = sum(lp) / len(lp)
    tot = sum((x - mu) ** 2 for x in lp)
    if tot <= 0:
        return None
    best = None
    for i in range(min_band, len(lp) - min_band + 1):
        L = lp[:i]; R = lp[i:]
        mL = sum(L) / len(L); mR = sum(R) / len(R)
        w = sum((x - mL) ** 2 for x in L) + sum((x - mR) ** 2 for x in R)
        if best is None or w < best[0]:
            best = (w, i, mL, mR)
    w, i, mL, mR = best
    gvf = 1 - w / tot
    ratio = math.exp(mR - mL)
    return (i, gvf, ratio)


def _right_tail_split(groups: dict, log_fn, *,
                       min_size: int = 8,
                       min_cv: float = 0.5,
                       mad_k: float = 2.5,
                       multi_pop_madmed: float = 0.40,
                       extreme_range_ratio: float = 20.0,
                       extreme_mad_k: float = 1.5,
                       min_tail: int = 2,
                       min_main: int = 3,
                       max_iters: int = 3,
                       max_total_tail_frac: float = 0.40) -> dict:
    """MAD-based bi-directional tail split, ITERATIVE (max `max_iters` passes
    per direction).

    Each iteration on the current MAIN sub-cluster:
      - centro = mediana
      - dispersione = MAD (median absolute deviation)
      - soglia destra = median + mad_k × MAD  → tail RIGHT → `#highN`
      - soglia sinistra = median - mad_k × MAD → tail LEFT → `#lowN`
      - se nessun tail valido in entrambe le direzioni → STOP
      - altrimenti restringi main e ripeti

    Why BOTH sides matter (real example "nintendo/nintendo switch"):
      MAIN: console Switch  €90-€180 (mediana €130)
      #high: bundle premium / collector edition €350+
      #low: accessori "Accessori Nintendo Switch" / parti / cavi €10-€20
    Without #low, the cluster would still be "polluted" by accessories
    pretending to be the console.

    Robust to outliers (mediana/MAD vs mean/std), grades multiple populations
    in one cluster.
    """
    import statistics as _st
    out: dict[str, list] = {}
    n_clusters_split = 0
    n_total_iters    = 0
    n_total_moved    = 0
    n_high = n_low = 0
    n_extreme_range  = 0          # clusters that triggered the aggressive mode
    for k, v in groups.items():
        priced = [it for it in v if it.get("price") and it["price"] > 0]
        noprice = [it for it in v if not (it.get("price") and it["price"] > 0)]
        if len(priced) < min_size:
            out[k] = v; continue
        # Initial CV gate — uniform-price clusters are not touched.
        prs = [it["price"] for it in priced]
        mean_p = _st.mean(prs)
        std_p = _st.pstdev(prs) if len(prs) > 1 else 0.0
        if not mean_p or std_p / mean_p <= min_cv:
            out[k] = v; continue

        # ADAPTIVE THRESHOLD — trigger aggressive mode (`extreme_mad_k`) when
        # the cluster shows either:
        #   A) high robust dispersion (MAD/median ≥ multi_pop_madmed=0.40):
        #      catches "bimodal moderate-range" clusters like Tamron lenses
        #      (€140-€650 mediana €308 → MAD/median ~0.50);
        #   B) extreme min→max ratio (≥ extreme_range_ratio=20×):
        #      catches "broad-range tail-heavy" clusters where the median
        #      MAD stays small but a few extreme outliers (€11-€755 = 69×)
        #      hide other populations.
        # Either signal alone is enough — they catch different shapes of the
        # multi-population problem. Both off → cluster looks single-product.
        _med0 = _st.median(prs)
        _mad0 = _st.median(abs(p - _med0) for p in prs)
        _min_p, _max_p = min(prs), max(prs)
        trigger_a = _med0 > 0 and _mad0 / _med0 >= multi_pop_madmed
        trigger_b = _min_p > 0 and _max_p / _min_p >= extreme_range_ratio
        # Sprint-4: NEW trigger C — log-price skew > 1.5 catches asymmetric
        # distributions (a long high tail next to a tight main body) that
        # don't trip A or B (small MAD/median, moderate max/min).
        import math as _m
        log_prs = [_m.log(p) for p in prs if p > 0]
        if len(log_prs) >= 4:
            lm = _st.mean(log_prs)
            lsd = _st.pstdev(log_prs)
            if lsd > 0:
                skew = sum((x - lm) ** 3 for x in log_prs) / (len(log_prs) * lsd**3)
            else:
                skew = 0.0
        else:
            skew = 0.0
        trigger_c = abs(skew) >= 1.5
        if trigger_a or trigger_b or trigger_c:
            k_eff = extreme_mad_k
            n_extreme_range += 1
        else:
            # Sprint-4: adaptive mad_k by partition size. Large clusters have
            # statistically stable MAD; we can be more aggressive (2.2 instead
            # of 2.5) since the false-split risk is lower. Very large (>30):
            # mad_k=1.8 — like the extreme branch but without the trigger.
            if len(priced) >= 30:
                k_eff = 1.8
            elif len(priced) >= 15:
                k_eff = 2.2
            else:
                k_eff = mad_k

        # #6: optionally measure dispersion in log-price space (symmetric in
        # log == multiplicative in €, the correct geometry for lognormal prices;
        # raw-€ symmetric thresholds over-split the right tail).
        import math as _mth
        _val = (lambda p: _mth.log(p)) if LOGSPACE_BITAIL else (lambda p: p)

        main = list(priced)
        original_n = len(priced)
        max_moved = int(original_n * max_total_tail_frac)
        moved_so_far = 0
        high_tails: list[list] = []
        low_tails:  list[list] = []

        # ── RIGHT-tail iterations ────────────────────────────────────────
        iter_idx = 0
        while iter_idx < max_iters and len(main) >= min_main + min_tail:
            curr_v = sorted(_val(it["price"]) for it in main)
            med = _st.median(curr_v)
            mad_raw = _st.median(abs(p - med) for p in curr_v)
            if mad_raw <= 0:
                break
            thr_hi = med + k_eff * mad_raw
            tail = [it for it in main if _val(it["price"]) > thr_hi]
            new_main = [it for it in main if _val(it["price"]) <= thr_hi]
            if len(tail) < min_tail or len(new_main) < min_main:
                break
            if moved_so_far + len(tail) > max_moved:
                break
            high_tails.append(tail)
            main = new_main
            moved_so_far += len(tail)
            iter_idx += 1
            n_total_iters += 1

        # ── LEFT-tail iterations (on the residual main) ──────────────────
        iter_idx = 0
        while iter_idx < max_iters and len(main) >= min_main + min_tail:
            curr_v = sorted(_val(it["price"]) for it in main)
            med = _st.median(curr_v)
            mad_raw = _st.median(abs(p - med) for p in curr_v)
            if mad_raw <= 0:
                break
            thr_lo = med - k_eff * mad_raw
            tail = [it for it in main if _val(it["price"]) < thr_lo]
            new_main = [it for it in main if _val(it["price"]) >= thr_lo]
            if len(tail) < min_tail or len(new_main) < min_main:
                break
            if moved_so_far + len(tail) > max_moved:
                break
            low_tails.append(tail)
            main = new_main
            moved_so_far += len(tail)
            iter_idx += 1
            n_total_iters += 1

        if not high_tails and not low_tails:
            out[k] = v; continue

        # Main cluster keeps the no-priced items (we can't grade them).
        out[k] = main + noprice
        # #highN: emit lowest-priced tail first (closer to main)
        high_tails.sort(key=lambda t: _st.median(it["price"] for it in t))
        for i, t in enumerate(high_tails, 1):
            out[f"{k} #high{i}"] = t
            n_high += 1
        # #lowN: emit highest-priced first (closer to main)
        low_tails.sort(key=lambda t: -_st.median(it["price"] for it in t))
        for i, t in enumerate(low_tails, 1):
            out[f"{k} #low{i}"] = t
            n_low += 1
        n_clusters_split += 1
        n_total_moved    += sum(len(t) for t in high_tails) + sum(len(t) for t in low_tails)

    if n_clusters_split:
        log_fn(f"  bi-tail split (MAD×{mad_k} / extreme×{extreme_mad_k} if MAD/med≥{multi_pop_madmed}, "
               f"iter≤{max_iters} each side): "
               f"{n_clusters_split} cluster spaccati ({n_high} #high, {n_low} #low), "
               f"{n_extreme_range} multi-pop detected, "
               f"{n_total_moved} item spostati in code")
    return out


def _price_split_jenks2(groups: dict, emb_of, log_fn, *,
                         min_size: int = 8,
                         min_cv: float = 0.5,
                         min_band: int = 3,
                         min_gvf: float = 0.60,         # sprint-4: 0.55→0.60
                         min_center_ratio: float = 2.0,
                         max_emb_cos: float = 0.95,
                         high_conf_gvf: float = 0.65,    # was 0.75, tuned via
                         high_conf_ratio: float = 4.0    # bench_pipeline.py
                         ) -> dict:                       # (+3.3 SCORE delta)
    """Optimal 1-D 2-break (Jenks / k-means k=2) split on log(price), with
    multiple guards. Where `_price_band_split` requires a clean chasm, this
    handles continuous-but-bimodal distributions (e.g. Wii câbles + consoles
    with intermediate prices smoothing out the gap).

    A split is accepted ONLY when ALL hold:
      - `len(items) >= min_size`                — enough mass to subdivide
      - price-CV > `min_cv`                     — some real spread
      - both sub-groups `>= min_band` items     — never shave a single deal
      - GVF (1 - within_var/total_var) `>= min_gvf` — real bimodality, not noise
      - center ratio `>= min_center_ratio` (linear) — substantial price gap
      - cosine of embedding centroids `< max_emb_cos` — different products
        (same product with new/used variance → near-identical titles → high
        cosine → NOT split, so an underpriced item still stands out as a deal)
    """
    import math
    import statistics as _st
    out: dict[str, list] = {}
    n_split = n_emb_block = 0
    for k, v in groups.items():
        priced = [it for it in v if it.get("price") and it["price"] > 0]
        noprice = [it for it in v if not (it.get("price") and it["price"] > 0)]
        if len(priced) < max(min_size, 2 * min_band):
            out[k] = v; continue
        prs = sorted(it["price"] for it in priced)
        m = _st.mean(prs); sd = _st.pstdev(prs) if len(prs) > 1 else 0.0
        if not m or sd / m <= min_cv:
            out[k] = v; continue
        lp = [math.log(p) for p in prs]
        mu = sum(lp) / len(lp)
        tot = sum((x - mu) ** 2 for x in lp)
        if tot <= 0:
            out[k] = v; continue
        # Optimal 2-break: minimize within-class variance.
        best = None
        for i in range(min_band, len(lp) - min_band + 1):
            L = lp[:i]; R = lp[i:]
            mL = sum(L) / len(L); mR = sum(R) / len(R)
            w = sum((x - mL) ** 2 for x in L) + sum((x - mR) ** 2 for x in R)
            if best is None or w < best[0]:
                best = (w, i, mL, mR)
        w, i, mL, mR = best
        gvf = 1 - w / tot
        ratio = math.exp(mR - mL)
        if gvf < min_gvf or ratio < min_center_ratio:
            out[k] = v; continue
        # Same sort key as `prs` so the split index `i` lines up with items.
        priced_sorted = sorted(priced, key=lambda it: it["price"])
        A = priced_sorted[:i]; B = priced_sorted[i:]
        # HIGH-CONFIDENCE bypass: when GVF AND ratio are both overwhelming, the
        # price evidence alone is conclusive. Needed because Vinted titles often
        # share strong tokens across products in the same cluster (e.g. "Nintendo
        # Wii" appears in both cables AND consoles → emb-cos > 0.95 and the
        # check would wrongly block an obvious bimodal split).
        high_conf = (gvf >= high_conf_gvf and ratio >= high_conf_ratio)
        # Embedding dissimilarity — the strongest deal-vs-product guard for
        # MEDIUM-confidence splits.
        if not high_conf and emb_of is not None:
            try:
                vecs_a = [emb_of[it["id"]] for it in A if it["id"] in emb_of]
                vecs_b = [emb_of[it["id"]] for it in B if it["id"] in emb_of]
                if len(vecs_a) < min_band or len(vecs_b) < min_band:
                    out[k] = v; continue
                ca = np.mean(vecs_a, axis=0); cb = np.mean(vecs_b, axis=0)
                ca = ca / max(float(np.linalg.norm(ca)), 1e-12)
                cb = cb / max(float(np.linalg.norm(cb)), 1e-12)
                if float(ca @ cb) > max_emb_cos:
                    n_emb_block += 1
                    out[k] = v; continue
            except Exception:
                out[k] = v; continue
        n_split += 1
        out[f"{k} [€{int(A[0]['price'])}-{int(A[-1]['price'])}]"] = A
        out[f"{k} [€{int(B[0]['price'])}-{int(B[-1]['price'])}]"] = B
        if noprice:
            out.setdefault(f"{k} [?]", []).extend(noprice)
    if n_split or n_emb_block:
        log_fn(f"  price-split (Jenks2): {n_split} cluster spaccati, "
               f"{n_emb_block} bloccati da emb-check (stesso prodotto)")
    return out


# ─── SPRINT 7 — Architect interventions #3, #4, #5, #7 ──────────────────


def _drop_floor_spam(groups: dict, log_fn, *,
                      floor_price: float = 10.0,
                      min_cluster_size: int = 15,
                      max_ratio_to_median: float = 0.20,
                      max_drop_frac: float = 0.10) -> dict:
    """Intervento #5: isolate floor-priced listing-bait (€10 stubs) into a
    side `#floorspam` cluster so they don't pollute the main cluster's price
    baseline. Activates only on clusters with:
       - size >= min_cluster_size (≥15 items)
       - median price >= 30€ (otherwise the cluster IS cheap by nature)
       - candidate items: price ≤ min(floor*1.5, median * 0.20)
       - max max_drop_frac of cluster can be dropped (≤10%, no main shave)
    """
    import statistics as _st
    out: dict[str, list] = {}
    n_dropped = 0
    for k, v in groups.items():
        pr = [it["price"] for it in v if it.get("price") and it["price"] > 0]
        if len(v) < min_cluster_size or len(pr) < 8:
            out[k] = v; continue
        med = _st.median(pr)
        if med < 30:
            out[k] = v; continue
        threshold = min(floor_price * 1.5, med * max_ratio_to_median)
        to_drop = [it for it in v
                   if it.get("price") and 0 < it["price"] <= threshold]
        if not to_drop or len(to_drop) > max_drop_frac * len(v):
            out[k] = v; continue
        dropped_ids = {id(it) for it in to_drop}
        out[k] = [it for it in v if id(it) not in dropped_ids]
        out.setdefault(f"{k} #floorspam", []).extend(to_drop)
        n_dropped += len(to_drop)
    if n_dropped:
        log_fn(f"  floor-spam drop: {n_dropped} listing-bait isolati in #floorspam")
    return out


def _split_blob_by_first_token(groups: dict, log_fn, *,
                                min_size: int = 50,
                                min_distinct_ratio: float = 0.50,
                                min_subgroup: int = 4) -> dict:
    """Intervento #4: detect 'mappazzoni' (cluster-discarica) where titles
    start with mostly DIFFERENT first informative tokens — HDBSCAN aggregated
    on shared trailing tokens. Split by first_informative_token when at least
    2 sub-groups of size ≥ min_subgroup emerge. Leftover items stay in the
    original cluster.
    """
    from collections import Counter as _C
    out: dict[str, list] = {}
    n_blob = 0
    for k, v in groups.items():
        if len(v) < min_size:
            out[k] = v; continue
        firsts: list[str] = []
        for it in v:
            toks = [t for t in _normalize(it.get("title", "")).split()
                    if len(t) >= 3 and t not in _NOISE]
            firsts.append(toks[0] if toks else "")
        distinct_ratio = len({f for f in firsts if f}) / max(1, len(firsts))
        if distinct_ratio < min_distinct_ratio:
            out[k] = v; continue
        fc = _C(firsts)
        big = [t for t, c in fc.items() if c >= min_subgroup and t]
        if len(big) < 2:
            out[k] = v; continue
        buckets: dict[str, list] = {t: [] for t in big}
        leftover: list = []
        for it, ft in zip(v, firsts):
            if ft in buckets:
                buckets[ft].append(it)
            else:
                leftover.append(it)
        for ft, items_in in buckets.items():
            out[f"{k} ::{ft}"] = items_in
        if leftover:
            out[k] = leftover
        n_blob += 1
    if n_blob:
        log_fn(f"  blob-split (first-token): {n_blob} cluster-discarica spaccati")
    return out


def _merge_sibling_fragments(groups: dict, emb_of, log_fn, *,
                              min_size: int = 8,
                              max_median_ratio: float = 1.15,
                              min_emb_cos: float = 0.97) -> dict:
    """Intervento #3: re-merge `<base>` + `<base>#1` + `<base>#2` ... when
    their median prices differ by less than 15% AND text centroids are nearly
    identical (cos ≥ 0.97). This re-unifies fragments produced by the price
    feature within brand-homogeneous partitions where it spuriously split
    same-product price tiers (€95-105 vs €105-120 of the same iPhone 11).
    Preserves #high/#low/#img/#floorspam/#imgN suffixes — only strips bare #N.
    """
    import statistics as _st
    from collections import defaultdict
    families: dict[str, list[str]] = defaultdict(list)
    # Only fuse bare "#N" siblings (#1/#2/#3), NOT #highN/#lowN/#imgN/#floorspam
    _BARE_SUFFIX = re.compile(r"#\d+$")
    for k in list(groups.keys()):
        base = _BARE_SUFFIX.sub("", k)
        # Don't fuse semantic suffixes
        if any(tag in k for tag in (" #high", " #low", " #img", " #floor", "[€", "[?]", "::")):
            continue
        families[base].append(k)
    out = dict(groups)
    n_merged = 0
    for base, siblings in families.items():
        if len(siblings) < 2:
            continue
        stats = []
        for k in siblings:
            v = out.get(k, [])
            if len(v) < min_size:
                continue
            pr = [it["price"] for it in v if it.get("price") and it["price"] > 0]
            if len(pr) < 3:
                continue
            med = _st.median(pr)
            embs_v = [emb_of[it["id"]] for it in v if it["id"] in emb_of]
            if len(embs_v) < 3:
                continue
            cen = np.mean(embs_v, axis=0)
            cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
            # Compute medoid name (cf. lex-sig merge — same rationale)
            dists = np.array([1.0 - float(emb_of[it["id"]] @ cen) for it in v
                              if it["id"] in emb_of])
            medoid_idx = int(np.argmin(dists))
            medoid_item = [it for it in v if it["id"] in emb_of][medoid_idx]
            name_key = (_cluster_sig(v) if STABLE_MERGE_KEY
                        else _make_embedding_cluster_key(medoid_item.get("title", "")))
            stats.append((k, med, cen, name_key))
        if len(stats) < 2:
            continue
        # Union-find pairwise: merge whenever ratio<=1.15 AND cos>=0.97 AND
        # the medoid-derived display name is IDENTICAL between siblings
        # (prevents "iphone 15 pro" + "iphone 15 pro max" fusion).
        parent = {s[0]: s[0] for s in stats}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                ki, mi, ci, mki = stats[i]
                kj, mj, cj, mkj = stats[j]
                if mi <= 0 or mj <= 0:
                    continue
                if mki != mkj:
                    continue   # NAME-EQUALITY guard
                ratio = max(mi, mj) / min(mi, mj)
                cos = float(ci @ cj)
                if ratio <= max_median_ratio and cos >= min_emb_cos:
                    ri, rj = find(ki), find(kj)
                    if ri != rj:
                        canonical = base if base in (ri, rj) else min(ri, rj)
                        other = rj if canonical == ri else ri
                        parent[other] = canonical
        # Apply merges
        merged_into: dict[str, str] = {}
        for k in [s[0] for s in stats]:
            root = find(k)
            if root != k:
                merged_into[k] = root
        for src, dst in merged_into.items():
            if src in out and dst in out and src != dst:
                out[dst] = out[dst] + out.pop(src)
                n_merged += 1
    if n_merged:
        log_fn(f"  sibling-merge: {n_merged} cluster fratelli fusi "
               f"(ratio≤{max_median_ratio}, emb_cos≥{min_emb_cos})")
    return out


def _merge_by_lex_signature(groups: dict, emb_of, log_fn, *,
                             min_size: int = 4,
                             min_emb_cos: float = 0.98,
                             max_median_ratio: float = 1.10) -> dict:
    """Intervento #7 (RIVISTO sprint-7b): coalesce clusters with identical
    lex_sig AND same medoid-derived cluster-key. Without name-equality this
    fused iPhone 15 Pro with iPhone 15 Pro Max (same brand, model="15", close
    centroids, price ratio ~1.20). Now requires:
       - same brand canonical
       - shared model-number tokens
       - centroids cos ≥ 0.98 (was 0.96) — VERY close in text space
       - median price ratio ≤ 1.10 (was 1.25) — almost same price
       - same `_make_embedding_cluster_key(medoid_title)` — identical
         normalized name (this is the user's "stesso nome rappresentativo"
         requirement: iphone 15 pro max != iphone 15 pro 128)
    """
    import statistics as _st
    from collections import Counter as _C
    groups_meta = []
    for k, v in groups.items():
        if len(v) < min_size:
            continue
        # Skip semantic-suffix clusters (already split intentionally)
        if any(tag in k for tag in (" #high", " #low", " #img", " #floor", "[€", "[?]", "::")):
            continue
        brs, nums_all = [], []
        for it in v:
            b, nums = _lex_sig(it.get("title", ""), it.get("brand"))
            if b:
                brs.append(b)
            nums_all.extend(nums)
        if not brs:
            continue
        brand = max(set(brs), key=brs.count)
        nums_top = {n for n, c in _C(nums_all).most_common(3)
                    if c >= len(v) * 0.3}
        if not nums_top:
            continue
        pr = [it["price"] for it in v if it.get("price") and it["price"] > 0]
        med = _st.median(pr) if pr else 0
        embs = [emb_of[it["id"]] for it in v if it["id"] in emb_of]
        if len(embs) < 3:
            continue
        cen = np.mean(embs, axis=0)
        cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
        # Compute medoid title → normalized cluster-key name (= what the
        # display label would show). Two clusters are mergeable ONLY if this
        # name is identical (user requirement: "stesso nome rappresentativo").
        dists = np.array([1.0 - float(emb_of[it["id"]] @ cen) for it in v
                          if it["id"] in emb_of])
        medoid_idx = int(np.argmin(dists))
        medoid_item = [it for it in v if it["id"] in emb_of][medoid_idx]
        name_key = (_cluster_sig(v) if STABLE_MERGE_KEY
                    else _make_embedding_cluster_key(medoid_item.get("title", "")))
        groups_meta.append((k, brand, nums_top, med, cen, name_key))
    out = dict(groups)
    used = set()
    n_merged = 0
    for i in range(len(groups_meta)):
        ki, bi, ni, mi, ci, mki = groups_meta[i]
        if ki in used or ki not in out:
            continue
        for j in range(i + 1, len(groups_meta)):
            kj, bj, nj, mj, cj, mkj = groups_meta[j]
            if kj in used or kj not in out:
                continue
            if bi != bj or not (ni & nj):
                continue
            # NAME-EQUALITY guard: same medoid-derived cluster name required.
            if mki != mkj:
                continue
            if mi > 0 and mj > 0:
                ratio = max(mi, mj) / min(mi, mj)
                if ratio > max_median_ratio:
                    continue
            if float(ci @ cj) < min_emb_cos:
                continue
            out[ki] = out[ki] + out.pop(kj)
            used.add(kj)
            n_merged += 1
    if n_merged:
        log_fn(f"  lex-sig merge: {n_merged} cluster con stessa firma "
               f"brand+modello+nome fusi (cos≥{min_emb_cos}, ratio≤{max_median_ratio})")
    return out


# ─── REVIEW INTERVENTIONS (toggleable; defaults preserve current behavior) ──
# Each flag defaults to the *current* pipeline so refine_clusters is a no-op
# change until a flag is flipped (the A/B harness flips them per-variant).
ENABLE_MODEL_TOKEN_SPLIT = False# #2: lexical model-code split (CPU/GPU)
ENABLE_OUTLIER_EJECT     = False   # adaptive: eject members far from a TIGHT
                                   # cluster's centroid (threshold rises with the
                                   # cluster's own cohesion) — kicks out accessories
                                   # /wrong-brand intruders the price-split misses
ENABLE_MIXTURE_SPLIT     = True# FINAL pass: split device+accessory/game price
                                   # mixtures (gvf≥0.6, ratio≥3, no emb gate) that
                                   # share the brand token so text can't separate
ENABLE_PRODUCT_SPLIT     = False# storage-aware model-id split, ALL clusters
                                   # (iPhone 12≠13, Galaxy S21≠S22, i5-6600≠6500;
                                   # storage tiers 64≠256 and years are ignored)
STABLE_MERGE_KEY         = True# #8: deterministic merge name-key (vs medoid)
LOGSPACE_BITAIL          = False# #6: bi-tail MAD in log-price space
CONSOLIDATE_PRICE_SPLIT  = False# #10: drop the 2nd Jenks2 pass + baseline guard
DISABLE_BITAIL           = False# experiment: skip ONLY the bi-tail (#high/#low)
                                   # pure price-tier split; keep Jenks2 (which is
                                   # emb-gated → only splits real product mixtures)
DISABLE_PRICE_SPLIT      = False# experiment: skip ALL price splitting (Jenks2
                                   # ×2 + bi-tail) — isolates the coverage↔purity
                                   # fragmentation lever. SEMANTIC splits (mixtures)
                                   # are what we want; pure price-tier splits drop
                                   # items below the detector's size floor.

# Tunable refine thresholds (defaults == current production values, so leaving
# them untouched is a no-op). The config-search (scripts/ab_search.py) patches
# these to explore the clustering-quality space.
TUNE_JENKS_GVF     = 0.6# _price_split_jenks2 min GVF (↑ = split less)
TUNE_JENKS_RATIO   = 2.0# _price_split_jenks2 min center ratio
TUNE_JENKS_COS     = 0.97# _price_split_jenks2 max emb-cos (↓ = split less)
TUNE_BITAIL_MADK   = 2.5# _right_tail_split mad_k (↑ = split less)
TUNE_SIBLING_RATIO = 1.15# _merge_sibling_fragments max median ratio (↑ = merge more)
TUNE_SIBLING_COS   = 0.97# _merge_sibling_fragments min emb-cos (↓ = merge more)
TUNE_LEXSIG_COS    = 0.98# _merge_by_lex_signature min emb-cos (↓ = merge more)
TUNE_LEXSIG_RATIO  = 1.1# _merge_by_lex_signature max median ratio (↑ = merge more)
TUNE_MIX_GVF       = 0.6# _mixture_split min GVF (↑ = split fewer mixtures)
TUNE_MIX_RATIO     = 2.5# _mixture_split min modal ratio (↑ = split fewer)
TUNE_EJECT_K       = 3.0    # _eject_embedding_outliers robust-z (↓ = eject more)
TUNE_EJECT_MINMED  = 0.92   # only eject from clusters whose median cohesion ≥ this
TUNE_EJECT_FLOOR   = 0.90   # never eject members with cos ≥ this (legit variation)

# ── #2: model-token split for tech (CPU/GPU/RAM/SSD) partitions ─────────────
_TECH_BRANDS = {"intel", "amd", "nvidia"}
_TECH_CAT_HINTS = ("processor", "processore", "processeur", "procesador",
                   "cpu", "gpu", "graphics", "scheda video", "scheda madre",
                   "motherboard", "ram", "memoria ram", "ssd", "hard disk",
                   "hdd", "componenti")


def _is_tech_cluster(v: list[dict]) -> bool:
    brs, cats = set(), set()
    for it in v:
        b, _ = _lex_sig(it.get("title", ""), it.get("brand"))
        if b:
            brs.add(b)
        c = (it.get("category") or "").lower()
        if c:
            cats.add(c)
    if brs & _TECH_BRANDS:
        return True
    return any(h in c for c in cats for h in _TECH_CAT_HINTS)


def _pick_model_token(nums) -> str | None:
    """Pick the most discriminating model-code token (the SKU number, e.g.
    '6600' in 'i5 6600k'). Prefer the most-digits token, tie-break by length.
    Strip a single trailing letter so '6600k'/'6600' don't fragment a chip."""
    if not nums:
        return None
    def _digits(t):
        return sum(c.isdigit() for c in t)
    cand = max(nums, key=lambda t: (_digits(t), len(t)))
    # only meaningful as a SKU if it carries ≥3 digits (i5/i7 alone won't split)
    if _digits(cand) < 3:
        return None
    m = re.match(r"([a-z]*\d+)[a-z]?$", cand)
    return m.group(1) if m else cand


def _split_by_model_token(groups: dict, log_fn, *,
                          min_size: int = 8, min_subgroup: int = 3) -> dict:
    """#2: in tech partitions, split a cluster by exact model-code token when
    ≥2 distinct SKUs each have ≥min_subgroup items. Dense text encoders treat
    'i5 6600' and 'i5 6500' as near-identical → HDBSCAN merges them; this
    separates them lexically. Gated to tech clusters to avoid over-splitting."""
    from collections import defaultdict
    out: dict[str, list] = {}
    n_split = 0
    for k, v in groups.items():
        if any(tag in k for tag in (" #high", " #low", " #img", " #floor",
                                    "[€", "[?]", "::")):
            out[k] = v; continue
        if len(v) < min_size or not _is_tech_cluster(v):
            out[k] = v; continue
        buckets: dict[str, list] = defaultdict(list)
        for it in v:
            _, nums = _lex_sig(it.get("title", ""), it.get("brand"))
            mt = _pick_model_token(nums)
            buckets[mt].append(it)
        big = {t: g for t, g in buckets.items() if t and len(g) >= min_subgroup}
        if len(big) < 2:
            out[k] = v; continue
        leftover = [it for t, g in buckets.items()
                    if not (t and len(g) >= min_subgroup) for it in g]
        for t, g in big.items():
            out[f"{k} ::m:{t}"] = g
        if leftover:
            out[k] = leftover
        n_split += 1
    if n_split:
        log_fn(f"  model-token split: {n_split} cluster tech spaccati per SKU")
    return out


_YEAR_TOK = re.compile(r"^(19|20)\d\d$")


def _model_ids(title: str) -> frozenset:
    """Model-identity numeric tokens from a title, in the same normalized space
    the pipeline uses. EXCLUDES storage tiers ('128' in '128 gb') and listing
    years ('2021') — those are specs/metadata, not product identity. So
    'iphone 12 128gb' and 'iphone 12 256gb' share id {'12'}, while 'iphone 13'
    differs ({'13'})."""
    toks = _normalize(title).split()
    out = set()
    for i, t in enumerate(toks):
        if not any(c.isdigit() for c in t):
            continue
        if _YEAR_TOK.match(t) or t in _NOISE or len(t) > 7:
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t.isdigit() and nxt in ("gb", "tb", "mb"):     # storage tier → skip
            continue
        out.add(t)
    return frozenset(out)


def _split_by_product_id(groups: dict, log_fn, *,
                         min_size: int = 12, min_subgroup: int = 4) -> dict:
    """Split a cluster into distinct model identities when ≥2 model-id groups
    each have ≥min_subgroup items. Items with no model-id (number-less titles,
    e.g. 'Nintendo Switch Lite') stay together in the main cluster — they are
    not split. Catches HDBSCAN over-merges of near-homograph models that a dense
    text encoder can't separate (iPhone 12/13, Galaxy S21/S22, i5-6600/6500)."""
    out: dict[str, list] = {}
    n_split = 0
    for k, v in groups.items():
        if any(tag in k for tag in (" #high", " #low", " #img", " #floor",
                                    "[€", "[?]", "::")):
            out[k] = v; continue
        if len(v) < min_size:
            out[k] = v; continue
        sig: dict[frozenset, list] = {}
        for it in v:
            sig.setdefault(_model_ids(it.get("title", "")), []).append(it)
        big = {s: g for s, g in sig.items() if s and len(g) >= min_subgroup}
        if len(big) < 2:
            out[k] = v; continue
        leftover = [it for s, g in sig.items()
                    if not (s and len(g) >= min_subgroup) for it in g]
        for s, g in big.items():
            tag = "-".join(sorted(s))[:20]
            out[f"{k} ::p:{tag}"] = g
        if leftover:
            out[k] = leftover
        n_split += 1
    if n_split:
        log_fn(f"  product-id split: {n_split} cluster spaccati per identità modello")
    return out


def _eject_embedding_outliers(groups: dict, emb_of, log_fn, *,
                              min_size: int = 10, k: float = 3.0,
                              min_median_cos: float = 0.92,
                              max_keep_cos: float = 0.90,
                              max_eject_frac: float = 0.30) -> dict:
    """Adaptive cohesion-gated purity filter (user's idea): in a TIGHT cluster
    (median cos-to-centroid ≥ min_median_cos) the bar to belong is high, so a
    member whose cos sits below `median − k·MAD` is an embedding outlier (an
    accessory/part/wrong-brand intruder that shares the brand token) and is
    EJECTED to a singleton. Loose clusters (low median cohesion) are left
    untouched — the threshold literally rises with the cluster's own tightness.
    Catches contamination the price-split misses (same price, different product)."""
    import statistics as _st
    out: dict[str, list] = {}
    n_ej = 0
    for key, v in groups.items():
        ids = [it for it in v if it["id"] in emb_of]
        if len(ids) < min_size:
            out[key] = v; continue
        V = np.array([emb_of[it["id"]] for it in ids], dtype=np.float32)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        cen = V.mean(0); cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
        sims = V @ cen
        med = float(np.median(sims))
        if med < min_median_cos:
            out[key] = v; continue                 # loose cluster → don't touch
        mad = 1.4826 * float(np.median(np.abs(sims - med))) or 1e-6
        thr = med - k * mad
        # Eject only items that are BOTH statistical outliers AND below an
        # absolute floor — so legit title variation in a very tight cluster
        # (tiny MAD) isn't ejected, only genuinely-different products.
        ej_idx = [i for i, s in enumerate(sims) if s < thr and s < max_keep_cos]
        if not ej_idx or len(ej_idx) > max_eject_frac * len(ids):
            out[key] = v; continue
        ej_set = {ids[i]["id"] for i in ej_idx}
        out[key] = [it for it in v if it["id"] not in ej_set]
        for j, i in enumerate(ej_idx):
            out[f"{key} #out{j}"] = [ids[i]]        # unique singleton, drops out
        n_ej += len(ej_idx)
    if n_ej:
        log_fn(f"  outlier-eject: {n_ej} intrusi espulsi da cluster compatti "
               f"(soglia adattiva med−{k}·MAD)")
    return out


def _mixture_split(groups: dict, log_fn, *, min_gvf: float = 0.60,
                   min_ratio: float = 3.0, min_band: int = 3,
                   max_iters: int = 2) -> dict:
    """FINAL device+accessory de-mixer. Splits a cluster on its log-price Jenks2
    break when the two modes are strong (GVF≥min_gvf) and far apart
    (ratio≥min_ratio), each ≥min_band items — NO embedding gate (the mixture
    shares the brand+model token, so text similarity is high; price bimodality
    is the only reliable signal). Iterates so a 3-way mix (accessory/game/device)
    fully separates. Runs after the merges so it also catches mixtures the merges
    re-created."""
    import math
    work = list(groups.items())
    out: dict[str, list] = {}
    n_split = 0
    while work:
        k, v = work.pop()
        priced = [it for it in v if it.get("price") and it["price"] > 0]
        noprice = [it for it in v if not (it.get("price") and it["price"] > 0)]
        if len(priced) < 2 * min_band:
            out[k] = v; continue
        stats = _jenks2_stats([it["price"] for it in priced], min_band=min_band)
        if not stats:
            out[k] = v; continue
        idx, gvf, ratio = stats
        if gvf < min_gvf or ratio < min_ratio:
            out[k] = v; continue
        sp = sorted(priced, key=lambda it: it["price"])
        A = sp[:idx]; B = sp[idx:]
        n_split += 1
        ka = f"{k} [€{int(A[0]['price'])}-{int(A[-1]['price'])}]"
        kb = f"{k} [€{int(B[0]['price'])}-{int(B[-1]['price'])}]"
        # re-feed both sides (bounded) so multi-modal clusters fully separate
        depth = k.count("[€")
        if depth < max_iters:
            work.append((ka, A)); work.append((kb, B + noprice))
        else:
            out[ka] = A; out[kb] = B + noprice
            continue
    if n_split:
        log_fn(f"  mixture-split: {n_split} split device+accessori/giochi per prezzo")
    return out


def _cluster_sig(v: list[dict]) -> tuple:
    """#8: deterministic merge name-key = (brand, dominant model-nums, family).
    Order-independent (unlike the single medoid title), so a legitimate merge
    isn't missed just because the medoid changed between runs."""
    from collections import Counter
    brs, nums_all, fams = [], [], []
    for it in v:
        b, nums = _lex_sig(it.get("title", ""), it.get("brand"))
        if b:
            brs.append(b)
        nums_all.extend(nums)
        toks = set(_TRANSLATE.get(t, t)
                   for t in _normalize(it.get("title", "")).split())
        for f in _FAMILY_TOKENS_FOR_KEY:
            if f in toks:
                fams.append(f); break
    brand = max(set(brs), key=brs.count) if brs else ""
    nums_top = frozenset(n for n, c in Counter(nums_all).most_common(3)
                         if c >= len(v) * 0.3)
    fam = max(set(fams), key=fams.count) if fams else ""
    return (brand, nums_top, fam)


def refine_clusters(items: list[dict], mapping: dict[int, str], conn,
                    log_fn=print, emb_cache=None,
                    gate_hi: float | None = None,
                    gate_lo: float | None = None) -> dict[int, str]:
    """Post-process a density-clustering `mapping` ({item_id: cluster_key}):

      1. Lexical-gated singleton rescue — absorb leftover singletons into the
         nearest multi-item cluster when embeddings are close (HI) or close-ish
         AND brand/model agree (LO). See RESCUE_GATE_* for the calibration.
         `gate_hi`/`gate_lo` override the module defaults (the thresholds are
         model-specific — e5 sims sit on a higher scale than MiniLM).
      2. Price-band split of price-incoherent clusters.

    Then recompute & persist centroids for the final key set. `items` must carry
    id/title/brand/price. Returns the refined mapping. No-op without embeddings.
    """
    if not EMBEDDINGS_AVAILABLE or not items:
        return mapping
    gate_hi = RESCUE_GATE_HI if gate_hi is None else gate_hi
    gate_lo = RESCUE_GATE_LO if gate_lo is None else gate_lo

    by_id = {it["id"]: it for it in items}
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(mapping.get(it["id"], "unknown"), []).append(it)

    # Embeddings (reused for centroids + rescue). Use the shared cache when the
    # caller already embedded the corpus, else embed here.
    if emb_cache is not None:
        raw = np.array([emb_cache[it["id"]] for it in items])
    else:
        model = _get_model()
        raw = model.encode([_embed_text(it) for it in items],
                           show_progress_bar=False, convert_to_numpy=True, batch_size=256)
    raw = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    emb_of = {it["id"]: e for it, e in zip(items, raw)}

    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    singletons = [v[0] for k, v in groups.items() if len(v) == 1]

    n_rescued = 0
    if multi and singletons:
        mk = list(multi.keys())
        cmat = np.array([
            np.mean([emb_of[it["id"]] for it in multi[k]], axis=0) for k in mk
        ])
        cmat = cmat / np.maximum(np.linalg.norm(cmat, axis=1, keepdims=True), 1e-12)
        # lexical signature per cluster
        cbrand, cnums = {}, {}
        for k in mk:
            sigs = [_lex_sig(it.get("title", ""), it.get("brand")) for it in multi[k]]
            brs = [b for b, _ in sigs if b]
            cbrand[k] = max(set(brs), key=brs.count) if brs else ""
            cc: dict[str, int] = {}
            for _, nums in sigs:
                for nm in nums:
                    cc[nm] = cc.get(nm, 0) + 1
            thr = max(2, 0.3 * len(multi[k]))
            cnums[k] = {nm for nm, c in cc.items() if c >= thr}

        for it in singletons:
            e = emb_of[it["id"]]
            sims = cmat @ e
            j = int(sims.argmax())
            s = float(sims[j])
            k = mk[j]
            absorb = False
            if s >= gate_hi:
                absorb = True
            elif s >= gate_lo:
                b, nums = _lex_sig(it.get("title", ""), it.get("brand"))
                if b and b == cbrand[k] and (not cnums[k] or (nums & cnums[k])):
                    absorb = True
            if absorb:
                mapping[it["id"]] = k
                multi[k].append(it)
                n_rescued += 1
        log_fn(f"  rescued {n_rescued}/{len(singletons)} singletons (lexical-gated)")

    # Rebuild groups from the rescued mapping.
    groups = {}
    for it in items:
        groups.setdefault(mapping[it["id"]], []).append(it)
    # Price-split (Jenks2, 2026-05-26): optimal 1-D 2-break on log(price) with
    # multi-tier guards. Where `_price_band_split` needed a clean chasm, this
    # also catches continuous-but-bimodal distributions (e.g. Wii câbles + Wii
    # consoles in the same cluster) without ever shaving a single deal.
    # Dry-run (scripts/dry_run_price_split.py) on 88k items: 544 splits over
    # 2507 candidates — 1754 blocked by emb-check (same-product protection).
    # Old gap-based `_price_band_split` kept above for reference / opt-in.
    if not DISABLE_PRICE_SPLIT:
        groups = _price_split_jenks2(groups, emb_of, log_fn,
                                     min_gvf=TUNE_JENKS_GVF,
                                     min_center_ratio=TUNE_JENKS_RATIO,
                                     max_emb_cos=TUNE_JENKS_COS)

    # Sprint-7 #5: drop floor-priced listing-bait (€10 stubs) into side
    # #floorspam cluster BEFORE bi-tail so the MAD calculation sees clean
    # prices. Without this, the bait pulls the bi-tail threshold so far down
    # it never fires on real outliers.
    groups = _drop_floor_spam(groups, log_fn)

    # Right-tail split (Tukey IQR): after Jenks2 has handled bimodal splits,
    # remove residual right-tail outliers (rare/limited editions, accessory
    # bundles at unusual prices) into a side '<key> #high' cluster so the main
    # cluster is a clean population for anomaly detection. ~1% items move,
    # CV main halves on average. See dry_run in scripts/.
    if not DISABLE_PRICE_SPLIT and not DISABLE_BITAIL:
        groups = _right_tail_split(groups, log_fn, mad_k=TUNE_BITAIL_MADK)

    # Sprint-4: SECOND Jenks2 pass on the bi-tail residual. The MAIN cluster
    # of a heavily-split cluster (e.g. one with #high1 #high2 #low1 #low2)
    # sometimes still hides a residual bimodal distribution that the first
    # Jenks2 pass missed because the unsplit cluster was dominated by tails.
    # With tails extracted, the main is cleaner and a second Jenks2 can
    # catch the residual split. Empirical: ~5-10% extra splits.
    # #10: optionally drop this 2nd pass to test whether it adds signal or
    # just fragments clusters below the deal-detector's MIN_CLUSTER floor.
    if not CONSOLIDATE_PRICE_SPLIT and not DISABLE_PRICE_SPLIT:
        groups = _price_split_jenks2(groups, emb_of, log_fn,
                                     min_gvf=TUNE_JENKS_GVF,
                                     min_center_ratio=TUNE_JENKS_RATIO,
                                     max_emb_cos=TUNE_JENKS_COS)

    # Sprint-7 #3: merge sibling fragments <base>#1/#2/#3 that share the same
    # product (cos≥0.97) and have similar median price (ratio≤1.15). Fixes
    # the iPhone 11 / Switch OLED fragmentation caused by price-feature
    # splitting same-product price tiers within brand-homogeneous partitions.
    groups = _merge_sibling_fragments(groups, emb_of, log_fn,
                                      max_median_ratio=TUNE_SIBLING_RATIO,
                                      min_emb_cos=TUNE_SIBLING_COS)

    # Sprint-7 #7: lex-sig merge — coalesce clusters with shared brand AND
    # shared model number (e.g. 'apple iphone 11' + 'iphone 11' across batches)
    # when text centroids agree (cos≥0.96) and prices compatible (ratio≤1.25).
    groups = _merge_by_lex_signature(groups, emb_of, log_fn,
                                     min_emb_cos=TUNE_LEXSIG_COS,
                                     max_median_ratio=TUNE_LEXSIG_RATIO)

    # Sprint-7 #4: split mappazzoni — large clusters (size≥50) where >50% of
    # titles start with different first-tokens are HDBSCAN aggregates on
    # trailing tokens (xenosaga/kenka mappazzoni found by scout). Split by
    # first informative token.
    groups = _split_blob_by_first_token(groups, log_fn)

    # #2: lexical model-code split for tech (CPU/GPU/RAM) clusters — separates
    # near-homograph SKUs (i5 6600 vs i5 6500) that the text encoder merges.
    if ENABLE_MODEL_TOKEN_SPLIT:
        groups = _split_by_model_token(groups, log_fn)

    # storage-aware model-identity split (all clusters): separate iPhone 12/13,
    # Galaxy S21/S22, i5-6600/6500 that HDBSCAN merged.
    if ENABLE_PRODUCT_SPLIT:
        groups = _split_by_product_id(groups, log_fn)

    # Adaptive cohesion-gated outlier ejection (tight clusters reject intruders).
    if ENABLE_OUTLIER_EJECT:
        groups = _eject_embedding_outliers(groups, emb_of, log_fn,
                                           k=TUNE_EJECT_K,
                                           min_median_cos=TUNE_EJECT_MINMED,
                                           max_keep_cos=TUNE_EJECT_FLOOR)

    # FINAL de-mixer: catch device+accessory/game price mixtures (incl. ones the
    # merges re-created) that the emb-gated Jenks2 can't split (shared brand token).
    if ENABLE_MIXTURE_SPLIT:
        groups = _mixture_split(groups, log_fn,
                                min_gvf=TUNE_MIX_GVF, min_ratio=TUNE_MIX_RATIO)

    # Final mapping + recomputed centroids.
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for k, v in groups.items():
        for it in v:
            mapping[it["id"]] = k
        mean_vec = np.mean([emb_of[it["id"]] for it in v], axis=0)
        cat_counts: dict[str, int] = {}
        for it in v:
            c = (it.get("category") or "").strip()
            if c:
                cat_counts[c] = cat_counts.get(c, 0) + 1
        # 30% quorum (see cluster_partitioned comment): avoid a single mis-
        # categorized listing dictating the label of the whole cluster.
        top_cat = None
        if cat_counts:
            best_cat = max(cat_counts, key=cat_counts.get)
            if cat_counts[best_cat] / max(1, len(v)) >= 0.30:
                top_cat = best_cat
        pr = sorted(float(it["price"]) for it in v
                    if it.get("price") and it["price"] > 0)
        price_med = pr[len(pr) // 2] if pr else None
        rows.append((k, _emb_to_blob(mean_vec.astype("float32")), len(v), now_ts,
                     _display_label(k, top_cat), price_med))
    _persist_centroids(conn, rows, wipe=True)
    log_fn(f"  refined: {len(groups)} clusters, {len(rows)} centroids")
    return mapping


# ─── Image embedding + two-stage refine ─────────────────────────────────────

_clip_model = None
_clip_lock = threading.Lock()


def _get_clip():
    """Lazy-load the CLIP image model (regenerate subprocess only)."""
    global _clip_model
    if _clip_model is None:
        with _clip_lock:
            if _clip_model is None:
                from sentence_transformers import SentenceTransformer
                _clip_model = SentenceTransformer(IMAGE_MODEL_NAME)
    return _clip_model


def embed_item_images(items: list[dict], log_fn=print,
                      max_photos: int = IMAGE_MAX_PHOTOS,
                      max_items: int | None = None) -> dict:
    """For items that have `photo_urls` but no `image_emb` yet, download their
    photos (in RAM), CLIP-embed and MEAN-pool (multi-photo) → {id: float32 vec}.
    Images are never written to disk. Returns only newly-computed vectors.

    `max_items` caps how many are embedded in one call — used by the regenerate
    so it never blocks on a huge backlog (the continuous image worker does the
    bulk); None = no cap (the worker)."""
    if not EMBEDDINGS_AVAILABLE:
        return {}
    import io
    import urllib.request
    try:
        from PIL import Image
    except ImportError:
        log_fn("  ⚠ PIL non disponibile → niente embedding immagini")
        return {}
    targets = [it for it in items if it.get("photo_urls") and not it.get("image_emb")]
    if max_items is not None and len(targets) > max_items:
        log_fn(f"  immagini: {len(targets)} mancanti, ne faccio {max_items} "
               f"(il resto lo fa l'image worker)")
        targets = targets[:max_items]
    if not targets:
        return {}
    log_fn(f"  immagini: {len(targets)} item da vettorizzare (scarico foto)…")

    def dl(u):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=10).read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None

    # Download all photos in PARALLEL (the serial network I/O is the bottleneck;
    # a thread pool gives ~5-8x throughput so the worker can keep up with inflow).
    from concurrent.futures import ThreadPoolExecutor
    jobs = []   # (item_index, url)
    kept = []
    for it in targets:
        urls = [u for u in (it["photo_urls"] or "").split(",") if u][:max_photos]
        if not urls:
            continue
        k = len(kept); kept.append(it["id"])
        for u in urls:
            jobs.append((k, u))
    with ThreadPoolExecutor(max_workers=16) as ex:
        images = list(ex.map(lambda j: dl(j[1]), jobs))
    flat, idx = [], []
    for (k, _u), im in zip(jobs, images):
        if im is not None:
            flat.append(im); idx.append(k)
    if not flat:
        return {}
    ev = _get_clip().encode(flat, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
    acc: dict[int, list] = {}
    for e, k in zip(ev, idx):
        acc.setdefault(k, []).append(e)
    out = {kept[k]: np.mean(es, axis=0).astype("float32") for k, es in acc.items()}
    log_fn(f"  immagini: vettorizzati {len(out)} item")
    return out


def image_refine(mapping: dict[int, str], items: list[dict], img_of: dict,
                 log_fn=print) -> dict[int, str]:
    """Stage 2: split text clusters by IMAGE (multi-photo CLIP) into visual
    sub-products. Runs on clusters where Jenks2 saw a bimodal price signal but
    couldn't act on it — either because the metric was below Jenks2's full
    threshold, or because the title-embedding-cosine blocked the split (e.g.
    everyone titles their listing "Nintendo Switch", both consoles and games).

    Two guards prevent over-splitting:
      1. NEAR-MISS gate (Jenks2-style stats): cluster must have GVF >=
         NEAR_MISS_GVF and ratio >= NEAR_MISS_RATIO — i.e. the price
         distribution shows real bimodality, not flat noise.
      2. IMAGE-COS guard: after HDBSCAN finds visual sub-clusters, their image
         centroids must be sufficiently DIFFERENT (cos < MAX_IMG_COS). If
         HDBSCAN found "sub-clusters" that all look visually similar (e.g.
         same product photographed from 2 angles), reject the split.

    `img_of` = {id: image vector}. Returns the refined mapping (sub-keys
    '<key> #imgN')."""
    if not HDBSCAN_AVAILABLE or not img_of:
        return mapping
    norm_img = {i: v / max(float(np.linalg.norm(v)), 1e-12) for i, v in img_of.items()}
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(mapping[it["id"]], []).append(it)
    H = _get_hdbscan().HDBSCAN
    out = dict(mapping)
    n_split = n_no_signal = n_imgcos_block = n_price_block = n_jaccard_block = 0
    for k, v in groups.items():
        if len(v) < IMAGE_REFINE_MIN:
            continue
        # GATE 1 — Jenks2 near-miss: skip clusters with no bimodal price signal
        # (those are either price-uniform = same product, or already split by
        # Jenks2 upstream — both cases image_refine shouldn't touch).
        pr = [it["price"] for it in v if it.get("price")]
        stats = _jenks2_stats(pr, min_band=3)
        if stats is None:
            n_no_signal += 1; continue
        _idx, gvf, ratio = stats
        if gvf < IMAGE_REFINE_NEAR_MISS_GVF or ratio < IMAGE_REFINE_NEAR_MISS_RATIO:
            n_no_signal += 1; continue

        members = [it for it in v if it["id"] in norm_img]
        if len(members) < IMAGE_REFINE_MIN:
            continue
        X = np.array([norm_img[it["id"]] for it in members])
        labels = H(min_cluster_size=IMAGE_REFINE_MIN_SUB, metric="euclidean",
                   cluster_selection_method="leaf", core_dist_n_jobs=-1).fit_predict(X)
        uniq = sorted({int(l) for l in labels if l != -1})
        if len(uniq) < 2:
            continue

        # GATE 2 — image-cos: validate that the sub-clusters are visually
        # different. Pairwise cosine between sub-cluster image centroids; if
        # any pair is too similar, abort the split.
        centroids = {}
        for l in uniq:
            sub = np.array([norm_img[it["id"]] for it, lab in zip(members, labels) if int(lab) == l])
            c = sub.mean(axis=0); c = c / max(float(np.linalg.norm(c)), 1e-12)
            centroids[l] = c
        max_cos = 0.0
        ls = list(uniq)
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                cos = float(centroids[ls[i]] @ centroids[ls[j]])
                if cos > max_cos:
                    max_cos = cos
        if max_cos >= IMAGE_REFINE_MAX_IMG_COS:
            n_imgcos_block += 1; continue

        # GATE 3 — price-validation: accept the visual split ONLY when the
        # sub-clusters have substantially DIFFERENT median prices. Same product
        # in different colors looks visually different but has the same price →
        # REJECT. Different products (iPhone vs case) → big price gap → ACCEPT.
        # Skipped when sub-clusters have too few priced items to compare.
        if IMAGE_REFINE_MIN_PRICE_RATIO > 1.0:
            import statistics as _st
            sub_medians = []
            for l in uniq:
                sub_prices = [it["price"] for it, lab in zip(members, labels)
                              if int(lab) == l and it.get("price") and it["price"] > 0]
                if len(sub_prices) >= 2:
                    sub_medians.append(_st.median(sub_prices))
            if len(sub_medians) >= 2:
                ratio = max(sub_medians) / min(sub_medians)
                if ratio < IMAGE_REFINE_MIN_PRICE_RATIO:
                    n_price_block += 1; continue

        # GATE 4 (sprint-7 architect #2) — lexical jaccard: sub-clusters with
        # ZERO token overlap between titles are CLIP-noise (same seller / pose
        # / lighting), not real product structure. Reject if no pair shares
        # at least 10% of informative tokens.
        sub_tokens = {}
        for l in uniq:
            toks = set()
            for it, lab in zip(members, labels):
                if int(lab) == l:
                    for t in _normalize(it.get("title", "")).split():
                        if len(t) >= 3 and t not in _NOISE:
                            toks.add(t)
            sub_tokens[l] = toks
        max_jaccard = 0.0
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                a, b = sub_tokens[ls[i]], sub_tokens[ls[j]]
                if not a or not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                if union:
                    j_val = inter / union
                    if j_val > max_jaccard:
                        max_jaccard = j_val
        if max_jaccard < 0.10:
            n_jaccard_block += 1
            continue

        n_split += 1
        for it, l in zip(members, labels):
            if l != -1:
                out[it["id"]] = f"{k} #img{int(l)}"
    if n_split or n_imgcos_block or n_price_block:
        log_fn(f"  image-refine: {n_split} cluster spaccati per immagine, "
               f"{n_imgcos_block} bloccati da img-cos-check, "
               f"{n_price_block} bloccati da price-ratio (stessa popolazione di prezzo), "
               f"{n_jaccard_block} bloccati da jaccard titoli (CLIP-noise), "
               f"{n_no_signal} senza segnale Jenks2 (skipped)")
    return out


def match_cluster_batch(items: list[dict], conn,
                        threshold: float = SIMILARITY_THRESHOLD,
                        ) -> dict[int, tuple[str | None, float]]:
    """
    Lookup-only variant of cluster_titles_batch: for each item return the
    closest existing cluster_key (or None if no centroid passes `threshold`)
    along with the cosine similarity. Does NOT create new clusters or update
    centroids — safe to call against any read-only / production category DB.

    Returns:
        {item_id: (cluster_key | None, similarity)}
    """
    out: dict[int, tuple[str | None, float]] = {}
    if not items or not EMBEDDINGS_AVAILABLE:
        return {it["id"]: (None, 0.0) for it in items}

    centroids: dict[str, "np.ndarray"] = {}
    medians: dict[str, float | None] = {}
    try:
        rows_c = list(conn.execute(
            "SELECT cluster_key, embedding, price_median FROM cluster_centroids"))
    except Exception:
        rows_c = [(r[0], r[1], None) for r in conn.execute(
            "SELECT cluster_key, embedding FROM cluster_centroids")]
    for row in rows_c:
        try:
            centroids[row[0]] = _blob_to_emb(row[1])
            medians[row[0]] = row[2]
        except Exception:
            continue
    if not centroids:
        return {it["id"]: (None, 0.0) for it in items}
    band_index = _band_base_index(medians) if BAND_AWARE_MATCH else {}

    model = _get_model()
    titles = [_embed_text(it) for it in items]
    embs   = model.encode(titles, show_progress_bar=False, convert_to_numpy=True)

    if INCREMENTAL_VECTOR_MATCH:
        ckeys = list(centroids.keys())
        cmat = np.array([centroids[k] for k in ckeys], dtype=np.float32)
        cmat /= np.maximum(np.linalg.norm(cmat, axis=1, keepdims=True), 1e-12)
        en = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        sims_all = en @ cmat.T                       # (n_items, n_centroids)
        js = sims_all.argmax(axis=1)
        for idx, it in enumerate(items):
            j = int(js[idx]); s = float(sims_all[idx, j])
            if s >= threshold:
                key = _band_route(ckeys[j], it.get("price"), band_index)
                out[it["id"]] = (key, s)
            else:
                out[it["id"]] = (None, max(s, 0.0))
        return out

    for it, emb in zip(items, embs):
        best_key, best_sim = None, -1.0
        for ck, c_emb in centroids.items():
            sim = _cosine(emb, c_emb)
            if sim > best_sim:
                best_sim, best_key = sim, ck
        if best_sim >= threshold:
            best_key = _band_route(best_key, it.get("price"), band_index)
            out[it["id"]] = (best_key, best_sim)
        else:
            out[it["id"]] = (None, best_sim if best_sim >= 0 else 0.0)
    return out
