# Vinted Monitor — Pipeline di Clustering: Briefing per Code Review

## Obiettivo del progetto

Sistema di **rilevamento occasioni** su Vinted (marketplace second-hand).
Pipeline: scrape catalogo → clustering item per prodotto → futuro detector di
anomalie prezzo (item molto sotto mediana del cluster = potenziale occasione).

Il **clustering è funzionale**, non d'esposizione: i cluster main devono essere
"popolazioni pulite" di prodotti identici a prezzi coesi, perché diventano la
**baseline statistica** per anomaly detection.

## KPI principali

| Metrica | Valore attuale | Target |
|---|---|---|
| SCORE = cohesion × price-coh | **83.9** | massimo |
| QSCORE = SCORE × brand_pure × (1-blob_share) | **~82** | massimo |
| ANOM_READY = % item in cluster IDEAL | **5%** | massimo |
| N_BLOB (cluster spuri large) | **0** | 0 |
| Cohesion | **0.966** | ~0.97 (max teorico) |
| Price-coh | **87%** | massimo |
| Brand-purity | **98%** | massimo |

**Cluster IDEAL** = size≥10, CV prezzi<0.30, brand-purity≥95%.

## Pipeline ad alto livello (4 stages)

```
Stage 1 · Embedding text (e5-large multilingue, 1024-d L2-normalized)
   ├─ _normalize: lowercase, regex strip, subword split (iphone15→iphone 15),
   │  memory canonicalization (128go→128 gb), price/€ strip
   ├─ _embed_text: prepend brand, strip _NOISE (548 token multilingue),
   │  _TRANSLATE (15 famiglie: jeu/giochi/games→game, casque/cuffie→headphones),
   │  _BRAND_ALIASES (Redmi/Poco→Xiaomi)
   ├─ Cache DB: text_emb + text_emb_version (v6)
   └─ Continuous text_worker pre-embedda in background

Stage 2 · Clustering (cluster_partitioned + _hdbscan_compute)
   ├─ Partition by `category|brand`: 174k items in 16k partitions
   │  - "heavy" (size>=50): 233 partizioni processate in ProcessPoolExecutor(5)
   │  - "light" (size<50): main thread
   │  - Partition con __nobrand__ ha 46k items (collo)
   ├─ HDBSCAN density-based, metric=euclidean, eom, min_cluster_size=2
   ├─ PRICE FEATURE: concat log(price) z-scored come dim extra (1025-d)
   │  - λ = 0.60 (calibrato via sweep)
   │  - CV gate [0.30, 0.80]: skip partizioni con CV troppo bassa/alta
   └─ Centroidi persistiti in cluster_centroids (1024-d text-only per compat)

Stage 3 · Refine (refine_clusters)
   ├─ rescue_singletons: assorbe singleton in cluster vicini
   │  - Gate HI cosine 0.94, LO 0.88 + match lessicale brand+modello
   ├─ _price_split_jenks2: spacca cluster bimodali (2 popolazioni prezzo)
   │  - GVF>=0.55, ratio>=2.0
   │  - High-conf bypass: GVF>=0.65, ratio>=4.0
   └─ _right_tail_split: bi-tail MAD iterativo (#high + #low, max 3 iter ciascuna)
      - mad_k=2.5 default
      - extreme_mad_k=1.5 if MAD/median≥0.40 OR max/min≥20× (multi-pop trigger)

Stage 4 · Image refine (image_refine)
   ├─ Gate Jenks2 near-miss (gvf>=0.40, ratio>=1.5) — solo cluster sotto-bimodali
   ├─ HDBSCAN su CLIP L-14 vectors → sub-cluster visivi `#imgN`
   ├─ img-cos guard 1.0 (disabilitato — accetta anche sub-cluster simili)
   └─ price-ratio guard 1.0 (disabilitato)

Eval inline al regen → SCORE, QSCORE, ANOM_READY, N_BLOB, N_GIANT
```

## File chiave

| Path | Cosa |
|---|---|
| `core/clustering.py` | TUTTA la pipeline (~1400 righe) — preprocessing, embedding, HDBSCAN, refine, image, eval |
| `core/category_db.py` | SQLite schema, _migrate, item ops, indici composti |
| `core/text_worker.py` | Worker continuo embedding text (e5-large) |
| `core/image_worker.py` | Worker continuo embedding immagini (CLIP L-14, multi-photo) |
| `core/sold_check_worker.py` + `core/sold_check.py` | Worker drip HTTP item-page |
| `core/category_monitor.py` | Scrape catalog (no embedding) |
| `core/vinted_client.py` | HTTP client con curl_cffi + rate limiter |
| `scripts/regenerate_clusters.py` | Orchestratore regen (4 stages + cleanup + workers pause) |
| `scripts/eval_db.py` | Eval canonico standalone |
| `webgui/services.py`, `webgui/app.py` | GUI NiceGUI |
| `core/category_shared.db` | DB sqlite (~2 GB, 184k items attivi) |

## Componenti chiave & configurazione attuale

### Preprocessing (`core/clustering.py` linee ~107-470)
- `_NOISE`: **548 token** filler multilingua (IT/EN/FR/DE/ES/NL/PT)
  - Condizione: nuovo/perfetto/ottime/neuf/sehr/comme/usado…
  - Colori (TUTTE le lingue): nero/black/schwarz/noir/negro/preto/zwart…
  - Aggettivi: digitale/funzionante/elektrische/elétrico…
  - Preposizioni/connettori: de/in/della/von/du/des/non/sous…
  - Edizioni/versioni: edition/edizione/generation/version
  - Packaging: boîte/scatola/caja/blister/ovp
  - Generici tecnici: ordinateur/computer/windows/android/auto
- `_TRANSLATE`: 15 famiglie multilingua → canonical EN
  - GAME (jeu/giochi/games/spiel/juego)
  - CONTROLLER (manette/controller/mando)
  - CAMERA (kamera/fotocamera/cámara/appareil)
  - HEADPHONES, PHONE, CASE, CHARGER, LAPTOP, TABLET, MOUSE
  - KEYBOARD, SPEAKER, SCREEN, WATCH, PRINTER
- `_BRAND_ALIASES`: Redmi/Poco/Roidmi → Xiaomi
- `_STRIP_PATTERNS`: regex per "nuovo con cartellino", "come nuovo", "X€" finale, punteggiatura
- Subword: `iphone15` → `iphone 15` (regex `\d{2,}` per non spaccare i7/m3/ps5)
- Memory: `128go/128gb/128tb` → `128 gb` (canonical)
- **Item con titolo 100% noise → eliminato dal DB** (es. "État neuf jamais utilisé")

### HDBSCAN + Price feature (`core/clustering.py` linee ~800-960)
```python
PRICE_FEATURE_LAMBDA = 0.60       # peso dim prezzo concatenata
PRICE_FEATURE_CV_MIN = 0.30       # CV partition < skip (omogenea)
PRICE_FEATURE_CV_MAX = 0.80       # CV partition > skip (rischio blob)
```

Pre-calcolo CV partizione: z-scored log(price) concatenato all'embedding 1024-d
→ 1025-d → L2 re-normalize → HDBSCAN. Centroidi persistiti restano 1024-d.

Multiprocessing partizioni: 233 heavy partition in pool(5 workers, core_dist_n_jobs=1
ciascuno), 16k light in main thread. Mega-partizione `__nobrand__` (46k) è il collo.

### Refine: Jenks2 + bi-tail (`core/clustering.py` linee ~1100-1300)
- Jenks2: `high_conf_gvf=0.65`, `high_conf_ratio=4.0` (aggressive bypass)
- Bi-tail MAD iterativo: `mad_k=2.5`, `extreme_mad_k=1.5`,
  trigger ADAPTIVE: `MAD/median≥0.40 OR max/min≥20×`
- Max 3 iter per direzione (high/low), max_total_tail_frac=0.40

### Image refine (`core/clustering.py` ~1300-1400)
- Gate Jenks2 near-miss: `IMAGE_REFINE_NEAR_MISS_GVF=0.40`, `RATIO=1.5`
- `IMAGE_REFINE_MAX_IMG_COS=1.0` (disabilitato)
- `IMAGE_REFINE_MIN_PRICE_RATIO=1.0` (disabilitato)
- `IMAGE_REFINE_MIN=6`, `MIN_SUB=2`
- CLIP-L-14, multi-photo mean (max 4 photos), download parallelo (16 thread)

### Eval (`core/clustering.py` `eval_metrics()`)
- Basic: cohesion, price_coh, brand_pure, coverage, SCORE
- Extended: n_ideal, ideal_items, anom_ready, n_blob, blob_items, n_giant, qscore

## Storia delle ottimizzazioni (cosa abbiamo già fatto)

```
57.3 baseline → 65.5 Jenks2 → 70.5 noise+img-refine → 75.9 Jenks2 aggressive
→ 76.0 img-cos off → 80.2 + right-tail → 82.5 MAD iter → 82.7 bi-tail
→ 83.0 adaptive MAD/median → 83.4 → 83.9 ← STATO ATTUALE
```

## Cosa NON funziona / cose già scartate

- **Brand-token removal nelle partizioni** (#26): testato, peggiora SCORE
- **Description Vinted nel testo**: abortito, sono solo "ottime condizioni"
- **Cross-encoder validation**: paper-grade, sforzo alto, non ancora fatto
- **Contrastive fine-tune e5**: pending, sforzo molto alto
- **Sharding `__nobrand__`** per accelerare: pending

## Stato corrente DB

- ~184k item attivi
- ~43k cluster (di cui ~700 IDEAL, ~7k size≥8)
- 178k embedding text_emb (cache v5/v6)
- 122k image_emb (CLIP)
- ZERO mappazzoni size≥30+CV>0.6 (con NOISE v6 + λ=0.60 + CV gate)

## Obiettivi della code review

Per ogni componente:
1. **Cosa fa esattamente** (capire il codice attuale)
2. **Problemi/edge case** non gestiti
3. **Miglioramenti possibili** con stima ROI:
   - Alto ROI (+1 SCORE / +2% ANOM_READY)
   - Medio
   - Basso (cosmetico, refactor)
4. **Rischi** di ogni proposta

Non basta "questo si può migliorare" — voglio **suggerimenti concreti**
con esempi di codice o algoritmo proposto.
