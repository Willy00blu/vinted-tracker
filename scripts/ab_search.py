"""Config search for the BEST clustering "model" (= pipeline configuration) for
price-anomaly-baseline quality (ABQ, see bench.py).

Uses the instant lab: cached fp16 embeddings + cached sharded partition, so each
config = one fast refine pass (~8s). Two stages:
  1. discrete grid over the high-impact toggles (price-split / merge / etc.)
  2. greedy coordinate ascent over the numeric refine thresholds around the
     best stage-1 config.

Every evaluation is appended to data/ab_cache/leaderboard.jsonl. Whenever ABQ
beats the running best, the config + metrics are checkpointed to
data/ab_cache/best_model.json  ← the "backup del modello per i nuovi record".

Run:  python3.13 scripts/ab_search.py
Apply the winner to production later with:  scripts/apply_model.py
"""
import sys, os, json, time, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
import ab_lab as L
from bench import benchmark

CACHE = ROOT / "data" / "ab_cache"
LEADER = CACHE / "leaderboard.jsonl"
BEST = CACHE / "best_model.json"

# Knobs the search may set, with their production defaults (captured live).
KNOBS = ["DISABLE_BITAIL", "STABLE_MERGE_KEY", "LOGSPACE_BITAIL",
         "ENABLE_MODEL_TOKEN_SPLIT", "ENABLE_PRODUCT_SPLIT", "ENABLE_MIXTURE_SPLIT",
         "CONSOLIDATE_PRICE_SPLIT", "DISABLE_PRICE_SPLIT",
         "TUNE_JENKS_GVF", "TUNE_JENKS_RATIO", "TUNE_JENKS_COS", "TUNE_BITAIL_MADK",
         "TUNE_SIBLING_RATIO", "TUNE_SIBLING_COS", "TUNE_LEXSIG_COS",
         "TUNE_LEXSIG_RATIO", "RESCUE_GATE_HI", "RESCUE_GATE_LO",
         "TUNE_MIX_GVF", "TUNE_MIX_RATIO"]
DEFAULTS = {k: getattr(cl, k) for k in KNOBS}

_state = {"items": None, "emb": None, "img": None, "raw": None,
          "best": -1.0, "n": 0}


def setup():
    items, emb, img = L.load_subset()
    raw = L.raw_partition(items, emb)
    _state.update(items=items, emb=emb, img=img, raw=raw)
    if BEST.exists():
        _state["best"] = json.load(open(BEST)).get("OBJ", -1.0)
    print(f"subset={len(items)} items, raw clusters={len(set(raw.values()))}, "
          f"running best OBJ={_state['best']:.4f}\n", flush=True)


def apply(config):
    for k in KNOBS:
        setattr(cl, k, config.get(k, DEFAULTS[k]))


def evaluate(config, label):
    apply(config)
    t0 = time.time()
    m = L.refine(_state["items"], _state["raw"], _state["emb"], _state["img"])
    b = benchmark(_state["items"], m, _state["emb"])
    dt = time.time() - t0
    _state["n"] += 1
    rec = {"label": label, "OBJ": b["OBJ"], "ABQ": b["ABQ"], "HARM": b["HARM"],
           "cov": b["coverage"], "bpure": b["bpure"], "specpure": b["specpure"],
           "unimodal": b["unimodal"], "sep": b["sep"], "mix": b["mix_pct"],
           "nclu": b["n_clusters"],
           "config": {k: config[k] for k in config if config[k] != DEFAULTS[k]},
           "t": round(dt, 1)}
    with open(LEADER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    star = ""
    if b["OBJ"] > _state["best"]:
        _state["best"] = b["OBJ"]
        json.dump({**rec, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "full_config": {k: config.get(k, DEFAULTS[k]) for k in KNOBS}},
                  open(BEST, "w"), indent=2)
        star = "  ★ NEW BEST → best_model.json"
    print(f"[{_state['n']:3d}] {label:30} OBJ={b['OBJ']:.4f} ABQ={b['ABQ']:.4f} "
          f"cov={b['coverage']:4.1f}% uni={b['unimodal']:.3f} mix={b['mix_pct']:4.1f}% "
          f"({dt:.0f}s){star}", flush=True)
    return b["OBJ"]


def stage1():
    print("=== STAGE 1: discrete toggles ===", flush=True)
    bools = ["DISABLE_BITAIL", "STABLE_MERGE_KEY", "ENABLE_MIXTURE_SPLIT"]
    best_cfg, best_abq = dict(DEFAULTS), -1.0
    seen = set()
    for combo in itertools.product([False, True], repeat=len(bools)):
        cfg = dict(DEFAULTS)
        for b, val in zip(bools, combo):
            cfg[b] = val
        key = tuple(cfg[b] for b in bools)
        if key in seen:
            continue
        seen.add(key)
        label = "+".join(b.replace("TUNE_", "").replace("ENABLE_", "")
                         for b, v in zip(bools, combo) if v) or "baseline"
        abq = evaluate(cfg, label)
        if abq > best_abq:
            best_abq, best_cfg = abq, dict(cfg)
    # also probe the LOGSPACE bitail variant when bitail is kept on
    if not best_cfg["DISABLE_BITAIL"]:
        cfg = dict(best_cfg); cfg["LOGSPACE_BITAIL"] = True
        abq = evaluate(cfg, "best+logspace-bitail")
        if abq > best_abq:
            best_abq, best_cfg = abq, dict(cfg)
    return best_cfg, best_abq


def stage2(base_cfg):
    print("\n=== STAGE 2: greedy numeric coordinate ascent ===", flush=True)
    grid = {
        "TUNE_JENKS_GVF":     [0.55, 0.60, 0.70, 0.80],
        "TUNE_JENKS_COS":     [0.92, 0.95, 0.97],
        "TUNE_JENKS_RATIO":   [1.7, 2.0, 2.5],
        "TUNE_SIBLING_RATIO": [1.15, 1.30, 1.50],
        "TUNE_SIBLING_COS":   [0.95, 0.97, 0.985],
        "TUNE_LEXSIG_COS":    [0.96, 0.98],
        "TUNE_LEXSIG_RATIO":  [1.10, 1.25, 1.40],
        "RESCUE_GATE_HI":     [0.92, 0.94],
        "RESCUE_GATE_LO":     [0.86, 0.88, 0.90],
        "TUNE_MIX_RATIO":     [2.5, 3.0, 4.0],
        "TUNE_MIX_GVF":       [0.55, 0.60, 0.70],
    }
    if base_cfg["DISABLE_BITAIL"]:
        grid.pop("TUNE_BITAIL_MADK", None)
    else:
        grid["TUNE_BITAIL_MADK"] = [2.2, 2.5, 3.0, 3.5]
    cfg = dict(base_cfg)
    best_abq = evaluate(cfg, "stage2-start")
    for _pass in range(2):
        improved = False
        for param, vals in grid.items():
            cur = cfg[param]
            local_best_v, local_best_abq = cur, best_abq
            for v in vals:
                if v == cur:
                    continue
                trial = dict(cfg); trial[param] = v
                abq = evaluate(trial, f"{param.replace('TUNE_','')}={v}")
                if abq > local_best_abq:
                    local_best_abq, local_best_v = abq, v
            if local_best_v != cur:
                cfg[param] = local_best_v; best_abq = local_best_abq; improved = True
        if not improved:
            break
    return cfg, best_abq


def main():
    setup()
    LEADER.unlink(missing_ok=True)
    best_cfg, _ = stage1()
    best_cfg, best_abq = stage2(best_cfg)
    apply(dict(DEFAULTS))
    print(f"\n=== SEARCH DONE: {_state['n']} configs, best ABQ={_state['best']:.4f} ===")
    print(f"best_model.json:")
    print(json.dumps(json.load(open(BEST))["config"], indent=2))


if __name__ == "__main__":
    main()
