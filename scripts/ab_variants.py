"""Run all refine-stage A/B variants on the cached raw partition (fast).
Each variant flips ONE module flag in core.clustering and re-runs refine+image.
Baseline = all flags off (must match the standalone baseline run).
"""
import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
import core.clustering as cl
import ab_harness as H


def reset_flags():
    cl.ENABLE_MODEL_TOKEN_SPLIT = False
    cl.STABLE_MERGE_KEY = False
    cl.LOGSPACE_BITAIL = False
    cl.CONSOLIDATE_PRICE_SPLIT = False


def main():
    items, emb, img = H.load()
    raw = H.raw_partition(items, emb)           # cached
    print(f"loaded {len(items)} items; raw clusters={len(set(raw.values()))}\n")

    variants = [
        ("BASELINE", {}),
        ("#2 model-token split", {"ENABLE_MODEL_TOKEN_SPLIT": True}),
        ("#8 stable merge-key", {"STABLE_MERGE_KEY": True}),
        ("#6 bi-tail log-space", {"LOGSPACE_BITAIL": True}),
        ("#10 drop 2nd Jenks2", {"CONSOLIDATE_PRICE_SPLIT": True}),
        ("ALL refine combined", {"ENABLE_MODEL_TOKEN_SPLIT": True,
                                 "STABLE_MERGE_KEY": True,
                                 "LOGSPACE_BITAIL": True}),
    ]
    base_map = None
    for name, flags in variants:
        reset_flags()
        for f, val in flags.items():
            setattr(cl, f, val)
        t0 = time.time()
        m = H.refine(items, raw, emb, img, do_image=True)
        H.evaluate(items, m, emb, name)
        print(f"{'':30} ({time.time()-t0:.0f}s)\n")
        if name == "BASELINE":
            base_map = m
            json.dump({str(k): v for k, v in m.items()},
                      open("/tmp/ab_final_base.json", "w"))
            print("  [dumped baseline final mapping -> /tmp/ab_final_base.json]\n")
    reset_flags()


if __name__ == "__main__":
    main()
