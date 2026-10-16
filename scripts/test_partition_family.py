"""Task #2: does partitioning branded items by cat|brand|FAMILY prevent
device+accessory mixtures at the source? Fresh partition of the subset with
PARTITION_FAMILY_FOR_BRANDED=True, refined with the best config, benchmarked
against the cached best (raw partition + same refine config).

Judge on: HARM/mix (should drop if the hypothesis holds), coverage (fragmenting
same-product across partitions would lower it), OBJ overall.
"""
import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")
os.environ.setdefault("VINTED_EMBED_MODEL", "intfloat/multilingual-e5-large")
import core.clustering as cl
import ab_lab as L
from bench import benchmark
from ab_search import KNOBS, DEFAULTS

CACHE = ROOT / "data" / "ab_cache"
RAW_FAM = CACHE / "raw_sub_family.json"
BEST = json.load(open(CACHE / "best_model.json"))["config"]


def setcfg():
    for k in KNOBS:
        setattr(cl, k, DEFAULTS[k])
    for k, v in BEST.items():
        setattr(cl, k, v)


def main():
    items, emb, img = L.load_subset()
    print(f"subset {len(items)} items; best config: {json.dumps(BEST)}", flush=True)

    # reference: cached raw partition + best refine
    raw0 = L.raw_partition(items, emb)
    setcfg()
    m0 = L.refine(items, raw0, emb, img)
    b0 = benchmark(items, m0, emb)
    print(f"{'REF (cat|brand partition)':34} OBJ={b0['OBJ']:.3f} ABQ={b0['ABQ']:.3f} "
          f"HARM={b0['HARM']:.3f} MASK={b0['MASK']:.3f} cov={b0['coverage']:.1f}% "
          f"mix={b0['mix_pct']:.1f}%", flush=True)

    # fresh partition with family key for branded items
    if RAW_FAM.exists():
        raw1 = {int(k): v for k, v in json.load(open(RAW_FAM)).items()}
    else:
        conn, p = L._temp_conn()
        cl.PARALLEL_SHARD_LARGE = True
        cl.PARTITION_FAMILY_FOR_BRANDED = True
        t0 = time.time()
        raw1 = cl.cluster_partitioned(items, conn=conn, min_cluster_size=2,
                                      log_fn=lambda *a, **k: None, emb_cache=emb)
        cl.PARALLEL_SHARD_LARGE = False
        cl.PARTITION_FAMILY_FOR_BRANDED = False
        conn.close(); os.unlink(p)
        json.dump({str(k): v for k, v in raw1.items()}, open(RAW_FAM, "w"))
        print(f"[partition brand|family] {len(set(raw1.values()))} clusters "
              f"in {time.time()-t0:.0f}s → cached", flush=True)
    setcfg()
    m1 = L.refine(items, raw1, emb, img)
    b1 = benchmark(items, m1, emb)
    print(f"{'brand|family partition':34} OBJ={b1['OBJ']:.3f} ABQ={b1['ABQ']:.3f} "
          f"HARM={b1['HARM']:.3f} MASK={b1['MASK']:.3f} cov={b1['coverage']:.1f}% "
          f"mix={b1['mix_pct']:.1f}%", flush=True)
    verdict = "WINNER" if b1["OBJ"] > b0["OBJ"] else "rejected"
    print(f"\nbrand|family partition: {verdict} "
          f"(ΔOBJ={b1['OBJ']-b0['OBJ']:+.3f}, Δcov={b1['coverage']-b0['coverage']:+.1f}pp, "
          f"Δmix={b1['mix_pct']-b0['mix_pct']:+.1f}pp)", flush=True)


if __name__ == "__main__":
    main()
