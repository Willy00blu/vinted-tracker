"""Show / apply the best clustering model found by ab_search.py.

The backup lives in data/ab_cache/best_model.json (auto-saved on every new ABQ
record). This script prints the winning config as the diff from the production
defaults, and with --write patches those module-level defaults in
core/clustering.py in place (a .bak copy is made first).

  python3.13 scripts/apply_model.py            # show the winning config
  python3.13 scripts/apply_model.py --write     # apply it to core/clustering.py
"""
import sys, json, re, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "data" / "ab_cache" / "best_model.json"
SRC = ROOT / "core" / "clustering.py"


def main():
    if not BEST.exists():
        print("no best_model.json yet — run scripts/ab_search.py first"); return
    d = json.load(open(BEST))
    full = d["full_config"]
    diff = d.get("config", {})
    print(f"BEST MODEL  ABQ={d['ABQ']:.4f}  cov={d['cov']:.1f}%  uni={d['unimodal']:.3f} "
          f"mix={d['mix']:.1f}%  (saved {d.get('saved_at','?')})")
    print(f"label: {d['label']}")
    if not diff:
        print("→ winner == production defaults; nothing to change."); return
    print("\nchanges vs production defaults:")
    for k, v in diff.items():
        print(f"  {k} = {v!r}")
    if "--write" not in sys.argv:
        print("\n(dry-run) re-run with --write to patch core/clustering.py")
        return
    bak = SRC.with_suffix(f".py.bak.{int(time.time())}")
    shutil.copy2(SRC, bak)
    text = SRC.read_text()
    for k, v in full.items():
        # replace the first top-level assignment `NAME = <value>` (with optional inline comment)
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)([^\n#]+)(\s*#.*)?$", re.M)
        if pat.search(text):
            text = pat.sub(lambda m: f"{m.group(1)}{v!r}{m.group(3) or ''}", text, count=1)
    SRC.write_text(text)
    print(f"\n✓ patched {SRC} (backup: {bak.name})")


if __name__ == "__main__":
    main()
