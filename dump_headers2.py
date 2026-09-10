"""Print the exact channel labels of every include=1 recording in the manifest.

Run from the Repo1 root:
    python dump_headers.py --config configs/windows_chb.json
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/windows_chb.json")
ap.add_argument("--manifest", default=None)
a = ap.parse_args()

cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
root = Path(cfg["root"])
manifest = Path(a.manifest or cfg["manifest"])

with manifest.open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["include"] == "1"]

import mne

want = [c.upper() for c in cfg["channels"]]
aliases = {k.upper(): v.upper() for k, v in cfg.get("channel_aliases", {}).items()}
layouts = defaultdict(list)

for r in rows:
    p = root / r["path"]
    raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
    names = list(raw.ch_names)
    raw.close()
    layouts[tuple(names)].append((r["subject"], r["path"]))

print(f"{len(rows)} selected recordings, {len(layouts)} distinct channel layouts\n")

for i, (names, recs) in enumerate(layouts.items(), 1):
    subs = sorted({s for s, _ in recs})
    raw_have = [n.upper() for n in names]
    have = [aliases.get(n, n) for n in raw_have]          # apply channel_aliases
    missing = [c for c in want if c not in have]
    extra = [n for n, m in zip(names, have) if m not in want]
    renamed = [f"{n} -> {aliases[n.upper()]}" for n in names if n.upper() in aliases]
    dupes = sorted({c for c in want if have.count(c) > 1})
    print(f"--- layout {i}: {len(recs)} recording(s), subjects {', '.join(subs)}")
    print(f"    channels ({len(names)}): {', '.join(names)}")
    print(f"    aliased            : {renamed if renamed else 'none'}")
    print(f"    MISSING after alias: {missing if missing else 'none'}")
    print(f"    AMBIGUOUS duplicate: {dupes if dupes else 'none'}")
    print(f"    not used by config : {extra if extra else 'none'}")
    print()
