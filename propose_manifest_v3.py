"""Propose a reviewed CHB-MIT manifest for Repo1.

This script does NOT decide the science. It reads the inventory produced by
`eegstudy.cli inventory`, checks which recordings have an events table and how
many annotated seizure intervals each one contains, and writes a PROPOSED
manifest that you must read and approve before using.

It never writes local/manifest.csv directly. It writes
local/manifest.proposed.csv, prints a summary, and stops. Copying that file
over local/manifest.csv is your explicit confirmation that:

  * annotation coverage for every included recording is complete, and
  * the accepted seizure counts match the CHB-MIT summary files.

Usage (from the Repo1 root, inside the project venv):

    .\\.venv\\Scripts\\python.exe propose_manifest.py --config configs/windows.json \\
        --inventory local/manifest.csv --patients 6 --out local/manifest.proposed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def rows_csv(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def seizure_spans(events_path, cfg):
    """[(onset, offset), ...] for accepted seizure rows."""
    rows = rows_csv(events_path, "\t")
    col = cfg["seizure_column"]
    vals = set(cfg["seizure_values"])
    out = []
    for r in rows:
        if col not in r:
            raise KeyError(
                f"{events_path}: configured seizure_column '{col}' is absent. "
                f"Columns present: {sorted(r)}"
            )
        if r[col] in vals:
            a = float(r["onset"]); d = float(r["duration"])
            if d > 0:
                out.append((a, a + d))
    return sorted(out)


def best_interval(spans, duration, span_len):
    """Aligned [start, stop) of length <= span_len covering the most seizure time.

    For a recording no longer than span_len this is simply the whole file.
    For a longer one it is the sub-interval holding the annotated seizures --
    declared enrichment, not a performance-driven choice.
    """
    if duration <= span_len:
        return 0.0, float(int(duration // 4) * 4)
    if not spans:
        return 0.0, float(int(span_len // 4) * 4)

    def covered(start):
        stop = start + span_len
        return sum(max(0.0, min(b, stop) - max(a, start)) for a, b in spans)

    cands = {0.0, duration - span_len}
    for a, b in spans:
        for c in (a - 60.0, a, b - span_len, (a + b) / 2 - span_len / 2):
            cands.add(min(max(c, 0.0), duration - span_len))
    start = max(sorted(cands), key=covered)
    start = float(int(start // 4) * 4)
    start = min(start, float(int((duration - span_len) // 4) * 4))
    return start, start + float(int(span_len // 4) * 4)


def seizure_count(events_path, cfg):
    """Number of accepted seizure intervals in one events TSV."""
    rows = rows_csv(events_path, "\t")
    col = cfg["seizure_column"]
    vals = set(cfg["seizure_values"])
    n = 0
    for r in rows:
        if col not in r:
            raise KeyError(
                f"{events_path}: configured seizure_column '{col}' is absent. "
                f"Columns present: {sorted(r)}"
            )
        if r[col] in vals:
            n += 1
    return n


SUBNUM = re.compile(r"(?:sub|chb)-?0*(\d+)$", re.I)


def canonical_group(bids_id):
    """sub-07 -> chb07. Leaves anything unrecognised untouched.

    chb21 is deliberately NOT merged here: the package's own patient_group()
    does that merge and is covered by a unit test, so we hand it a chb-style
    name and let the tested code path own the decision.
    """
    m = SUBNUM.match(str(bids_id).strip())
    if not m:
        return bids_id
    return "chb%02d" % int(m.group(1))


def duration_seconds(path):
    """Recording duration without loading samples."""
    import mne

    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    suffix = Path(path).suffix.lower()
    if suffix not in readers:
        raise ValueError(f"Unsupported format for duration probe: {path}")
    raw = readers[suffix](path, preload=False, verbose="ERROR")
    try:
        return raw.n_times / float(raw.info["sfreq"]), list(raw.ch_names)
    finally:
        raw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/windows.json")
    ap.add_argument("--inventory", default="local/manifest.csv")
    ap.add_argument("--out", default="local/manifest.proposed.csv")
    ap.add_argument(
        "--patients",
        type=int,
        default=12,
        help="Number of independent patient groups. The splitter requires at "
        "least 2x the fold count, so five folds need 10 or more.",
    )
    ap.add_argument(
        "--per-patient",
        type=int,
        default=2,
        help="Seizure-containing recordings per patient.",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=3600.0,
        help="Cap the selected interval per recording. prepare() refuses any "
        "interval longer than 7200 s, and shorter intervals keep the cache small.",
    )
    ap.add_argument(
        "--canonical-groups",
        action="store_true",
        help="Rewrite BIDS sub-NN identifiers to chbNN in the GROUP column, so "
        "the tested chb01/chb21 same-patient merge applies. Only use this once "
        "you have confirmed sub-NN corresponds to chbNN in your conversion.",
    )
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["root"])
    rows = rows_csv(args.inventory)
    if not rows:
        sys.exit("Inventory is empty. Run `eegstudy.cli inventory` first.")

    budget = int(cfg.get("max_source_bytes", 4_500_000_000))

    by_group = defaultdict(list)
    skipped_no_events = 0
    for r in rows:
        if not r.get("events"):
            skipped_no_events += 1
            continue
        by_group[r["group"]].append(r)

    if not by_group:
        sys.exit(
            "No recording in the inventory has a sibling *_events.tsv. Check the "
            "BIDS conversion before continuing; annotations cannot be inferred."
        )

    print(f"Groups with annotations: {len(by_group)}  "
          f"(recordings without an events table: {skipped_no_events})")

    # Count accepted seizures per recording.
    enriched = defaultdict(list)
    for group, recs in sorted(by_group.items()):
        for r in recs:
            try:
                spans = seizure_spans(safe_join(root, r["events"]), cfg)
            except FileNotFoundError:
                continue
            if spans:
                r["_spans"] = spans
                r["_seizures"] = len(spans)
                enriched[group].append(r)

    ranked = sorted(enriched.items(), key=lambda kv: -len(kv[1]))
    if len(ranked) < args.patients:
        print(
            f"WARNING: only {len(ranked)} independent groups have annotated "
            f"seizures; requested {args.patients}. The splitter needs at least "
            f"2x the fold count (10 groups for 5 folds)."
        )

    chosen, total_bytes, channels_seen = [], 0, None
    for group, recs in ranked[: args.patients]:
        recs = sorted(recs, key=lambda r: -r["_seizures"])[: args.per_patient]
        for r in recs:
            p = safe_join(root, r["path"])
            size = int(r.get("bytes") or p.stat().st_size)
            if total_bytes + size > budget:
                print(f"Budget reached; stopping before {r['path']}")
                break
            secs, names = duration_seconds(p)
            if channels_seen is None:
                channels_seen = names
            start_s, stop_s = best_interval(r["_spans"], secs, args.max_seconds)
            inside = [(a, b) for a, b in r["_spans"] if a >= start_s and b <= stop_s]
            if not inside:
                print(f"  skip {r['path']}: no complete seizure inside the "
                      f"{args.max_seconds:.0f}s retained interval")
                continue
            sz_secs = sum(b - a for a, b in inside)
            if stop_s - start_s < 4:
                continue
            out_group = canonical_group(group) if args.canonical_groups else group
            total_bytes += size
            print(f"  {r['subject']}  {Path(r['path']).name}  "
                  f"[{int(start_s)}, {int(stop_s)}) s  "
                  f"{len(inside)} seizure(s), {sz_secs:.0f}s ictal")
            chosen.append(
                dict(
                    include=1,
                    path=r["path"],
                    subject=r["subject"],
                    group=out_group,
                    start_s=f"{int(start_s)}",
                    stop_s=f"{int(stop_s)}",
                    events=r["events"],
                    annotation_complete=1,
                    task=r.get("task", "") or "szMonitoring",
                    bytes=size,
                )
            )

    if not chosen:
        sys.exit("Nothing selected. Widen --patients or check the seizure column.")

    fields = list(rows[0].keys())
    for row in chosen:
        for k in fields:
            row.setdefault(k, "")
    write_csv(args.out, [{k: r.get(k, "") for k in fields} for r in chosen], fields)

    groups = sorted({r["group"] for r in chosen})
    n_ch = len(cfg["channels"])
    est = sum(int(float(r["stop_s"]) / 4) * n_ch * 4 * cfg["fs"] * 4 for r in chosen)
    cache_budget = int(cfg.get("max_cache_bytes", 1_000_000_000))
    print()
    print(f"Proposed {len(chosen)} recordings from {len(groups)} independent groups.")
    print(f"Groups: {', '.join(groups)}")
    if args.canonical_groups:
        pairs = sorted({(r["subject"], r["group"]) for r in chosen})
        print("Group rewrite applied (subject -> group): "
              + ", ".join(f"{a}->{b}" for a, b in pairs))
        print("  NOTE: chb21 merges into chb01 inside prepare(); both are one patient.")
    print(f"Selected source bytes: {total_bytes/1e9:.2f} GB of {budget/1e9:.2f} GB budget")
    print(f"Estimated cache:       {est/1e9:.2f} GB of {cache_budget/1e9:.2f} GB budget"
          + ("   *** OVER BUDGET: lower --max-seconds ***" if est > cache_budget else ""))
    print(f"Written to: {args.out}")
    print()
    if channels_seen is not None:
        want = [c.upper() for c in cfg["channels"]]
        have = [c.upper() for c in channels_seen]
        missing = [c for c in want if c not in have]
        print(f"Header channels in the first selected file ({len(channels_seen)}):")
        print("  " + ", ".join(channels_seen))
        if missing:
            print(f"MISSING configured channels: {missing}")
            print("  -> add channel_aliases in the config, or edit 'channels'.")
        else:
            print("All configured channels are present in that header.")
    print()
    print("NEXT: open the proposed file, verify every row against the CHB-MIT")
    print("summary for that patient, then copy it over local/manifest.csv.")


def safe_join(root, rel):
    root = Path(root).resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Manifest paths must stay inside the dataset root")
    return p


if __name__ == "__main__":
    main()
