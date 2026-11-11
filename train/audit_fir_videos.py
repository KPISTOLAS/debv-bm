"""Audit the FIR split for temporal leakage between frames of the same source recording.

FIR file names encode the source recording and a frame index, e.g.
'-FIR-2023-7-_MP4-3631_jpg.rf.<hash>.jpg' -> recording '-FIR-2023-7-_MP4', frame 3631.
For every test frame the script reports the gap (in frame indices) to the nearest training
or validation frame of the same recording, and tabulates classes per recording and split.

Usage:
  python audit_fir_videos.py --root "FIR dataset root with train/ valid/ test/" --out fir_video_audit.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"^(.*?)[-_]?(\d+)_(jpg|png|jpeg)\.rf\.")
SPLITS = ("train", "valid", "test")


def long_path(p: Path) -> str:
    s = str(p.resolve())
    return "\\\\?\\" + s if os.name == "nt" and not s.startswith("\\\\?\\") else s


def recording(name: str) -> tuple[str, int | None]:
    m = PATTERN.match(name)
    return (m.group(1), int(m.group(2))) if m else ("other (no frame index)", None)


def main() -> None:
    ap = argparse.ArgumentParser(description="FIR temporal-leakage audit")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--names", nargs="+", default=["Bird", "Boar", "Cat", "Deer", "Flying-squirrel", "Fox", "Raccoon"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frames = {s: collections.defaultdict(list) for s in SPLITS}
    classes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for split in SPLITS:
        lab_dir = long_path(args.root / split / "labels")
        for fname in os.listdir(lab_dir):
            rec, idx = recording(fname[:-4])
            frames[split][rec].append(idx)
            with open(os.path.join(lab_dir, fname), encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        classes[rec][split][args.names[int(line.split()[0])]] += 1

    gaps = []
    for rec, idxs in frames["test"].items():
        ref = np.array(sorted(i for s in ("train", "valid") for i in frames[s].get(rec, []) if i is not None))
        for i in idxs:
            if i is None or not len(ref):
                gaps.append(None)
            else:
                gaps.append(int(np.min(np.abs(ref - i))))
    g = np.array([x for x in gaps if x is not None])
    recs = sorted(classes, key=lambda r: -sum(len(frames[s].get(r, [])) for s in SPLITS))
    table = {r: {"images": {s: len(frames[s].get(r, [])) for s in SPLITS},
                 "instances": {s: dict(classes[r][s]) for s in SPLITS}} for r in recs}
    out = {
        "n_test": len(gaps),
        "n_test_with_frame_index": int(len(g)),
        "nearest_trainval_gap_median": float(np.median(g)) if len(g) else None,
        "n_gap_le_1": int((g <= 1).sum()), "n_gap_le_5": int((g <= 5).sum()), "n_gap_le_20": int((g <= 20).sum()),
        "recordings_in_all_splits": [r for r in recs if all(frames[s].get(r) for s in SPLITS)],
        "per_recording": table,
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{'recording':26s} {'tr/va/te':>14s}  " + " ".join(f"{n[:6]:>6s}" for n in args.names))
    for r in recs:
        im = table[r]["images"]
        tot = collections.Counter()
        for s in SPLITS:
            tot.update(classes[r][s])
        print(f"{r[:26]:26s} {im['train']:5d}/{im['valid']:4d}/{im['test']:3d}  " + " ".join(f"{tot[n]:6d}" for n in args.names))
    print({k: out[k] for k in ("n_test", "n_test_with_frame_index", "nearest_trainval_gap_median", "n_gap_le_1", "n_gap_le_5", "n_gap_le_20")})


if __name__ == "__main__":
    main()
