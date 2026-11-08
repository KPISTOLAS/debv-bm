"""Turn the review tool's export into a hand-verified BAMBI test split.

Only images marked verified are kept. Train and val are linked from the source dataset unchanged,
so the verified test split is scored by the same data.yaml layout. A summary of the changes
(boxes added, removed, resized) is written next to the labels for the paper's methods section.

Usage:
  python import_bambi_corrections.py --export bambi_test_corrections.json \
    --src ../../data/bambi_rgb_owl --out ../../data/bambi_rgb_owl_verified
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    i = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return i / (a[3] * a[4] + b[3] * b[4] - i + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.export.read_text(encoding="utf-8"))
    for s in ("train", "val"):
        for sub in ("images", "labels"):
            dst = args.out / sub / s
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(args.src / sub / s, dst)
    for sub in ("images", "labels"):
        dst = args.out / sub / "test"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    stats = {"images_total": len(d["items"]), "images_verified": 0, "boxes_original": 0, "boxes_final": 0,
             "boxes_unchanged_iou_0.9": 0, "boxes_adjusted": 0, "boxes_removed": 0, "boxes_added": 0,
             "notes": {}}
    for it in d["items"]:
        if not it["verified"]:
            continue
        stats["images_verified"] += 1
        orig, new = it["original"], it["boxes"]
        stats["boxes_original"] += len(orig)
        stats["boxes_final"] += len(new)
        used = set()
        for o in orig:  # greedy match original -> final boxes
            best, bj = 0.0, -1
            for j, n in enumerate(new):
                if j not in used and (v := iou(o, n)) > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.1:
                used.add(bj)
                stats["boxes_unchanged_iou_0.9" if best >= 0.9 else "boxes_adjusted"] += 1
            else:
                stats["boxes_removed"] += 1
        stats["boxes_added"] += len(new) - len(used)
        if it.get("notes"):
            stats["notes"][it["stem"]] = it["notes"]
        shutil.copy2(args.src / "images" / "test" / f"{it['stem']}.jpg", args.out / "images" / "test")
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in new]
        (args.out / "labels" / "test" / f"{it['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                           encoding="utf-8")
    names = d["names"]
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: [{', '.join(names)}]\n", encoding="utf-8")
    (args.out / "verification_summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "notes"}, indent=1))


if __name__ == "__main__":
    main()
