"""Keep only BAMBI frames whose every box was found by OWL in the RGB image itself.

Reads the per-flight annotation and provenance files (small, via HTTP range requests; no video)
for the frames already in an output of build_bambi_detect.py, re-applies frame_boxes() with
min_rgb_detected=1.0, and copies the passing images and labels to a new dataset. Frames with any
box placed only by the frame-wide shift (provenance ``consensus``) are dropped. The same rule is
applied to train, val and test.

Usage:
  python clean_bambi_owl.py --bambi-repo ../../external/bambi-Dataset --config ../configs/diurnal_bambi.yaml \
    --src ../../data/bambi_rgb --manifest ../configs/splits/bambi_rgb_manifest.json \
    --out ../../data/bambi_rgb_owl --out-manifest ../configs/splits/bambi_rgb_owl_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from build_bambi_detect import ZenodoFlight, ZenodoIndex, frame_boxes, read_mot, read_provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bambi-repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="dataset written by build_bambi_detect.py")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = cfg["names"]
    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    index = ZenodoIndex(args.bambi_repo, "rgb")
    for s in ("train", "val", "test"):
        (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)

    def one(split: str, flight: str, frames: list[int]) -> tuple[str, str, list[int], Counter, Counter]:
        src = ZenodoFlight(index.urls(flight))
        try:
            rows = read_mot(src.member(f"{flight}_rgb_gt.txt").decode("utf-8"))
            pb = src.member(f"{flight}_provenance.csv")
            prov = read_provenance(pb.decode("utf-8") if pb else None)
            mb = src.member(f"{flight}_mask_w.png")
            mask = cv2.imdecode(np.frombuffer(mb, np.uint8), cv2.IMREAD_GRAYSCALE) if mb else None
        finally:
            src.close()
        kept, boxes, dropped = [], Counter(), Counter()
        for f in frames:
            b, why = frame_boxes(rows[f], cfg, prov, mask, args.min_visible, 1.0)
            if not b:
                dropped[why or "no boxes"] += 1
                continue
            stem = f"bambi{flight}_{f:08d}"
            shutil.copy2(args.src / "images" / split / f"{stem}.jpg", args.out / "images" / split / f"{stem}.jpg")
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in b]
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            kept.append(f)
            for c, *_ in b:
                boxes[names[c]] += 1
        return split, flight, kept, boxes, dropped

    out_frames = {s: {} for s in ("train", "val", "test")}
    counts = {s: {"images": 0, "boxes": Counter(), "flights_with_frames": 0} for s in out_frames}
    dropped_all = Counter()
    jobs = [(s, fl, fr) for s, d in man["frames"].items() for fl, fr in d.items()]
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            s, fl, kept, boxes, dropped = fu.result()
            dropped_all.update(dropped)
            if kept:
                out_frames[s][fl] = kept
                counts[s]["images"] += len(kept)
                counts[s]["flights_with_frames"] += 1
                counts[s]["boxes"].update(boxes)
            print(f"[{n}/{len(jobs)}] flight {fl} ({s}): kept {len(kept)}", flush=True)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: [{', '.join(names)}]\n", encoding="utf-8")
    res = {k: v for k, v in man.items() if k not in ("frames", "counts", "skipped_frames", "failed_flights")}
    res["frame_rule"] = {**man.get("frame_rule", {}), "min_rgb_detected": 1.0,
                         "note": "subset of the source manifest: every box found by OWL in RGB (owl/smoothed)"}
    res["counts"] = {s: {**v, "boxes": dict(v["boxes"])} for s, v in counts.items()}
    res["dropped_frames"] = dict(dropped_all)
    res["frames"] = out_frames
    args.out_manifest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["counts"]), json.dumps(res["dropped_frames"]))


if __name__ == "__main__":
    main()
