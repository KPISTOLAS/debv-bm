"""Spatial–temporal near-duplicate filter for Tier 2 JSON detections."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM spatial–temporal deduplication")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--output", type=Path, default=None, help="Default: results-dir/dedup.json")
    return p.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_ts(rec: dict) -> datetime | None:
    raw = rec.get("timestamp_utc") or rec.get("timestamp")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def flatten(results_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "dedup.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("timestamp_utc")
        for det in data.get("detections", []):
            items.append(
                {
                    "file": path.name,
                    "image": data.get("image"),
                    "timestamp_utc": ts,
                    "xyxy": det.get("xyxy"),
                    "label": det.get("species") or det.get("detector_class"),
                    "confidence": det.get("confidence"),
                }
            )
    return items


def main() -> None:
    args = parse_args()
    items = flatten(args.results_dir)
    keep = [True] * len(items)
    for i, a in enumerate(items):
        if not keep[i] or not a.get("xyxy"):
            continue
        ta = parse_ts(a)
        for j in range(i + 1, len(items)):
            if not keep[j] or not items[j].get("xyxy"):
                continue
            tb = parse_ts(items[j])
            if ta and tb and abs((ta - tb).total_seconds()) > args.seconds:
                continue
            if iou(a["xyxy"], items[j]["xyxy"]) >= args.iou:
                keep[j] = False
    kept = [x for x, k in zip(items, keep) if k]
    dropped = len(items) - len(kept)
    out = args.output or (args.results_dir / "dedup.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_in": len(items), "n_kept": len(kept), "n_dropped": dropped, "records": kept}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"dedup wrote {out} kept={len(kept)} dropped={dropped}")


if __name__ == "__main__":
    main()
