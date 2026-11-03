"""Regenerate paper AD/IC faithfulness (detector Grad-CAM++ vs Grad-CAM).

Protocol (paper explainability section; Chattopadhyay et al., WACV 2018):
  N=150 held-out detections (100 correct, 50 wrong)
  salient = top 20% activations *inside the GT box*
  AD  = mean max(0, (Y - O) / Y) * 100   (delete salient, measure drop)
  IC  = fraction where salient-only crop raises confidence vs full image
  IoU = top-20% mask vs the GT box (at crop resolution this is the keep-rate)

Requires detector weights + a labelled YOLO split (not in git).
Published 27.91 / 0.00 / 0.20 need the archived fine-tune, not a random YOLO.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM detector AD/IC faithfulness")
    p.add_argument("--weights", type=Path, required=True, help="YOLOv11 .pt (paper fine-tune)")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results") / "faithfulness.json")
    p.add_argument("--n-correct", type=int, default=100)
    p.add_argument("--n-wrong", type=int, default=50)
    p.add_argument("--top-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--iou-match", type=float, default=0.1)
    p.add_argument("--max-per-image", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="20/10 sample for a smoke run")
    return p.parse_args()


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    hm = np.clip(heatmap.astype(np.float32), 0.0, None)
    if hm.size == 0:
        return hm
    lo, hi = float(hm.min()), float(hm.max())
    if hi - lo < 1e-8:
        return np.zeros_like(hm)
    return (hm - lo) / (hi - lo)


def salient_mask_roi(heatmap_roi: np.ndarray, top_fraction: float) -> np.ndarray:
    if heatmap_roi.size == 0:
        return np.zeros_like(heatmap_roi, dtype=bool)
    hm = normalize_heatmap(heatmap_roi)
    thresh = float(np.percentile(hm.ravel(), 100.0 * (1.0 - top_fraction)))
    mask = hm >= thresh
    if not mask.any():
        k = max(1, int(np.ceil(top_fraction * hm.size)))
        mask = np.zeros_like(hm, dtype=bool)
        mask.ravel()[np.argsort(hm.ravel())[-k:]] = True
    return mask


def average_drop(y: float, o: float) -> float:
    if y <= 1e-12:
        return 0.0
    return max(0.0, (y - o) / y * 100.0)


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (aa + bb - inter))


def heatmap_box_iou(heatmap_roi: np.ndarray, top_fraction: float) -> float:
    salient = salient_mask_roi(heatmap_roi, top_fraction)
    gt = np.ones_like(salient, dtype=bool)
    inter = np.logical_and(salient, gt).sum()
    union = np.logical_or(salient, gt).sum()
    return float(inter / union) if union else 0.0


def yolo_to_xyxy(xc, yc, bw, bh, w, h):
    x1 = int(xc * w - bw * w / 2)
    y1 = int(yc * h - bh * h / 2)
    x2 = int(xc * w + bw * w / 2)
    y2 = int(yc * h + bh * h / 2)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def load_yolo_labels(path: Path) -> list[tuple]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(float(parts[0])), *map(float, parts[1:5])))
    return boxes


def apply_mask_in_bbox(image_rgb, bbox, salient, mode: str) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    out = image_rgb.astype(np.float32).copy()
    crop = out[y1:y2, x1:x2]
    if crop.size == 0:
        return image_rgb
    if mode == "drop":
        fill = crop.mean(axis=(0, 1))
        crop[salient] = fill
        out[y1:y2, x1:x2] = crop
    elif mode == "keep":
        blank = np.zeros_like(out)
        c = image_rgb[y1:y2, x1:x2].astype(np.float32)
        kept = np.zeros_like(c)
        kept[salient] = c[salient]
        blank[y1:y2, x1:x2] = kept
        out = blank
    return np.clip(out, 0, 255).astype(np.uint8)


def welch_t(a: list[float], b: list[float]) -> dict:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) < 2 or len(y) < 2:
        return {"t": None, "note": "need >=2 samples per group"}
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    denom = (vx / nx + vy / ny) ** 0.5
    t = float((x.mean() - y.mean()) / denom) if denom else 0.0
    return {
        "t": t,
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "n_a": nx,
        "n_b": ny,
        "note": "Welch t on AD (correct vs wrong). p-value omitted (no scipy pin).",
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "average_drop_pct": round(float(np.mean([r["ad"] for r in rows])), 2),
        "increase_in_confidence_pct": round(100.0 * float(np.mean([r["ic"] for r in rows])), 2),
        "mean_iou_top20_vs_box": round(float(np.mean([r["iou"] for r in rows])), 4),
    }


def confidence_at_bbox(model, image_rgb, target_class, gt_bbox, conf, min_iou) -> float:
    results = model.predict(image_rgb, conf=conf, verbose=False, imgsz=640)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0
    boxes = results[0].boxes
    best = 0.0
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != target_class:
            continue
        xyxy = boxes.xyxy[i].cpu().numpy()
        pred = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        if iou_xyxy(pred, gt_bbox) >= min_iou:
            best = max(best, float(boxes.conf[i].item()))
    return best


def match_pred(model, image_rgb, gt_bbox, conf, min_iou):
    results = model.predict(image_rgb, conf=conf, verbose=False, imgsz=640)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return -1, 0.0
    boxes = results[0].boxes
    best_iou, best_cls, best_conf = 0.0, -1, 0.0
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        pred = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        v = iou_xyxy(pred, gt_bbox)
        if v > best_iou:
            best_iou = v
            best_cls = int(boxes.cls[i].item())
            best_conf = float(boxes.conf[i].item())
    if best_iou < min_iou:
        return -1, 0.0
    return best_cls, best_conf


def eval_method(model, cam, selected, args) -> list[dict]:
    rows = []
    for inst in selected:
        img = cv2.imread(str(inst["image_path"]))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_bbox = inst["gt_bbox"]
        pred_cls = inst["pred_cls"]
        y = confidence_at_bbox(model, rgb, pred_cls, gt_bbox, args.conf, args.iou_match)
        if y <= 1e-6:
            continue
        hm = cam.heatmap_roi(rgb, gt_bbox)
        ch, cw = gt_bbox[3] - gt_bbox[1], gt_bbox[2] - gt_bbox[0]
        if hm.shape[:2] != (ch, cw):
            hm = cv2.resize(hm, (cw, ch), interpolation=cv2.INTER_CUBIC)
        salient = salient_mask_roi(hm, args.top_fraction)
        occ = apply_mask_in_bbox(rgb, gt_bbox, salient, "drop")
        keep = apply_mask_in_bbox(rgb, gt_bbox, salient, "keep")
        o = confidence_at_bbox(model, occ, pred_cls, gt_bbox, args.conf, args.iou_match)
        c = confidence_at_bbox(model, keep, pred_cls, gt_bbox, args.conf, args.iou_match)
        rows.append(
            {
                "image": inst["image_path"].name,
                "correct": pred_cls == inst["gt_cls"],
                "ad": average_drop(y, o),
                "ic": bool(c >= y),
                "iou": heatmap_box_iou(hm, args.top_fraction),
                "y": y,
                "o": o,
                "c": c,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_correct, args.n_wrong = 20, 10
    from ultralytics import YOLO

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from yolo_cam import YoloCam

    if not args.images.is_dir() or not args.labels.is_dir():
        raise SystemExit(f"Missing images/labels: {args.images} {args.labels}")

    model = YOLO(str(args.weights))
    rng = random.Random(args.seed)
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng.shuffle(files)
    pool = []
    for img_path in files:
        labels = load_yolo_labels(args.labels / f"{img_path.stem}.txt")
        rng.shuffle(labels)
        for box in labels[: args.max_per_image]:
            pool.append((img_path, box))

    correct, wrong = [], []
    for img_path, box in pool:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt_cls, xc, yc, bw, bh = box
        gt_bbox = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        if gt_bbox[2] <= gt_bbox[0] or gt_bbox[3] <= gt_bbox[1]:
            continue
        pred_cls, _ = match_pred(model, rgb, gt_bbox, args.conf, args.iou_match)
        entry = {"image_path": img_path, "gt_cls": gt_cls, "pred_cls": pred_cls, "gt_bbox": gt_bbox}
        if pred_cls >= 0 and pred_cls == gt_cls:
            correct.append(entry)
        else:
            wrong.append(entry)

    rng.shuffle(correct)
    rng.shuffle(wrong)
    selected = correct[: args.n_correct] + wrong[: args.n_wrong]
    if len(selected) < args.n_correct + args.n_wrong:
        print(
            f"Warning: only {len(correct)} correct and {len(wrong)} wrong in pool; "
            f"using {len(selected)} (paper used 100/50)"
        )

    report = {
        "protocol": {
            "n_target": args.n_correct + args.n_wrong,
            "n_correct_target": args.n_correct,
            "n_wrong_target": args.n_wrong,
            "top_fraction": args.top_fraction,
            "note": (
                "Paper: AD 27.91% overall (20.35% correct, 43.03% wrong), "
                "IC 0.00%, mean IoU 0.20, Grad-CAM++ vs Grad-CAM 27.9 vs 32.1. "
                "Those figures need the archived detector checkpoint."
            ),
        },
        "methods": {},
    }
    for method in ("plusplus", "gradcam"):
        cam = YoloCam(model, method=method)
        rows = eval_method(model, cam, selected, args)
        ok = [r for r in rows if r["correct"]]
        bad = [r for r in rows if not r["correct"]]
        report["methods"][method] = {
            "overall": summarize(rows),
            "correct": summarize(ok),
            "wrong": summarize(bad),
            "welch_ad_correct_vs_wrong": welch_t([r["ad"] for r in ok], [r["ad"] for r in bad]),
            "samples": [{k: v for k, v in r.items() if k in ("image", "correct", "ad", "ic", "iou")} for r in rows],
        }
        print(method, report["methods"][method]["overall"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
