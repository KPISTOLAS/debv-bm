"""Detector-CAM faithfulness on held-out detections (Grad-CAM++ vs Grad-CAM, plus controls).

For each sampled ground-truth animal (up to --n-correct correctly and --n-wrong wrongly
classified by the detector), Y is the detector confidence for the predicted class at the
box on the unmodified image, and the CAM heatmap H is computed on the box crop
(yolo_cam.py, class-score target by default). Three perturbations are scored:

  Explanation map (Chattopadhay et al., WACV 2018, adapted to a box):
      the box region is replaced by crop * H (context outside the box unchanged).
      AD = mean max(0, (Y - O) / Y) * 100,  IC = % of samples with O > Y.
  Deletion (top-k):  the top --top-fraction of H inside the box is filled with the
      crop mean; deletion drop = mean max(0, (Y - O) / Y) * 100.
  Keep-only (top-k): only the top --top-fraction of H inside the box is kept, the rest
      of the image is zeroed; keep-IC = % of samples with O > Y.
  Random controls:   deletion and keep-only repeated with --random-repeats masks of the
      same pixel count inside the box, taken as the top-k of a random --random-grid x
      --random-grid noise map upsampled to the crop (blobs at the CAM resolution; 0 =
      scattered pixels). A faithful map should delete more confidence than a random mask
      of matched area and structure.

Images are read with OpenCV (BGR) and passed to YOLO.predict as BGR arrays, which is the
Ultralytics convention for NumPy input; the CAM gets the RGB crop.

95% intervals: bootstrap over source images (a sample can contribute up to
--max-per-image boxes from one image), --bootstrap resamples.

Requires detector weights + a labelled YOLO split (not in git).
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
    p = argparse.ArgumentParser(description="DEBV-BM detector CAM faithfulness")
    p.add_argument("--weights", type=Path, required=True, help="YOLO .pt (deployed detector)")
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
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--random-grid", type=int, default=20, help="0 = scattered random pixels")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--target", choices=("class", "energy"), default="class")
    p.add_argument("--methods", nargs="+", default=["gradcampp", "gradcam"],
                   choices=["gradcampp", "gradcam", "legacy_pp"])
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
    """Exactly ceil(top_fraction * n) highest pixels of the map (ties broken by order)."""
    if heatmap_roi.size == 0:
        return np.zeros_like(heatmap_roi, dtype=bool)
    hm = normalize_heatmap(heatmap_roi)
    k = max(1, int(np.ceil(top_fraction * hm.size)))
    mask = np.zeros(hm.shape, dtype=bool)
    mask.ravel()[np.argsort(hm.ravel(), kind="stable")[-k:]] = True
    return mask


def random_mask_like(mask: np.ndarray, rng: np.random.Generator, grid: int | None = 20) -> np.ndarray:
    """Random mask with the same pixel count as `mask`.

    grid=None: uniformly scattered pixels. grid=G: top-k of a G x G uniform noise map
    upsampled (bicubic) to the crop, i.e. blobs at the spatial resolution of the CAM layer
    (stride 32 at 640 px -> 20 x 20), so area and blob structure match the salient mask.
    """
    k = int(mask.sum())
    if grid is None:
        out = np.zeros(mask.shape, dtype=bool)
        out.ravel()[rng.choice(mask.size, size=k, replace=False)] = True
        return out
    noise = rng.random((grid, grid)).astype(np.float32)
    smooth = cv2.resize(noise, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
    return salient_mask_roi(smooth, k / mask.size)


def relative_drop(y: float, o: float) -> float:
    if y <= 1e-12:
        return 0.0
    return max(0.0, (y - o) / y * 100.0)


# Backwards-compatible name used by tests and older scripts.
average_drop = relative_drop


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


def apply_mask_in_bbox(image, bbox, salient, mode: str) -> np.ndarray:
    """Channel-order agnostic: works on BGR or RGB arrays alike."""
    x1, y1, x2, y2 = bbox
    out = image.astype(np.float32).copy()
    crop = out[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    if mode == "drop":
        fill = crop.mean(axis=(0, 1))
        crop[salient] = fill
        out[y1:y2, x1:x2] = crop
    elif mode == "keep":
        blank = np.zeros_like(out)
        c = image[y1:y2, x1:x2].astype(np.float32)
        kept = np.zeros_like(c)
        kept[salient] = c[salient]
        blank[y1:y2, x1:x2] = kept
        out = blank
    else:
        raise ValueError(mode)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_explanation_map(image, bbox, heatmap) -> np.ndarray:
    """Chattopadhay explanation map inside the box: crop * H, context unchanged."""
    x1, y1, x2, y2 = bbox
    out = image.astype(np.float32).copy()
    crop = out[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    out[y1:y2, x1:x2] = crop * normalize_heatmap(heatmap)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def confidence_at_bbox(model, image_bgr, target_class, gt_bbox, conf, min_iou) -> float:
    results = model.predict(image_bgr, conf=conf, verbose=False, imgsz=640)
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


def match_pred(model, image_bgr, gt_bbox, conf, min_iou):
    results = model.predict(image_bgr, conf=conf, verbose=False, imgsz=640)
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


METRICS = ("ad", "ic", "del_drop", "del_drop_random", "keep_ic", "keep_ic_random", "del_minus_random")


def eval_method(model, cam, selected, args) -> list[dict]:
    rows = []
    rng = np.random.default_rng(args.seed)
    score = lambda img, inst: confidence_at_bbox(model, img, inst["pred_cls"], inst["gt_bbox"], args.conf, args.iou_match)  # noqa: E731
    for inst in selected:
        bgr = cv2.imread(str(inst["image_path"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_bbox = inst["gt_bbox"]
        y = score(bgr, inst)
        if y <= 1e-6:
            continue
        hm = cam.heatmap_roi(rgb, gt_bbox, target_class=inst["pred_cls"])
        ch, cw = gt_bbox[3] - gt_bbox[1], gt_bbox[2] - gt_bbox[0]
        if hm.shape[:2] != (ch, cw):
            hm = cv2.resize(hm, (cw, ch), interpolation=cv2.INTER_CUBIC)
        salient = salient_mask_roi(hm, args.top_fraction)

        o_exp = score(apply_explanation_map(bgr, gt_bbox, hm), inst)
        o_del = score(apply_mask_in_bbox(bgr, gt_bbox, salient, "drop"), inst)
        o_keep = score(apply_mask_in_bbox(bgr, gt_bbox, salient, "keep"), inst)
        rd, rk = [], []
        for _ in range(args.random_repeats):
            rmask = random_mask_like(salient, rng, args.random_grid or None)
            rd.append(relative_drop(y, score(apply_mask_in_bbox(bgr, gt_bbox, rmask, "drop"), inst)))
            rk.append(float(score(apply_mask_in_bbox(bgr, gt_bbox, rmask, "keep"), inst) > y))
        del_drop = relative_drop(y, o_del)
        del_rand = float(np.mean(rd)) if rd else float("nan")
        rows.append(
            {
                "image": inst["image_path"].name,
                "correct": inst["pred_cls"] == inst["gt_cls"],
                "y": y,
                "ad": relative_drop(y, o_exp),
                "ic": float(o_exp > y),
                "del_drop": del_drop,
                "del_drop_random": del_rand,
                "del_minus_random": del_drop - del_rand,
                "keep_ic": float(o_keep > y),
                "keep_ic_random": float(np.mean(rk)) if rk else float("nan"),
                "iou": heatmap_box_iou(hm, args.top_fraction),
                "o_exp": o_exp,
                "o_del": o_del,
                "o_keep": o_keep,
            }
        )
    return rows


def _pct(v: float, key: str) -> float:
    return 100.0 * v if key in ("ic", "keep_ic", "keep_ic_random") else v


def cluster_bootstrap(rows: list[dict], stat, n_boot: int, seed: int) -> list[float] | None:
    """95% percentile interval of stat(rows) resampling source images with replacement."""
    by_img: dict[str, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    keys = sorted(by_img)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for i in pick for r in by_img[keys[i]]]
        v = stat(sample)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


def mean_of(key):
    def f(rows):
        return _pct(float(np.mean([r[key] for r in rows])), key) if rows else None
    return f


def wrong_minus_correct(key):
    def f(rows):
        ok = [r[key] for r in rows if r["correct"]]
        bad = [r[key] for r in rows if not r["correct"]]
        if not ok or not bad:
            return None
        return _pct(float(np.mean(bad)) - float(np.mean(ok)), key)
    return f


def summarize(rows: list[dict], n_boot: int, seed: int) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "n_images": len({r["image"] for r in rows})}
    for key in METRICS:
        out[key] = round(mean_of(key)(rows), 2)
        out[f"{key}_ci95"] = cluster_bootstrap(rows, mean_of(key), n_boot, seed)
    out["mean_iou_topk_vs_box"] = round(float(np.mean([r["iou"] for r in rows])), 4)
    return out


def paired_method_diff(a_rows, b_rows, key, n_boot, seed):
    """Mean(a - b) of `key` over samples present in both methods, image-clustered CI."""
    b_map = {(r["image"], r["y"]): r for r in b_rows}
    pairs = [{"image": r["image"], "d": r[key] - b_map[(r["image"], r["y"])][key]}
             for r in a_rows if (r["image"], r["y"]) in b_map]
    if not pairs:
        return None
    stat = lambda rows: _pct(float(np.mean([p["d"] for p in rows])), key) if rows else None  # noqa: E731
    return {"n": len(pairs), "diff": round(stat(pairs), 2), "ci95": cluster_bootstrap(pairs, stat, n_boot, seed)}


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_correct, args.n_wrong = 20, 10
        args.bootstrap = min(args.bootstrap, 200)
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
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt_cls, xc, yc, bw, bh = box
        gt_bbox = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        if gt_bbox[2] <= gt_bbox[0] or gt_bbox[3] <= gt_bbox[1]:
            continue
        pred_cls, _ = match_pred(model, bgr, gt_bbox, args.conf, args.iou_match)
        entry = {"image_path": img_path, "gt_cls": gt_cls, "pred_cls": pred_cls, "gt_bbox": gt_bbox}
        if pred_cls >= 0 and pred_cls == gt_cls:
            correct.append(entry)
        elif pred_cls >= 0:
            wrong.append(entry)  # misclassified; missed boxes (pred_cls -1) have no class to explain

    rng.shuffle(correct)
    rng.shuffle(wrong)
    selected = correct[: args.n_correct] + wrong[: args.n_wrong]
    print(f"pool: {len(correct)} correct, {len(wrong)} misclassified; selected {len(selected)}")

    report = {
        "protocol": {
            "weights": str(args.weights),
            "n_correct_target": args.n_correct,
            "n_wrong_target": args.n_wrong,
            "n_correct_available": len(correct),
            "n_wrong_available": len(wrong),
            "top_fraction": args.top_fraction,
            "random_repeats": args.random_repeats,
            "random_grid": args.random_grid,
            "target": args.target,
            "bootstrap": args.bootstrap,
            "ci": "image-clustered percentile bootstrap",
            "predict_input": "BGR uint8 (Ultralytics NumPy convention)",
        },
        "methods": {},
    }
    per_method = {}
    for method in args.methods:
        cam = YoloCam(model, method=method, target=args.target)
        rows = eval_method(model, cam, selected, args)
        cam.close()
        per_method[method] = rows
        ok = [r for r in rows if r["correct"]]
        bad = [r for r in rows if not r["correct"]]
        report["methods"][method] = {
            "overall": summarize(rows, args.bootstrap, args.seed),
            "correct": summarize(ok, args.bootstrap, args.seed),
            "wrong": summarize(bad, args.bootstrap, args.seed),
            "wrong_minus_correct": {
                key: {
                    "diff": None if (v := wrong_minus_correct(key)(rows)) is None else round(v, 2),
                    "ci95": cluster_bootstrap(rows, wrong_minus_correct(key), args.bootstrap, args.seed),
                }
                for key in ("ad", "del_drop", "del_minus_random")
            },
            "samples": [{k: (bool(v) if k == "correct" else v) for k, v in r.items()} for r in rows],
        }
        o = report["methods"][method]["overall"]
        print(method, {k: o.get(k) for k in ("n", "ad", "ic", "del_drop", "del_drop_random", "keep_ic", "keep_ic_random")})

    names = list(per_method)
    if len(names) >= 2:
        a, b = names[0], names[1]
        report[f"{a}_minus_{b}"] = {
            key: paired_method_diff(per_method[a], per_method[b], key, args.bootstrap, args.seed)
            for key in ("ad", "del_drop")
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
