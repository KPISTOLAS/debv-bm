"""Tree detector -> BioCLIP cascade on the Marche test photographs (plant row of Table 6).

Marche ships each labelled crown as a masked crop of its source photograph. The crop is
located in the photograph by masked template matching (at 1/4 scale), which gives the
ground-truth crown box. The plant YOLO detector runs on the full photograph at the deployed
operating point (conf 0.25, NMS IoU 0.5, imgsz 640); detections are matched one-to-one to
crown boxes at IoU >= 0.5. Reference is the BioCLIP head on the masked crowns (as in Table 2);
Cascade is the same head on the rectangular detector crops. Only some crowns per photograph
are labelled, so detector precision cannot be scored and only recall-based figures are given.

Usage:
  python eval_plant_cascade.py --archive Image_Dataset.zip \
      --manifest ../configs/splits/marche_split_manifest.json --detector detector_plants_yolo11l.pt \
      --config ../configs/plants_marche.yaml --bioclip bioclip_marche.pt --out plant_cascade.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_prompts import prompt_for_class
from eval_cascade import crop_rgb, greedy_match, iou_matrix, wilson
from train_bioclip import load_open_clip

CROWN = re.compile(r"Segmented_Labelled_Images/mascherargb_(immagine\d+)_oggetto_\d+classe-([a-z_]+)\.jpg$")
SCALE = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM plant detector -> BioCLIP cascade (Marche)")
    p.add_argument("--archive", type=Path, required=True, help="Image_Dataset.zip from Zenodo 14988733")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--detector", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bioclip", type=Path, required=True)
    p.add_argument("--pretrained", default="hf-hub:imageomics/bioclip")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def read(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)


def locate(photo: np.ndarray, crown: np.ndarray) -> tuple[list[float], float]:
    """Box of a masked crown crop inside its photograph, and the mean abs. error of the fit."""
    small = cv2.resize(photo, None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)
    tpl = cv2.resize(crown, None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = (cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) > 15).astype(np.float32)
    res = cv2.matchTemplate(small, tpl, cv2.TM_SQDIFF, mask=mask)
    _, _, (x0, y0), _ = cv2.minMaxLoc(res)
    h, w = crown.shape[:2]
    ph, pw = photo.shape[:2]
    full_mask = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY) > 15
    ys, xs = np.nonzero(full_mask)
    ys, xs = ys[::16], xs[::16]  # subsample mask pixels for the fine search
    best, err = (x0 * SCALE, y0 * SCALE), 255.0
    for dy in range(-SCALE, SCALE + 1):  # fine search at full resolution around the coarse match
        for dx in range(-SCALE, SCALE + 1):
            x, y = x0 * SCALE + dx, y0 * SCALE + dy
            if x < 0 or y < 0 or x + w > pw or y + h > ph:
                continue
            e = float(np.abs(photo[y + ys, x + xs].astype(float) - crown[ys, xs].astype(float)).mean())
            if e < err:
                best, err = (x, y), e
    x, y = best
    return [float(x), float(y), float(x + w), float(y + h)], err


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = list(cfg["names"])
    photos = set(json.loads(args.manifest.read_text(encoding="utf-8"))["photos"][args.split])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip, preprocess, tokenizer = load_open_clip("ViT-B-16", args.pretrained, device)
    clip.load_state_dict(torch.load(args.bioclip, map_location=device)["state_dict"])
    clip.eval()
    with torch.no_grad():
        text = F.normalize(clip.encode_text(tokenizer([prompt_for_class(n, str(cfg.get("modality", "rgb"))) for n in names]).to(device)), dim=-1)

    def classify(crops: list[Image.Image]) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(crops), 32):
                x = torch.stack([preprocess(c) for c in crops[i:i + 32]]).to(device)
                preds.append((F.normalize(clip.encode_image(x), dim=-1) @ text.T).argmax(-1).cpu().numpy())
        return np.concatenate(preds) if preds else np.zeros(0, dtype=int)

    detector = YOLO(str(args.detector))
    zf = zipfile.ZipFile(args.archive)
    crowns: dict[str, list[tuple[str, int]]] = {}
    for name in zf.namelist():
        if (m := CROWN.search(name)) and m.group(1) in photos:
            crowns.setdefault(m.group(1), []).append((name, names.index(m.group(2))))

    n_gt = n_det = 0
    fit_err, per_photo = [], {}
    gt_y, gt_crops, m_y, m_det_crops = [], [], [], []
    for photo_id in sorted(crowns):
        photo = read(zf, f"Image_Dataset/TrainDataset/Original_Images/{photo_id}.jpg")
        h, w = photo.shape[:2]
        boxes = []
        for name, cls in crowns[photo_id]:
            crown = read(zf, name)
            box, err = locate(photo, crown)
            fit_err.append(err)
            boxes.append(box)
            gt_y.append(cls)
            gt_crops.append(Image.fromarray(cv2.cvtColor(crown, cv2.COLOR_BGR2RGB)))
        n_gt += len(boxes)
        res = detector.predict(photo, conf=args.conf, iou=args.nms_iou, imgsz=args.imgsz, verbose=False)[0]
        det = res.boxes.xyxy.cpu().numpy()
        n_det += len(det)
        matches = greedy_match(iou_matrix(np.array(boxes), det), args.match_iou) if len(det) else []
        per_photo[photo_id] = {"crowns": len(boxes), "detections": int(len(det)), "matched": len(matches)}
        offset = len(gt_y) - len(boxes)
        for gi, di in matches:
            x1, y1, x2, y2 = det[di]
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))
            m_y.append((offset + gi, gt_y[offset + gi]))
            m_det_crops.append(crop_rgb(photo, (x1, y1, x2, y2)))

    gt_y = np.array(gt_y)
    ref_all = classify(gt_crops) == gt_y
    idx = np.array([i for i, _ in m_y], dtype=int)
    y_m = np.array([c for _, c in m_y], dtype=int)
    ref = int(ref_all[idx].sum()) if len(idx) else 0
    cas = int((classify(m_det_crops) == y_m).sum())
    n_m = len(idx)
    out = {
        "detector": str(args.detector), "bioclip": str(args.bioclip), "conf": args.conf, "imgsz": args.imgsz,
        "photos": len(crowns), "n_gt": n_gt, "n_detections": n_det, "n_matched": n_m,
        "template_fit_mae_max": round(max(fit_err), 2), "template_fit_mae_mean": round(float(np.mean(fit_err)), 2),
        "loc_recall": round(100 * n_m / n_gt, 2), "loc_recall_ci95": wilson(n_m, n_gt),
        "reference_all_gt": round(100 * float(ref_all.mean()), 2),
        "reference": round(100 * ref / n_m, 2) if n_m else None, "reference_ci95": wilson(ref, n_m),
        "cascade": round(100 * cas / n_m, 2) if n_m else None, "cascade_ci95": wilson(cas, n_m),
        "drop_pp": round(100 * (ref - cas) / n_m, 2) if n_m else None,
        "e2e": round(100 * cas / n_gt, 2), "e2e_ci95": wilson(cas, n_gt),
        "per_photo": per_photo,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_photo"}))


if __name__ == "__main__":
    main()
