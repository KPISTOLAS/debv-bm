"""Qualitative detection, BioCLIP, and explanation panels (results figures) with the deployed models.

Day: a given WAID test image. Night: the first FIR test image (sorted by name) whose label
file holds exactly one Boar whose box covers at least 2% of the image and does not touch the
border; the rule uses only the ground truth, so the example is not chosen by model output.
Detection panels show detector confidence only (the pipeline ignores the detector class);
BioCLIP panels show the head's label and softmax probability per crop. The explanation panel
shows the crop, the class-agnostic detector map of the faithfulness evaluation, and BioCLIP
Grad-CAM++ for the predicted class.

Usage:
  python make_example_figures.py --day-image WAID/images/test/dataset_2897.jpg --fir-test FIR/test \
      --waid-detector waid_yolo11l.pt --fir-detector fir_yolo11l.pt \
      --waid-bioclip waid_best.pt --fir-bioclip fir_best.pt --out figures/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tier2"))
from clip_prompts import prompt_for_class
from train_bioclip import load_open_clip
from yolo_cam import YoloCam

BOX = (40, 40, 230)  # BGR red
BOAR = 1  # FIR class id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM example figures")
    p.add_argument("--day-image", type=Path, required=True)
    p.add_argument("--fir-test", type=Path, required=True, help="FIR test folder with images/ and labels/")
    p.add_argument("--waid-detector", type=Path, required=True)
    p.add_argument("--fir-detector", type=Path, required=True)
    p.add_argument("--waid-bioclip", type=Path, required=True)
    p.add_argument("--fir-bioclip", type=Path, required=True)
    p.add_argument("--configs", type=Path, default=HERE.parent / "configs")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def pick_night_image(fir_test: Path) -> Path:
    for img in sorted((fir_test / "images").iterdir()):
        lines = [l.split() for l in (fir_test / "labels" / f"{img.stem}.txt").read_text().splitlines() if l.strip()]
        if len(lines) != 1 or int(lines[0][0]) != BOAR or len(lines[0]) != 5:
            continue
        cx, cy, w, h = map(float, lines[0][1:])
        inside = cx - w / 2 > 0.01 and cy - h / 2 > 0.01 and cx + w / 2 < 0.99 and cy + h / 2 < 0.99
        if inside and w * h >= 0.02:
            return img
    raise SystemExit("no single-Boar FIR test image")


def draw(img: np.ndarray, boxes: np.ndarray, labels: list[str]) -> np.ndarray:
    out = img.copy()
    scale = max(out.shape[:2]) / 640
    fs, th = 0.45 * scale, max(1, int(round(1.2 * scale)))
    for (x1, y1, x2, y2), text in zip(boxes.astype(int), labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX, max(1, int(round(2 * scale))))
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y1 - 3 if y1 - tht - base - 3 > 0 else y2 + tht + 3
        cv2.rectangle(out, (x1, ty - tht - base), (x1 + tw + 4, ty + base // 2), BOX, -1)
        cv2.putText(out, text, (x1 + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out


def load_head(cfg: Path, bioclip: Path):
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    names, modality = list(c["names"]), str(c.get("modality", "rgb"))
    clip, preprocess, tokenizer = load_open_clip("ViT-B-16", "hf-hub:imageomics/bioclip", "cuda")
    clip.load_state_dict(torch.load(bioclip, map_location="cuda")["state_dict"])
    clip.eval()
    with torch.no_grad():
        text = F.normalize(clip.encode_text(tokenizer([prompt_for_class(n, modality) for n in names]).cuda()), dim=-1)
    return names, clip, preprocess, text


def panels(img_path: Path, detector: Path, cfg: Path, bioclip: Path, stem: str, out: Path) -> None:
    img = cv2.imread(str(img_path))
    det = YOLO(str(detector)).predict(img, conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
    boxes, confs = det.boxes.xyxy.cpu().numpy(), det.boxes.conf.cpu().numpy()
    cv2.imwrite(str(out / f"{stem}_YOLO.png"), draw(img, boxes, [f"{c:.2f}" for c in confs]))

    names, clip, preprocess, text = load_head(cfg, bioclip)
    labels = []
    with torch.no_grad():
        for x1, y1, x2, y2 in boxes.astype(int):
            crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            p = (100.0 * F.normalize(clip.encode_image(preprocess(crop).unsqueeze(0).cuda()), dim=-1) @ text.T).softmax(-1)[0]
            k = int(p.argmax())
            labels.append(f"{names[k].replace('-', ' ').lower()} {float(p[k]):.2f}")
    cv2.imwrite(str(out / f"{stem}_Bioclip.png"), draw(img, boxes, labels))
    print(stem, img_path.name, len(boxes), "detections:", labels)


def explanation_panel(img_path: Path, detector: Path, cfg: Path, bioclip: Path, out_path: Path) -> None:
    from infer_pipeline import _ClipImageLogits, reshape_transform_vit, vit_target_layer
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img = cv2.imread(str(img_path))
    yolo = YOLO(str(detector))
    det = yolo.predict(img, conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
    x1, y1, x2, y2 = det.boxes.xyxy[int(det.boxes.conf.argmax())].cpu().numpy().astype(int)
    crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

    names, clip, preprocess, text = load_head(cfg, bioclip)
    wrapper = _ClipImageLogits(clip, text).module.cuda().eval()
    x = preprocess(crop).unsqueeze(0).cuda()
    with torch.no_grad():
        k = int(wrapper(x).argmax())
    with GradCAMPlusPlus(model=wrapper, target_layers=[vit_target_layer(wrapper.clip)], reshape_transform=reshape_transform_vit) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(k)])[0]
    side = x.shape[-1]
    base = np.asarray(crop.resize((side, side), Image.BICUBIC)).astype(np.float32) / 255.0
    overlay = show_cam_on_image(base, heat, use_rgb=True)

    det_heat = YoloCam(yolo, method="plusplus").heatmap_roi(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (x1, y1, x2, y2))
    det_overlay = show_cam_on_image(base, cv2.resize(det_heat, (side, side), interpolation=cv2.INTER_CUBIC), use_rgb=True)

    row = np.concatenate([np.uint8(base * 255), det_overlay, overlay], axis=1)
    big = cv2.resize(row, (6 * side, 2 * side), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_path), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print("explanation", img_path.name, "class", names[k])


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    night = pick_night_image(args.fir_test)
    shutil.copy2(night, args.out / "Night_Animal_Original.jpg")
    diurnal, nocturnal = args.configs / "diurnal.yaml", args.configs / "nocturnal.yaml"
    panels(args.day_image, args.waid_detector, diurnal, args.waid_bioclip, "Day_Animal", args.out)
    panels(night, args.fir_detector, nocturnal, args.fir_bioclip, "Night_Animal", args.out)
    explanation_panel(night, args.fir_detector, nocturnal, args.fir_bioclip, args.out / "grandcam_image_clear.png")


if __name__ == "__main__":
    main()
