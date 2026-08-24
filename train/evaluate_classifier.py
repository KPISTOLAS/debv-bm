"""Emit Table 2 (GT crops) and optional VLM cascade from a named checkpoint.

Stopping rule
-------------
Default is ``--stage gt`` on WAID + FIR only. Read the JSON (especially the
FIR confusion matrix and Bird row) before ``--stage cascade``.

Zero-shot vs paper (50-epoch unfrozen fine-tune):
  - Near 92/88/91 zero-shot: verify you loaded the intended checkpoint.
  - Well below, then fine-tune recovers: Path 1 holds; zero-shot is diagnostic,
    not reportable. Fine-tune is ``train_bioclip.py --pretrained hf-hub:imageomics/bioclip``
    (no OpenAI fallback). That path was never the default; it is untested.
  - Collapses and imageomics will not load: Path 2 (OpenAI CLIP), rename the paper.

FIR Bird is a taxon dump. Dominant Bird errors belong in Limitations, not as a
reason to switch checkpoints. camelus→camel is prompt-only.

Plants 125-way: refuse unless the Roboflow export is on disk with real names.
Do not report 92.00% from hardcoded JSON while the repo is public.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_prompts import prompt_for_class, refuse_placeholder_plants, yaml_names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CHECKPOINTS = {
    "bioclip": ("hf-hub:imageomics/bioclip", None),
    "openai-b16": ("ViT-B-16", "openai"),
    "openai-b32": ("ViT-B-32", "openai"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM classifier eval (GT crops, optional cascade)")
    p.add_argument("--checkpoint", choices=list(CHECKPOINTS), required=True)
    p.add_argument("--stage", choices=["gt", "cascade"], default="gt")
    p.add_argument("--domains", default="diurnal,nocturnal", help="comma: diurnal,nocturnal,plants")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--weights", type=Path, default=None, help="Optional fine-tuned .pt overlay")
    p.add_argument("--yolo-diurnal", type=Path, default=None)
    p.add_argument("--yolo-nocturnal", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--max-crops", type=int, default=0, help="Debug cap (0 = all)")
    p.add_argument("--dump-crops", type=Path, default=None, help="Write GT crops for train_bioclip.py")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "classifier_eval.json",
    )
    p.add_argument(
        "--waid",
        type=Path,
        default=None,
        help="WAID root with images/test and labels/test",
    )
    p.add_argument("--waid-yaml", type=Path, default=None, help="WAID class-name YAML (default: <waid>/data.yaml)")
    p.add_argument(
        "--fir",
        type=Path,
        default=None,
        help="FIR YOLO export root (test/images, test/labels, data.yaml)",
    )
    p.add_argument(
        "--plants-root",
        type=Path,
        default=None,
        help="Tree Species Identification v2 root (train/test class folders or YOLO layout)",
    )
    p.add_argument("--plants-names", type=Path, default=None)
    return p.parse_args()


def domain_specs(args: argparse.Namespace) -> dict:
    specs: dict = {}
    if args.waid is not None:
        specs["diurnal"] = {
            "images": args.waid / "images" / "test",
            "labels": args.waid / "labels" / "test",
            "yaml": args.waid_yaml or (args.waid / "data.yaml"),
            "modality": "rgb",
            "yolo_key": "yolo_diurnal",
        }
    if args.fir is not None:
        specs["nocturnal"] = {
            "images": args.fir / "test" / "images",
            "labels": args.fir / "test" / "labels",
            "yaml": args.fir / "data.yaml",
            "modality": "thermal",
            "yolo_key": "yolo_nocturnal",
        }
    return specs


def load_class_names(yaml_path: Path) -> list[str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return yaml_names(data["names"])


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def parse_yolo_label(path: Path, w: int, h: int, n_classes: int) -> list[dict]:
    gts = []
    if not path.exists():
        return gts
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= n_classes:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 > x1 and y2 > y1:
            gts.append({"cls": cls_id, "bbox": [x1, y1, x2, y2]})
    return gts


def crop_bgr(img, bbox) -> "object":
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (aa + bb - inter))


def iter_gt_crops(images_dir: Path, labels_dir: Path, names: list[str], max_crops: int):
    n = 0
    for img_path in list_images(images_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_yolo_label(labels_dir / f"{img_path.stem}.txt", w, h, len(names))
        for gt in gts:
            crop = crop_bgr(img, gt["bbox"])
            if crop.size == 0:
                continue
            yield img_path, crop, gt["cls"]
            n += 1
            if max_crops and n >= max_crops:
                return


def load_clip(checkpoint: str, device: str, overlay: Path | None):
    import open_clip
    import torch

    model_name, pretrained = CHECKPOINTS[checkpoint]
    if pretrained is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    print(f"loaded checkpoint={checkpoint} model={model_name} pretrained={pretrained}")
    if overlay is not None:
        try:
            raw = torch.load(overlay, map_location=device, weights_only=False)
        except TypeError:
            raw = torch.load(overlay, map_location=device)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        if isinstance(raw, dict) and any(k.startswith("module.") for k in raw):
            raw = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in raw.items()}
        incompatible = model.load_state_dict(raw, strict=True)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing or unexpected:
            raise SystemExit(
                f"Overlay {overlay} does not match {checkpoint}: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
    model.eval()
    return model, preprocess, tokenizer


def encode_prompts(tokenizer, names: list[str], modality: str, device):
    texts = [prompt_for_class(n, modality) for n in names]
    tokens = tokenizer(texts).to(device)
    return tokens, texts


def predict_crops(model, preprocess, tokenizer, names, modality, device, crops_cls, batch: int):
    import torch
    from PIL import Image

    tokens, prompt_texts = encode_prompts(tokenizer, names, modality, device)
    y_true: list[int] = []
    y_pred: list[int] = []
    buf_im = []
    buf_y = []

    def flush():
        if not buf_im:
            return
        x = torch.stack(buf_im).to(device)
        with torch.no_grad():
            image_features = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
            text_features = torch.nn.functional.normalize(model.encode_text(tokens), dim=-1)
            logits = 100.0 * image_features @ text_features.T
            pred = logits.argmax(dim=-1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(buf_y)
        buf_im.clear()
        buf_y.clear()

    for _path, crop, cls_id in crops_cls:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        im = preprocess(Image.fromarray(rgb))
        buf_im.append(im)
        buf_y.append(int(cls_id))
        if len(buf_im) >= batch:
            flush()
    flush()
    return y_true, y_pred, prompt_texts


def metrics(y_true: list[int], y_pred: list[int], names: list[str]) -> dict:
    n = len(names)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1
    per_class = {}
    recs, precs, f1s = [], [], []
    for i, name in enumerate(names):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        pre = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
        recs.append(rec)
        precs.append(pre)
        f1s.append(f1)
        per_class[name] = {
            "support": tp + fn,
            "recall": rec * 100.0,
            "precision": pre * 100.0,
            "f1": f1 * 100.0,
        }
    correct = sum(cm[i][i] for i in range(n))
    total = len(y_true)
    bird_idx = next((i for i, nm in enumerate(names) if nm.lower() == "bird"), None)
    bird = None
    if bird_idx is not None:
        bird_name = names[bird_idx]
        support = per_class[bird_name]["support"]
        err = support - cm[bird_idx][bird_idx]
        all_err = total - correct
        bird = {
            "support": support,
            "errors": err,
            "share_of_all_errors_pct": (100.0 * err / all_err) if all_err else 0.0,
            "note": (
                "Bird is a taxon dump, not a species. High Bird error is a dataset "
                "ceiling, not evidence that Path 1 failed."
            ),
        }
    return {
        "n": total,
        "accuracy_pct": 100.0 * correct / max(total, 1),
        "macro_recall_pct": 100.0 * sum(recs) / max(n, 1),
        "macro_precision_pct": 100.0 * sum(precs) / max(n, 1),
        "macro_f1_pct": 100.0 * sum(f1s) / max(n, 1),
        "per_class": per_class,
        "confusion_matrix": {"order": names, "counts": cm},
        "bird_limitation": bird,
        "hardcoded_table2_do_not_use": {
            "plants": 92.00,
            "diurnal": 88.08,
            "nocturnal": 91.00,
            "note": "Legacy JSON figures. Report accuracy_pct from this run, not these.",
        },
    }


def dump_crops(out_root: Path, names: list[str], crops) -> int:
    n = 0
    for _path, crop, cls_id in crops:
        dest = out_root / names[cls_id]
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / f"{n:06d}.jpg"), crop)
        n += 1
    return n


def eval_folder_classification(plants_root: Path, names: list[str], max_crops: int):
    """ImageFolder-style test/<class>/*.jpg if present; else YOLO test split."""
    test = plants_root / "test"
    images_dir = test / "images"
    if images_dir.is_dir():
        yield from iter_gt_crops(images_dir, test / "labels", names, max_crops)
        return
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = 0
    class_dirs = [d for d in test.iterdir() if d.is_dir()] if test.is_dir() else []
    for d in class_dirs:
        if d.name not in name_to_idx:
            continue
        idx = name_to_idx[d.name]
        for img_path in list_images(d):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            yield img_path, img, idx
            n += 1
            if max_crops and n >= max_crops:
                return


def run_cascade(spec, names, model, preprocess, tokenizer, modality, device, yolo_path, args):
    from ultralytics import YOLO
    import torch
    from PIL import Image

    yolo = YOLO(str(yolo_path))
    tokens, _ = encode_prompts(tokenizer, names, modality, device)
    with torch.no_grad():
        text_features = torch.nn.functional.normalize(model.encode_text(tokens), dim=-1)

    n_gt = n_det = n_matched = n_cls = 0
    images = list_images(spec["images"])
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_yolo_label(spec["labels"] / f"{img_path.stem}.txt", w, h, len(names))
        n_gt += len(gts)
        r = yolo.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        boxes = r[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            for b in boxes.xyxy.cpu().numpy():
                dets.append([float(x) for x in b.tolist()])
        n_det += len(dets)
        used = set()
        for gt in gts:
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(dets):
                if j in used:
                    continue
                v = iou_xyxy(gt["bbox"], d)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j < 0 or best_iou < args.iou_thr:
                continue
            used.add(best_j)
            n_matched += 1
            crop = crop_bgr(img, [int(x) for x in dets[best_j]])
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
                pred = int((100.0 * image_features @ text_features.T).argmax(dim=-1).item())
            # VLM argmax — never dets[j] class id
            if pred == gt["cls"]:
                n_cls += 1
    loc = 100.0 * n_matched / max(n_gt, 1)
    casc = 100.0 * n_cls / max(n_matched, 1)
    e2e = 100.0 * n_cls / max(n_gt, 1)
    return {
        "n_gt": n_gt,
        "n_detections": n_det,
        "n_matched_iou": n_matched,
        "n_vlm_correct": n_cls,
        "localisation_recall_pct": loc,
        "cascade_classification_accuracy_pct": casc,
        "end_to_end_instance_accuracy_pct": e2e,
        "protocol": "YOLO detect → IoU≥0.5 match → crop det box → VLM argmax (not detector class id)",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import torch

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip(args.checkpoint, device, args.weights)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    specs = domain_specs(args)
    report: dict = {
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "device": device,
        "overlay_weights": str(args.weights) if args.weights else None,
        "note": (
            "GT-crop accuracy is diagnostic if this is zero-shot. Paper claims a "
            "50-epoch unfrozen fine-tune. Do not paste hardcoded 92/88/91."
        ),
        "domains": {},
    }
    table_rows = []

    for domain in domains:
        if domain == "plants":
            if args.plants_root is None or not args.plants_root.exists():
                report["domains"]["plants"] = {
                    "skipped": True,
                    "reason": (
                        "Tree Species Identification v2 is not on disk. "
                        "92.00% cannot be reported as a run-backed number. "
                        "Re-download or pull the plant classification claim from the abstract."
                    ),
                }
                continue
            names_path = args.plants_names
            if names_path is None:
                yaml_p = args.plants_root / "data.yaml"
                if not yaml_p.exists():
                    raise SystemExit("Pass --plants-names YAML from the Roboflow export")
                names_path = yaml_p
            names = load_class_names(names_path)
            refuse_placeholder_plants(names)
            crops = eval_folder_classification(args.plants_root, names, args.max_crops)
            modality = "rgb"
            yolo_path = None
        else:
            spec = specs.get(domain)
            if spec is None:
                flag = "--waid" if domain == "diurnal" else "--fir"
                report["domains"][domain] = {
                    "skipped": True,
                    "reason": f"Pass {flag} pointing at the Roboflow export (not shipped in git)",
                }
                continue
            if not spec["images"].is_dir() or not spec["yaml"].exists():
                report["domains"][domain] = {
                    "skipped": True,
                    "reason": f"Missing images or YAML: {spec['images']} / {spec['yaml']}",
                }
                continue
            names = load_class_names(spec["yaml"])
            crops_list = list(iter_gt_crops(spec["images"], spec["labels"], names, args.max_crops))
            if args.dump_crops:
                dump_crops(args.dump_crops / domain / "train", names, crops_list)
            crops = crops_list
            modality = spec["modality"]
            yolo_path = args.yolo_diurnal if domain == "diurnal" else args.yolo_nocturnal

        y_true, y_pred, prompts = predict_crops(
            model, preprocess, tokenizer, names, modality, device, crops, args.batch
        )
        block = metrics(y_true, y_pred, names)
        block["prompts"] = prompts
        block["yaml_names"] = names
        if args.stage == "cascade":
            if domain == "plants":
                block["cascade"] = {
                    "skipped": True,
                    "reason": "Plant cascade on forest-trees is Tree-vs-background, not 125-way.",
                }
            elif yolo_path is None or not Path(yolo_path).exists():
                block["cascade"] = {
                    "skipped": True,
                    "reason": f"Pass --yolo-{domain} with detector weights",
                }
            else:
                block["cascade"] = run_cascade(
                    specs[domain],
                    names,
                    model,
                    preprocess,
                    tokenizer,
                    modality,
                    device,
                    yolo_path,
                    args,
                )
                gt_acc = block["accuracy_pct"]
                casc = block["cascade"]["cascade_classification_accuracy_pct"]
                block["cascade"]["drop_vs_gt_crop_pp"] = casc - gt_acc
        report["domains"][domain] = block
        table_rows.append(
            {
                "domain": domain,
                "checkpoint": args.checkpoint,
                "n": block["n"],
                "accuracy_pct": round(block["accuracy_pct"], 2),
                "macro_precision_pct": round(block["macro_precision_pct"], 2),
                "macro_recall_pct": round(block["macro_recall_pct"], 2),
                "macro_f1_pct": round(block["macro_f1_pct"], 2),
            }
        )
        print(
            f"{domain}: n={block['n']} acc={block['accuracy_pct']:.2f} "
            f"macroF1={block['macro_f1_pct']:.2f}"
        )
        if block.get("bird_limitation"):
            b = block["bird_limitation"]
            print(
                f"  Bird: support={b['support']} errors={b['errors']} "
                f"share_of_errors={b['share_of_all_errors_pct']:.1f}% "
                f"(dataset ceiling, not Path 1 failure)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.out.with_suffix(".csv"), table_rows)
    print(f"wrote {args.out}")
    print("STOP. Read confusion matrices before --stage cascade or renaming BioCLIP.")


if __name__ == "__main__":
    main()
