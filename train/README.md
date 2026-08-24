# Training (regional fine-tune)

Every new site needs its own closed-set inventory. Public Roboflow mAP/accuracy in the paper is **not** a field guarantee.

## YOLO

```bash
python train_yolo.py --data path/to/data.yaml --weights yolo11l.pt --epochs 100
# yolo11l.pt is the Ultralytics COCO-pretrained checkpoint (not from-scratch).
# seed is not passed; Ultralytics uses its library default.
python train_yolo.py --data path/to/data.yaml --weights runs/detect/debvbm_yolo/weights/best.pt --val-only
```

`data.yaml` must use paths on **your** machine (or relative to the yaml file). Do not commit Roboflow dumps.

## BioCLIP

```bash
python train_bioclip.py --data path/to/crops --config ../configs/diurnal.yaml
```

Crops: `train/<species>/*.jpg`. Defaults match the paper: 50 epochs, 224 px, batch 32, AdamW 10^{-4}, weight decay 0.01, cosine anneal to 10^{-6}, full unfreeze.

There is **no silent OpenAI fallback**. If `hf-hub:imageomics/bioclip` fails to load, training stops. Default `--pretrained openai` is OpenAI CLIP ViT-B-16, not BioCLIP. Path 1 (paper name BioCLIP) requires this command **and** a successful load:

```bash
python train_bioclip.py --data path/to/crops --config ../configs/nocturnal.yaml \
  --model ViT-B-16 --pretrained hf-hub:imageomics/bioclip
```

That Imageomics fine-tune has not been the default; treat it as untested until it completes.

## Classifier eval (replaces hardcoded Table 2 JSON)

Zero-shot GT crops first; stop; read confusion matrices (FIR **Bird** is a taxon dump, not a Path 1 failure):

```bash
python evaluate_classifier.py --checkpoint bioclip --stage gt \
  --waid path/to/WAID --waid-yaml path/to/yolo_dataset.yaml \
  --fir path/to/FIR_export
# alias: python eval_bioclip.py ... (same arguments)
```

`--checkpoint openai-b16` or `openai-b32` for Path 2. Optional `--dump-crops dir` writes YOLO GT crops into `train/<class>/` layout for `train_bioclip.py`. `--stage cascade` only after GT-crop numbers are read; requires `--yolo-diurnal` / `--yolo-nocturnal`. Cascade uses VLM argmax on detector boxes, never YOLO class ids.

Plant 125-way: only with Tree Species Identification v2 on disk (`--plants-root` + export `data.yaml`). Do not use `plants.example.yaml` placeholders. Without that export, **92.00% cannot be a run-backed number** and must not stay in the abstract sourced to hardcoded JSON.