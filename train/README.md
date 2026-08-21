# Training (regional fine-tune)

Every new site needs its own closed-set inventory. Public Roboflow mAP/accuracy in the paper is **not** a field guarantee.

## YOLO

```bash
python train_yolo.py --data path/to/data.yaml --weights yolo11l.pt --epochs 100
python train_yolo.py --data path/to/data.yaml --weights runs/detect/debvbm_yolo/weights/best.pt --val-only
```

`data.yaml` must use paths on **your** machine (or relative to the yaml file). Do not commit Roboflow dumps.

## BioCLIP

```bash
python train_bioclip.py --data path/to/crops --config ../configs/diurnal.yaml
```

Crops: `train/<species>/*.jpg`. Defaults match the paper: 50 epochs, 224 px, batch 32, AdamW \(10^{-4}\), weight decay 0.01, cosine anneal to \(10^{-6}\), full unfreeze.

To start from Imageomics BioCLIP weights when your `open_clip` build supports the hub id:

```bash
python train_bioclip.py --data path/to/crops --config ../configs/nocturnal.yaml \
  --model ViT-B-16 --pretrained hf-hub:imageomics/bioclip
```

Plant 125-way names: copy `configs/plants.example.yaml` and fill from Tree Species Identification v2.
