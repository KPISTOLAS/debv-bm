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

The R3 revision tables use this Imageomics fine-tune with `--lr 1e-5 --min-lr 1e-7` (see below): at the default `10^{-4}` the Marche plant head did not improve over zero-shot BioCLIP on validation (39.7% vs 38.4%).

## Classifier eval (replaces hardcoded Table 2 JSON)

Zero-shot GT crops first; stop; read confusion matrices (FIR **Bird** is a taxon dump, not a Path 1 failure):

```bash
python evaluate_classifier.py --checkpoint bioclip --stage gt \
  --waid path/to/WAID --waid-yaml path/to/yolo_dataset.yaml \
  --fir path/to/FIR_export
# alias: python eval_bioclip.py ... (same arguments)
```

`--checkpoint openai-b16` or `openai-b32` for Path 2. Optional `--dump-crops dir` writes YOLO GT crops into `train/<class>/` layout for `train_bioclip.py`. `--stage cascade` only after GT-crop numbers are read; requires `--yolo-diurnal` / `--yolo-nocturnal`. Cascade uses VLM argmax on detector boxes, never YOLO class ids.

Plant 125-way (Tree Species Identification v2) is no longer used by the paper; R3 replaces it with PureForest and Marche (next section). Do not use `plants.example.yaml` placeholders.

## R3 revision pipeline (Tables 1, 2 and error propagation)

Datasets, all public:

| Domain | Source | Split |
|---|---|---|
| Plant detection | forest trees v10 (Roboflow) | export as obtained, 2528/495/134 |
| Plant classification | PureForest, `huggingface.co/datasets/IGNF/PureForest` | official forest-level split |
| Plant classification (UAV) | Marche, Zenodo `10.5281/zenodo.14988733` | by source photo, `../configs/splits/marche_split_manifest.json` |
| Diurnal | WAID | test 402; 20% of training images held out, `../configs/splits/waid_crops_manifest.json` |
| Nocturnal | FIR Animals (cleaned, 7 classes) | export as obtained, `../configs/splits/fir_crops_manifest.json` |

```bash
# Plant classification sets
python build_pureforest_classify.py convert --raw path/to/PureForest --out data/pureforest
python build_marche_classify.py --archive path/to/Image_Dataset.zip --out data/marche

# Animal ground-truth crops (image-level splits)
python build_gt_crops.py --src path/to/WAID --layout images-labels --config ../configs/diurnal.yaml --out data/waid_crops --val-frac 0.2
python build_gt_crops.py --src path/to/FIR --layout split-first --config ../configs/nocturnal.yaml --out data/fir_crops

# BioCLIP heads (PureForest: --epochs 10; others: 50) and one-time test scoring
python train_bioclip.py --data data/marche --config ../configs/plants_marche.yaml \
  --pretrained hf-hub:imageomics/bioclip --lr 1e-5 --min-lr 1e-7 --workers 4 --out runs/marche/best.pt
python eval_folder_classifier.py --data data/marche --config ../configs/plants_marche.yaml \
  --weights runs/marche/best.pt --out results/marche_test.json

# Detectors (trained on the Ultralytics Platform with AdamW, lr0 0.001, batch 16, patience 20, seed 0)
python eval_detector_test.py --data path/to/data.yaml --weights-dir path/to/weights --out results/detect_test.json

# Error-propagation table (0.1 is the reported sensitivity run)
python error_propagation.py --waid results/waid_test.json --fir results/fir_test.json \
  --marche results/marche_test.json --out results/error_propagation.json --smooth 0.01

# Table 1 bootstrap intervals (validator's own per-image matches; paired model differences)
python bootstrap_detector_ci.py --data path/to/data.yaml --weights-dir path/to/weights --out results/ci.json

# Table 6: deployed YOLOv11l -> BioCLIP cascade, incl. detector-class vs BioCLIP McNemar test
python eval_cascade.py --images FIR/test/images --labels FIR/test/labels --detector fir_yolo11l.pt \
  --config ../configs/nocturnal.yaml --bioclip runs/fir/best.pt --out results/fir_cascade.json

# Faithfulness (AD/IC) on the deployed detector; per-sample rows are written for bootstrapping
python evaluate_faithfulness.py --weights waid_yolo11l.pt --images WAID/images/test --labels WAID/labels/test

# Throughput (Pipeline Performance) and classifier latency figure
python benchmark_pipeline.py --images FIR/test/images --detector fir_yolo11l.pt \
  --config ../configs/nocturnal.yaml --bioclip runs/fir/best.pt --out results/throughput.json
python benchmark_classifiers.py --out results/latency.json --figure latency.png

# Example detection / BioCLIP / explanation figures
python make_example_figures.py --day-image WAID/images/test/dataset_2897.jpg --fir-test FIR/test \
  --waid-detector waid_yolo11l.pt --fir-detector fir_yolo11l.pt \
  --waid-bioclip runs/waid/best.pt --fir-bioclip runs/fir/best.pt --out figures/

# Local reproducibility retraining of the detectors
python train_detectors.py --data plants=path/to/forest-trees/data.yaml --models yolov8n yolo11n
```

Grad-CAM++ needs `pip install grad-cam` (imported as `pytorch_grad_cam`).

## Faithfulness (AD/IC)

```bash
python evaluate_faithfulness.py --weights path/to/yolo.pt \
  --images path/to/images/test --labels path/to/labels/test
```

Defaults match the paper: 100 correct + 50 wrong, top-20% salient pixels inside the GT box, Grad-CAM++ and Grad-CAM. Checkpoints are not in git (`../WEIGHTS.md`).