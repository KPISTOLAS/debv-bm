# DEBV-BM reference software

Reference implementation of the **Distributed Edge-Based Vision Framework for Biodiversity Monitoring (DEBV-BM)** described in the accompanying paper. This is research software that matches the three-tier architecture (Raspberry Pi 5 acquisition, a station-side processing node, and optional cloud copy). It is **not** a commercial appliance and does not include UAV autopilot, solar-charge firmware, or a production dashboard.

**Repository:** [https://github.com/KPISTOLAS/debv-bm](https://github.com/KPISTOLAS/debv-bm)

## What is and is not claimed

| Claim in the paper | What this repo provides |
|---|---|
| Field deployments | Operational stages: capture, quality filter, buffer, infer, write results |
| mAP / BioCLIP accuracy | Public Roboflow **benchmarks**, not labelled field photos |
| New forest / new taxa | You **must** fine-tune (`train/`) on a local closed-set inventory |

Deep-learning inference (YOLOv11, BioCLIP, Grad-CAM++) runs on the **Tier 2** node, not on the Pi 5. The edge tier acquires, preprocesses, quality-filters, buffers, and uploads.

## Layout

```
edge_pi5/     Tier 1: capture, colour/brightness, downsample, QC, buffer, upload stub
tier2/        Tier 2: YOLO → crop → BioCLIP → Grad-CAM++, spatial–temporal dedup
train/        Fine-tune YOLO and BioCLIP for a regional species list
configs/      Closed-set inventories (diurnal 9, nocturnal 7, plant example)
examples/     Three public WAID test frames so scripts run without the full dump
```

## Quick start (no GPU, no weights)

```bash
python -m pip install -r edge_pi5/requirements-edge.txt
python edge_pi5/capture_preprocess.py --input examples --outbox outbox --buffer buffer
python tier2/dedup.py --results-dir results   # no-op if results/ is empty
```

With detector weights on a station PC:

```bash
python -m pip install -r tier2/requirements-tier2.txt
python tier2/infer_pipeline.py --inbox outbox --output results --weights path/to/yolo.pt --domain diurnal
```

## Regional adaptation (every new deployment)

Classifiers are **closed-set**. A new region needs a new inventory and another fine-tune; the public Roboflow numbers do not transfer automatically.

1. Write a species list (`configs/your_region.yaml`).
2. Label detector boxes and species crops (or start from the cited Roboflow sets, then add local images).
3. Fine-tune YOLO: `python train/train_yolo.py --data path/to/data.yaml`.
4. Fine-tune BioCLIP: `python train/train_bioclip.py --data path/to/crops --config configs/your_region.yaml`.
5. Evaluate on a **held-out local** split. Do not reuse the paper’s 92% plant figure as field accuracy.

Paper recipe for BioCLIP (implemented in `train/train_bioclip.py`): unfrozen backbone, 50 epochs, 224×224, batch 32, AdamW \(lr=10^{-4}\), weight decay 0.01, cosine annealing.

## Reproducibility (matches the paper Limitations item)

- **Seed.** The reported tables are single-run. Training scripts that produced those numbers did **not** pin a paper-wide seed. Ultralytics `args.yaml` files in local experiment folders show the library default (`0` or `42` depending on Ultralytics version), not a chosen experimental seed. `train/train_bioclip.py --seed 0` is a convenience default for new fine-tunes, not a claim that Tables 1–2 used seed 0.
- **YOLO weights.** Fine-tune from Ultralytics COCO-pretrained `yolo11l.pt` (`train/train_yolo.py --weights yolo11l.pt`). Not trained from scratch.
- **Classifier weights.** Loading Stevens et al. BioCLIP is `--pretrained hf-hub:imageomics/bioclip`. The CLI default remains OpenAI CLIP (`--pretrained openai`). Do not describe a run as BioCLIP unless that hub checkpoint actually loaded.
- **Versions.** Files `train/requirements-train.txt`, `tier2/requirements-tier2.txt`, and `edge_pi5/requirements-edge.txt` are **minimum** pins (`>=`), not a freeze of the Ubuntu training node. Record `python --version` and `pip freeze` from that node if you need bitwise-reproducible numbers.
- **Table 2 (BioCLIP on GT crops).** Roboflow dumps are not in git. Point `train/evaluate_classifier.py` (alias `train/eval_bioclip.py`) at your WAID / FIR / plant exports:

```bash
python train/evaluate_classifier.py --checkpoint bioclip --stage gt \
  --waid path/to/WAID --waid-yaml path/to/yolo_dataset.yaml \
  --fir path/to/FIR_export --out results/classifier_eval.json
```

`--stage cascade` needs `--yolo-diurnal` / `--yolo-nocturnal`. Plants 125-way needs `--plants-root` plus export names; `configs/plants.example.yaml` is not that inventory. Zero-shot hub weights are diagnostic; paper Table 2 is a 50-epoch unfrozen fine-tune. Do not paste hardcoded 92/88/91.

- **AD/IC (explainability).** Detector Grad-CAM++ vs Grad-CAM on GT boxes, top-20% salient mask, N=150 (100 correct / 50 wrong):

```bash
python train/evaluate_faithfulness.py --weights path/to/yolo.pt \
  --images path/to/WAID/images/test --labels path/to/WAID/labels/test \
  --out results/faithfulness.json
```

Published 27.91% AD / 0.00% IC / 0.20 IoU need the archived detector checkpoint (`WEIGHTS.md`). `--quick` is a smoke run only.

## Tests / CI

`python -m unittest discover -s tests -v` covers example QC, dedup IoU, AD/IC arithmetic, and ViT Grad-CAM helpers. GitHub Actions runs the edge quick-start plus those tests (no GPU, no paper weights). Requirements stay **minimum** `>=` pins; the paper already notes there is no Ubuntu lockfile. Record `pip freeze` from the training node if you need bitwise numbers.

## Archive

See [`WEIGHTS.md`](WEIGHTS.md) and `weights/paper/` (v1.0.2). Until a Zenodo DOI exists, paste it into `CITATION.cff` after minting. `main` is mutable.

## Hardware notes

- **Pi 5:** `edge_pi5/` runs in **folder mode** on any PC. On a Pi, point `--camera libcamera` if `picamera2` is installed (see `edge_pi5/README.md`).
- **Tier 2:** laptop or small GPU server. Detector checkpoints used by the local eval scripts are in `weights/paper/` (tag v1.0.2). BioCLIP fine-tunes are not in that folder.
- **UAV (field, not in this repo):** Holybro X500 V2 quadrotor (four rotors). Flight stack and payload wiring are not shipped here.

## Citation

See `CITATION.cff` and the journal article. Benchmark imagery: *forest trees v10*, Tree Species Identification v2, WAID, FIR Animals (Roboflow; cited in the paper).
