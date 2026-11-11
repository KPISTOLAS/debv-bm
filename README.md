# DEBV-BM reference software

Reference implementation of the **Distributed Edge-Based Vision Framework for Biodiversity Monitoring (DEBV-BM)** described in the accompanying paper. This is research software that matches the three-tier architecture (Raspberry Pi 5 acquisition, a station-side processing node, and optional cloud copy). It is **not** a commercial appliance and does not include UAV autopilot, solar-charge firmware, or a production dashboard.

**Repository:** [https://github.com/KPISTOLAS/debv-bm](https://github.com/KPISTOLAS/debv-bm)

## What is and is not claimed

| Claim in the paper | What this repo provides |
|---|---|
| Field deployments | Operational stages: capture, quality filter, buffer, infer, write results |
| mAP / BioCLIP accuracy | Public **benchmarks** (forest trees v10, PureForest, Marche, a WAID derivative, FIR Animals), not labelled field photos |
| New forest / new taxa | You **must** fine-tune (`train/`) on a local closed-set inventory |

Deep-learning inference (YOLO, BioCLIP, Grad-CAM++) runs on the **Tier 2** node, not on the Pi 5. The edge tier acquires, preprocesses, quality-filters, buffers, and uploads.

## Layout

```
edge_pi5/     Tier 1: capture, downsample, optional RGB colour correction, QC, buffer, upload stub
tier2/        Tier 2: YOLO → crop → BioCLIP → Grad-CAM++, per-camera temporal dedup
train/        Dataset builders, fine-tuning, and every evaluation behind the paper's tables
configs/      Closed-set inventories (diurnal 9, nocturnal 7, Marche 10, PureForest 13) and split manifests
examples/     Three public WAID test frames so scripts run without the full dump
```

## Quick start (no GPU, no weights)

```bash
python -m pip install -r edge_pi5/requirements-edge.txt
python edge_pi5/capture_preprocess.py --input examples --outbox outbox --buffer buffer --node-id cam01
python tier2/dedup.py --results-dir results   # no-op if results/ is empty
```

Edge preprocessing defaults match the training data: frames are downsampled (long side ≤ 1280 px) and quality-filtered, but **not** colour-corrected. `--colour` enables CLAHE + brightness correction on RGB frames (the setting of the field deployments; it lowers benchmark accuracy, see the paper's edge re-score table). `--thermal` marks thermal frames, which are never colour-corrected. `--node-id` names the camera; Tier 2 deduplicates only within one node.

With detector weights on a station PC:

```bash
python -m pip install -r tier2/requirements-tier2.txt
python tier2/infer_pipeline.py --inbox outbox --output results --weights path/to/yolo.pt --domain diurnal
```

## Regional adaptation (every new deployment)

Classifiers are **closed-set**. A new region needs a new inventory and another fine-tune; benchmark numbers do not transfer automatically.

1. Write a species list (`configs/your_region.yaml`).
2. Label detector boxes and species crops (or start from the cited public sets, then add local images).
3. Fine-tune YOLO: `python train/train_detectors.py --data region=path/to/data.yaml --models yolo11l yolov8n`, and pick the detector on validation mAP.
4. Fine-tune BioCLIP: `python train/train_bioclip.py --data path/to/crops --config configs/your_region.yaml`.
5. Evaluate on a **held-out local** split. Benchmark accuracy is not field accuracy.

Paper recipe for BioCLIP (`train/train_bioclip.py`): unfrozen backbone, 224×224, batch 32, AdamW lr 1e-5 (cosine to 1e-7), weight decay 0.01, 50 epochs (PureForest 10), checkpoint selected on validation. lr 1e-4 did not adapt the plant head.

## Reproducing the paper

`train/README.md` lists the command for every table and figure (dataset builders, test-split and class-agnostic detector evaluation, bootstrap intervals, animal and plant cascades, edge-preprocessing re-score, faithfulness, error propagation, throughput and latency). Split manifests are in `configs/splits/`. The trained weights are on Zenodo (`WEIGHTS.md`).

- **Seeds.** Detectors were trained with Ultralytics `seed=0`, deterministic mode; each table is a single training run. Local retraining of eight detectors reproduced Table 1 within 1.7 pp of mAP@0.5.
- **Pretrained weights.** Detectors start from the COCO-pretrained Ultralytics checkpoints; BioCLIP heads from `hf-hub:imageomics/bioclip` (pass `--pretrained hf-hub:imageomics/bioclip`; the CLI default is OpenAI CLIP).
- **Versions.** The requirement files are **minimum** pins. The paper's numbers were produced with Ultralytics 8.4.200, PyTorch 2.11.0 (CUDA 12.8), open-clip-torch 3.3.0, and grad-cam 1.5.7 on Windows 11 with an RTX 5080.

## Tests / CI

`python -m unittest discover -s tests -v` covers example QC, dedup IoU, AD/IC arithmetic, and ViT Grad-CAM helpers. GitHub Actions runs the edge quick-start plus those tests (no GPU, no paper weights).

## Archive

Software: concept DOI [10.5281/zenodo.22082968](https://doi.org/10.5281/zenodo.22082968) (resolves to the latest release); the paper's results were produced with release **v1.4.0**. Trained weights: concept DOI [10.5281/zenodo.23920497](https://doi.org/10.5281/zenodo.23920497); see [`WEIGHTS.md`](WEIGHTS.md). `main` is mutable.

## Hardware notes

- **Pi 5:** `edge_pi5/` runs in **folder mode** on any PC. On a Pi, point `--camera libcamera` if `picamera2` is installed (see `edge_pi5/README.md`).
- **Tier 2:** laptop or small GPU server. The paper's detector and BioCLIP weights are in the Zenodo record (`WEIGHTS.md`); `weights/paper/` keeps three earlier checkpoints.
- **UAV (field, not in this repo):** Holybro X500 V2 quadrotor (four rotors). Flight stack and payload wiring are not shipped here.

## Citation

See `CITATION.cff` and the journal article. Benchmark imagery: *forest trees v10* and FIR Animals (Roboflow Universe), PureForest (IGN), Marche UAV crowns (Zenodo 10.5281/zenodo.14988733), and a nine-label WAID derivative (WAID: Mou et al. 2023); all are cited in the paper.
