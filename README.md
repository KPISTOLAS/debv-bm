# DEBV-BM reference software

Reference implementation of the **Distributed Edge-Based Vision Framework for Biodiversity Monitoring (DEBV-BM)** described in the accompanying paper. This is research software that matches the three-tier architecture (Raspberry Pi 5 acquisition, a station-side processing node, and optional cloud copy). It is **not** a commercial appliance and does not include UAV autopilot, solar-charge firmware, or a production dashboard.

**Repository:** replace this placeholder after you create the GitHub remote:

`https://github.com/<ORG>/debv-bm`

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
examples/     Tiny synthetic frames so scripts run without Roboflow dumps
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

## Hardware notes

- **Pi 5:** `edge_pi5/` runs in **folder mode** on any PC. On a Pi, point `--camera libcamera` if `picamera2` is installed (see `edge_pi5/README.md`).
- **Tier 2:** laptop or small GPU server. Weights are not stored in git (see `.gitignore`).
- **Not in this repo:** DJI/PX4 flight code, MPPT solar controllers, field JPEGs, cloud IAM.

## Citation

See `CITATION.cff` and the journal article. Benchmark imagery: *forest trees v10*, Tree Species Identification v2, WAID, FIR Animals (Roboflow; cited in the paper).
