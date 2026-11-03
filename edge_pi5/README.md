# Tier 1 — Raspberry Pi 5

On-site acquisition and preprocessing only. No YOLO/BioCLIP on the Pi.

```bash
python capture_preprocess.py --input ../examples --outbox ../outbox --buffer ../buffer
```

On a Pi 5 with Camera Module and `picamera2`:

```bash
python capture_preprocess.py --camera libcamera --outbox /mnt/data/outbox --buffer /mnt/data/buffer
```

Frames are downsampled (long side <= 1280 px) and quality-filtered. Colour correction (CLAHE + brightness) is off by default because the released models were trained without it; `--colour` enables it for RGB frames, and `--thermal` frames are never colour-corrected. The blur threshold (`--blur-thr 60`) rejects many low-contrast thermal frames; lower it for thermal cameras. `--node-id` names the camera for per-camera deduplication on Tier 2.

Rejected frames (blur / under- / over-exposure) are logged and not uploaded. If the outbox copy fails, JPEGs stay in `--buffer` for later retry (`--retry-buffer`).
