# Tier 1 — Raspberry Pi 5

On-site acquisition and preprocessing only. No YOLO/BioCLIP on the Pi.

```bash
python capture_preprocess.py --input ../examples --outbox ../outbox --buffer ../buffer
```

On a Pi 5 with Camera Module and `picamera2`:

```bash
python capture_preprocess.py --camera libcamera --outbox /mnt/data/outbox --buffer /mnt/data/buffer
```

Rejected frames (blur / under- / over-exposure) are logged and not uploaded. If the outbox copy fails, JPEGs stay in `--buffer` for later retry (`--retry-buffer`).
