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

Rejected frames (blur / under- / over-exposure) are logged and not uploaded. Accepted frames are written to `--buffer` and copied to the outbox; the local copy is kept (a local archive), and if the copy fails the frame is retried later (`--retry-buffer`).

## System measurements (edge tier)

`bench_edge.py` runs the released edge path on a folder of raw camera frames and logs, per frame, the edge time, bytes captured and sent, delivery delay into the processing-node inbox, and the outage backlog. `--mode baseline` uploads the raw frames instead (central preprocessing). Suggested protocol, with the processing-node inbox mounted at `/mnt/tier2/inbox` on every Pi:

```bash
# 1. Single node, edge vs baseline, 500 representative raw frames (include empty frames)
python bench_edge.py --input raw/ --outbox /mnt/tier2/inbox --mode edge     --node-id pi-01 --out pi-01_edge.csv
python bench_edge.py --input raw/ --outbox /mnt/tier2/inbox --mode baseline --node-id pi-01 --out pi-01_base.csv
# 2. Outage: link down for frames 100-399, at the deployed capture interval
python bench_edge.py --input raw/ --outbox /mnt/tier2/inbox --mode edge --node-id pi-01 --interval 2 \
    --outage-start 100 --outage-frames 300 --out pi-01_outage.csv
# 3. All five nodes at once (start together, e.g. via ssh), same command with each --node-id
# 4. Aggregate
python summarize_edge_bench.py pi-*_*.csv --out edge_bench_summary.json
```

Log power during runs with `measure_power.py` and record the meter model, measurement point, and sampling interval.
