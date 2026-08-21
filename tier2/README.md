# Tier 2 — processing node

Runs on a field-station PC. Reads the Pi outbox (or any image folder).

```bash
python infer_pipeline.py --inbox ../outbox --output ../results --weights yolov11l.pt --domain diurnal
python dedup.py --results-dir ../results --iou 0.5 --seconds 30
```

If `--bioclip-weights` is omitted, boxes are still written and each crop is tagged `classifier_unavailable`. Grad-CAM++ is produced only when a classifier checkpoint is loaded and `pytorch-grad-cam` is installed.
