# Tier 2 — processing node

Runs on a field-station PC. Reads the Pi outbox (or any image folder).

```bash
python infer_pipeline.py --inbox ../outbox --output ../results --weights yolov11l.pt --domain diurnal
python dedup.py --results-dir ../results --iou 0.5 --seconds 30
```

If `--bioclip-weights` is omitted, boxes are still written and each crop is tagged `classifier_unavailable`. If the path is given, a key mismatch is a hard error (the run does not continue with un-adapted CLIP weights).

Classifier Grad-CAM++ (`--heatmaps`) needs `--bioclip-weights` and `pytorch-grad-cam`. BioCLIP is ViT-B/16: the heatmap uses the last transformer LayerNorm plus a ViT `reshape_transform`, not a CNN `Conv2d`. Detector-level Grad-CAM on full frames (paper nocturnal panel) is a separate YOLO explainer, not this crop path.
