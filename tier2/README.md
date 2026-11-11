# Tier 2 — processing node

Runs on a field-station PC. Reads the Pi outbox (or any image folder).

```bash
python infer_pipeline.py --inbox ../outbox --output ../results --weights yolov11l.pt --domain diurnal
python dedup.py --results-dir ../results --iou 0.5 --seconds 30
```

If `--bioclip-weights` is omitted, the pipeline runs in the detector-only operating mode: each detection's species is the detector's own class label (`classifier: detector`). This is the recommended mode for the animal streams: it is 2.3-5.8x cheaper in GPU time per image, was more accurate on the image-level benchmark splits, and on FIR recordings unseen in training neither label source was consistently better across three training seeds (paper, operating-mode and recording-split tables). Plants need `--bioclip-weights`, because the tree detector has no species classes. If the path is given, a key mismatch is a hard error (the run does not continue with un-adapted CLIP weights).

Classifier Grad-CAM++ (`--heatmaps`) needs `--bioclip-weights` and the `grad-cam` package (imported as `pytorch_grad_cam`). BioCLIP is ViT-B/16: the heatmap uses the last transformer LayerNorm plus a ViT `reshape_transform`, not a CNN `Conv2d`. The detector map used in the faithfulness evaluation is `train/yolo_cam.py` (class-score Grad-CAM++), not this path.
