"""Usage: python train_waid_official.py  (re-run to continue; finished models are skipped)

Train the remaining Table 1 detectors on the official WAID split (paper protocol) and score them."""
import json
from pathlib import Path
import os
import shutil

from ultralytics import YOLO

ROOT = os.environ.get("DEBV_ROOT", Path(__file__).resolve().parents[2].as_posix())  # folder holding debv-bm/, data/, AI_Plant/
P = ROOT + "/debv-bm/train/runs/detect"
D = ROOT + "/data/waid_official/data.yaml"
DD = ROOT + "/data/waid_official/data_dedup.yaml"
OUT = ROOT + "/data/waid_official/results_all.json"
MODELS = ["yolo11n", "yolov8l", "yolo11l"]


def score(w, data, split, single_cls=False):
    m = YOLO(w).val(data=data, split=split, imgsz=640, batch=16, conf=0.001, iou=0.5, single_cls=single_cls,
                    device=0, plots=False, verbose=False, project=P, name="eval_tmp", exist_ok=True)
    p, r = float(m.box.mp) * 100, float(m.box.mr) * 100
    return {"precision": round(p, 2), "recall": round(r, 2), "f1": round(2 * p * r / (p + r), 2) if p + r else 0,
            "mAP50": round(float(m.box.map50) * 100, 2), "mAP50_95": round(float(m.box.map) * 100, 2),
            "inference_ms": round(float(m.speed["inference"]), 2)}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "yolov8n" not in res:
        res["yolov8n"] = json.load(open(ROOT + "/data/waid_official/results_yolov8n.json"))
    for name in MODELS:
        if name in res:
            print("skip", name, flush=True)
            continue
        run = f"waid_official_{name}"
        last = f"{P}/{run}/weights/last.pt"
        if os.path.exists(last):  # interrupted earlier: continue from the last epoch with the same settings
            print("resume", name, flush=True)
            YOLO(last).train(resume=True)
        else:
            YOLO(f"{ROOT}/AI_Plant/yolo_local/{name}.pt").train(
                data=D, epochs=100, batch=16, imgsz=640, optimizer="AdamW", lr0=0.001, patience=20, seed=0,
                device=0, workers=6, project=P, name=run, exist_ok=True)
        w = f"{P}/{run}/weights/best.pt"
        shutil.copy2(w, f"{ROOT}/AI_Plant/zenodo_weights/detector_diurnal_waidofficial_{name}.pt")
        res[name] = {"val": score(w, D, "val"), "test_official": score(w, D, "test"), "test_dedup": score(w, DD, "test"),
                     "test_official_agnostic": score(w, D, "test", True), "test_dedup_agnostic": score(w, DD, "test", True)}
        json.dump(res, open(OUT, "w"), indent=2)
        print("RESULT", name, json.dumps(res[name]), flush=True)


if __name__ == "__main__":
    main()
