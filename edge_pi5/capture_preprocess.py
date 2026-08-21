"""Raspberry Pi 5 / folder-mode capture, QC, buffer, and upload stub.

Inference is not run here. See ../tier2 for YOLOv11–BioCLIP–Grad-CAM++.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEBV-BM Tier 1: acquire, preprocess, QC, buffer, upload")
    p.add_argument("--input", type=Path, help="Folder of images (folder mode; use on any PC)")
    p.add_argument("--camera", choices=["none", "libcamera"], default="none")
    p.add_argument("--outbox", type=Path, default=Path("outbox"), help="Destination seen by the Tier 2 node")
    p.add_argument("--buffer", type=Path, default=Path("buffer"), help="Local store if copy to outbox fails")
    p.add_argument("--max-side", type=int, default=1280, help="Downsample so the long side is at most this")
    p.add_argument("--blur-thr", type=float, default=60.0, help="Reject if Laplacian variance is below this")
    p.add_argument("--dark-thr", type=float, default=12.0, help="Reject if mean luma is below this")
    p.add_argument("--bright-thr", type=float, default=245.0, help="Reject if mean luma is above this")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--thermal", action="store_true", help="Mark frames as thermal infrared")
    p.add_argument("--retry-buffer", action="store_true", help="Copy any leftover buffer files to outbox")
    p.add_argument("--libcamera-frames", type=int, default=1)
    return p.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def capture_libcamera(n: int) -> list[np.ndarray]:
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise SystemExit(
            "picamera2 is not installed. On Raspberry Pi OS: sudo apt install python3-picamera2\n"
            "Or use --input with a folder of images."
        ) from e
    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    frames = []
    for _ in range(n):
        frames.append(cam.capture_array())
    cam.stop()
    return frames


def colour_brightness(bgr: np.ndarray) -> np.ndarray:
    """Simple colour normalisation and mid-tone brightness correction."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    mean_l = float(l.mean())
    if mean_l > 1:
        l = np.clip(l.astype(np.float32) * (128.0 / mean_l), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def downsample(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def qc_ok(bgr: np.ndarray, blur_thr: float, dark_thr: float, bright_thr: float) -> tuple[bool, str, dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luma = float(gray.mean())
    stats = {"laplacian_var": blur, "mean_luma": luma}
    if blur < blur_thr:
        return False, "blur", stats
    if luma < dark_thr:
        return False, "underexposed", stats
    if luma > bright_thr:
        return False, "overexposed", stats
    return True, "ok", stats


def write_record(
    bgr: np.ndarray,
    stem: str,
    dest_dir: Path,
    meta: dict,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    jpg = dest_dir / f"{stem}.jpg"
    js = dest_dir / f"{stem}.json"
    cv2.imwrite(str(jpg), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return jpg


def upload(src_dir: Path, outbox: Path, stem: str) -> bool:
    outbox.mkdir(parents=True, exist_ok=True)
    try:
        for ext in (".jpg", ".json"):
            shutil.copy2(src_dir / f"{stem}{ext}", outbox / f"{stem}{ext}")
        return True
    except OSError:
        return False


def process_bgr(bgr: np.ndarray, args: argparse.Namespace, source: str) -> None:
    bgr = colour_brightness(bgr)
    bgr = downsample(bgr, args.max_side)
    ok, reason, stats = qc_ok(bgr, args.blur_thr, args.dark_thr, args.bright_thr)
    now = datetime.now(timezone.utc).isoformat()
    stem = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    meta = {
        "timestamp_utc": now,
        "source": source,
        "thermal": bool(args.thermal),
        "gps": {"lat": args.lat, "lon": args.lon} if args.lat is not None and args.lon is not None else None,
        "qc": {"passed": ok, "reason": reason, **stats},
        "shape": list(bgr.shape),
    }
    if not ok:
        print(f"REJECT {source} ({reason}) {stats}")
        log = args.buffer / "rejected.jsonl"
        args.buffer.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        return
    write_record(bgr, stem, args.buffer, meta)
    if upload(args.buffer, args.outbox, stem):
        print(f"UPLOAD {stem}")
    else:
        print(f"BUFFERED {stem} (outbox copy failed)")


def retry_buffer(args: argparse.Namespace) -> None:
    if not args.buffer.exists():
        return
    for jpg in sorted(args.buffer.glob("*.jpg")):
        stem = jpg.stem
        if (args.outbox / f"{stem}.jpg").exists():
            continue
        if upload(args.buffer, args.outbox, stem):
            print(f"RETRY-OK {stem}")
        else:
            print(f"RETRY-FAIL {stem}")


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise SystemExit("Install edge dependencies: pip install -r edge_pi5/requirements-edge.txt")
    if args.retry_buffer:
        retry_buffer(args)
        return
    if args.camera == "libcamera":
        for i, frame in enumerate(capture_libcamera(args.libcamera_frames)):
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            process_bgr(frame, args, source=f"libcamera:{i}")
        return
    if args.input is None or not args.input.exists():
        raise SystemExit("Provide --input <folder> or --camera libcamera")
    for path in list_images(args.input):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"SKIP unreadable {path}", file=sys.stderr)
            continue
        process_bgr(bgr, args, source=str(path.name))


if __name__ == "__main__":
    main()
