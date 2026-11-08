"""Build a YOLO detection set from BAMBI (forest wildlife, nadir UAV video) with flight-level splits.

BAMBI (Praschl et al., NeurIPS 2026 Datasets Track, CC-BY-4.0) ships each flight as one video,
thermal on the left half and RGB on the right (1024x1024 each), with MOT annotations on sparse,
human-annotated key frames. This script turns it into a frame-level YOLO set that is safe to
benchmark on:

* Split unit is the **flight**, taken from BAMBI's official ``flight_metadata/splits.json``, so
  near-identical frames of one video can never land in two splits.
* Only key frames (``is_propagated == 0``) are used. Per flight, the first ``--max-per-flight``
  usable key frames are taken, each at least ``--min-gap`` video frames (default 60, ~2 s) after
  the previous one.
* RGB boxes come from the ``owl-transferred`` layer (``<id>_rgb_gt.txt``). That layer is
  machine-generated; a frame is skipped if any of its boxes kept the thermal position
  (provenance ``unchanged``), because such a box does not sit on the animal in RGB.
* A frame is skipped if fewer than ``--min-rgb-detected`` of its boxes were found by OWL in RGB
  itself (provenance ``owl``/``smoothed``; ``consensus`` boxes were placed by the frame-wide
  shift only). This removes fog, dusk and under-canopy frames where the animal is visible in
  thermal but not in RGB.
* A frame is skipped if it holds an animal outside the class map (e.g. fallow deer, human,
  ``Unknown``); keeping it would leave an unlabelled positive in the image. A box lying mostly
  (``--min-visible``) in the black border outside the valid image area is dropped.
* ``--review`` renders every test image with its boxes so the test split can be checked by hand
  (``review/test_checklist.csv``); report test scores only on the verified images.

Speed: by default nothing is downloaded to disk. Each flight's annotations are read out of its
Zenodo archive with HTTP range requests (a few MB); flights with no usable frame stop there. For
the others the video member is decompressed as a stream and decoding stops right after the last
wanted frame, so only the start of each video is transferred. ``--workers`` flights run at once
(Zenodo gives ~2.5 MB/s per connection). Flights whose files are already in ``--raw`` are read
locally. Finished flights are recorded in ``<out>/.done`` and skipped when the command is re-run.

Usage (from debv-bm/train):
  python build_bambi_detect.py --bambi-repo ../../external/bambi-Dataset \
      --config ../configs/diurnal_bambi.yaml --out ../../data/bambi_rgb --review --workers 6 \
      --manifest ../configs/splits/bambi_rgb_manifest.json
  # quick test on two flights:
  python build_bambi_detect.py ... --flights 146 152
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import cv2
import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bambi_remote import RangeFile

HALF = 1024  # each modality is 1024x1024 in the processed video
COLOURS = [(0, 200, 255), (0, 255, 0), (255, 120, 0), (255, 0, 255)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BAMBI flights -> YOLO detection set (flight-level splits)")
    p.add_argument("--bambi-repo", type=Path, required=True, help="clone of github.com/bambi-eco/Dataset")
    p.add_argument("--raw", type=Path, default=None, help="optional directory with already downloaded flights")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--modality", choices=["rgb", "thermal"], default=None, help="default: config 'modality'")
    p.add_argument("--flights", nargs="+", default=None, help="flight ids (default: every eligible flight)")
    p.add_argument("--min-gap", type=int, default=60, help="min video frames between sampled frames")
    p.add_argument("--max-per-flight", type=int, default=10)
    p.add_argument("--min-visible", type=float, default=0.5, help="min fraction of a box inside the valid mask")
    p.add_argument("--min-rgb-detected", type=float, default=0.5,
                   help="RGB only: min fraction of a frame's boxes that OWL itself found in RGB "
                        "(provenance owl/smoothed, not consensus); filters fog, dusk and under-canopy frames")
    p.add_argument("--workers", type=int, default=6, help="flights processed in parallel")
    p.add_argument("--review", action="store_true", help="render test images with boxes for hand checking")
    p.add_argument("--manifest", type=Path, default=None, help="write the frame-level split manifest here")
    return p.parse_args()


def eligible_flights(repo: Path, modality: str) -> dict[str, str]:
    """flight id -> split, restricted to flights that carry annotations for this modality."""
    splits = json.loads((repo / "flight_metadata" / "splits.json").read_text(encoding="utf-8"))
    split_of = {str(f): s for s, flights in splits.items() for f in flights}
    if modality == "rgb":
        summary = json.loads((repo / "flight_metadata" / "zenodo_upload_summary_owl_transferred.json").read_text(encoding="utf-8"))
        have = {str(f) for part in summary for f in part["flights"]}
    else:
        have = set()
        for f in split_of:
            meta = repo / "flight_metadata" / f"{f}_metadata.json"
            if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("species_present"):
                have.add(f)
    return {f: s for f, s in split_of.items() if f in have}


# ---------------------------------------------------------------- flight sources (local / Zenodo)

class LocalFlight:
    def __init__(self, raw: Path, flight: str):
        self.raw, self.flight = raw, flight

    def member(self, name: str) -> bytes | None:
        p = self.raw / name
        return p.read_bytes() if p.exists() else None

    def video(self):
        return open(self.raw / f"{self.flight}_matched_processed.mp4", "rb")

    def close(self) -> None:
        pass


class ZenodoFlight:
    """Members of a flight's Zenodo archives, read with HTTP range requests (no full download)."""

    def __init__(self, urls: list[str]):
        self.session = requests.Session()
        self.zips = [zipfile.ZipFile(RangeFile(self.session, u)) for u in urls]

    def member(self, name: str) -> bytes | None:
        for zf in self.zips:
            for info in zf.infolist():
                if Path(info.filename).name == name:
                    return zf.read(info)
        return None

    def video(self):
        for zf in self.zips:
            for info in zf.infolist():
                if info.filename.endswith("_matched_processed.mp4"):
                    return ForwardReader(zf, info)
        raise FileNotFoundError("no video member")

    def bytes_fetched(self) -> int:
        return sum(zf.fp.fetched for zf in self.zips if hasattr(zf.fp, "fetched"))

    def close(self) -> None:
        for zf in self.zips:
            zf.close()
        self.session.close()


class ForwardReader(io.RawIOBase):
    """Seekable-looking view of a deflated zip member that only ever decompresses forward.

    Forward seeks decompress and discard; short backward seeks are served from a ring buffer of
    the last 64 MB; anything else re-opens the member (counted in ``reopens``). Seeking to the
    end only records the position, so asking for the file size costs nothing.
    """

    KEEP = 64 * 1024 * 1024

    def __init__(self, zf: zipfile.ZipFile, info: zipfile.ZipInfo):
        self.zf, self.info, self.size = zf, info, info.file_size
        self.reopens = 0
        self._open()

    def _open(self) -> None:
        self.src = self.zf.open(self.info)
        self.done = 0               # bytes decompressed so far
        self.buf = bytearray()      # the last KEEP bytes decompressed
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = offset if whence == io.SEEK_SET else self.pos + offset if whence == io.SEEK_CUR else self.size + offset
        return self.pos

    def _fill_to(self, end: int) -> None:
        while self.done < end:
            chunk = self.src.read(min(8 * 1024 * 1024, end - self.done))
            if not chunk:
                break
            self.buf += chunk
            self.done += len(chunk)
        if len(self.buf) > self.KEEP:
            del self.buf[: len(self.buf) - self.KEEP]

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        start_of_buf = self.done - len(self.buf)
        if self.pos < start_of_buf:  # too far back: start over
            self.reopens += 1
            want = self.pos
            self.src.close()
            self._open()
            self.pos = want
            start_of_buf = 0
        self._fill_to(self.pos + n)
        start_of_buf = self.done - len(self.buf)
        data = bytes(self.buf[self.pos - start_of_buf: self.pos - start_of_buf + n])
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self.src.close()
        super().close()


class ZenodoIndex:
    """flight id -> archive URLs for the layers this modality needs (deposition listings cached)."""

    def __init__(self, repo: Path, modality: str):
        sys.path.insert(0, str(repo))
        import download_from_zenodo as d  # BAMBI's own downloader: summaries and file listings
        self.d = d
        self.layers = [d.build_flight_index(d.load_summary(d.VERSION_SUMMARIES["base"]))]
        if modality == "rgb":
            self.layers.append(d.build_flight_index(d.load_summary(d.VERSION_SUMMARIES["owl-transferred"])))
        self.files: dict[int, dict[str, str]] = {}
        self.lock = threading.Lock()

    def urls(self, flight: str) -> list[str]:
        out = []
        for layer in self.layers:
            entry = layer[flight]
            dep = entry["deposition_id"]
            with self.lock:
                if dep not in self.files:
                    self.files[dep] = self.d.get_deposition_files(self.d.ZENODO_API, dep, None)
            out.append(self.files[dep][entry["zip_name"]])
        return out[::-1]  # annotation layer first: its rgb_gt.txt must win over nothing


# ---------------------------------------------------------------- annotations -> frames

def read_mot(text: str) -> dict[int, list[list[str]]]:
    rows: dict[int, list[list[str]]] = defaultdict(list)
    for row in csv.reader(io.StringIO(text)):
        if row and row[12].strip() == "0":  # key frames only
            rows[int(row[0])].append(row)
    return rows


def read_provenance(text: str | None) -> dict[tuple[int, int], str]:
    if not text:
        return {}
    return {(int(r["frame"]), int(r["track_id"])): r["source"] for r in csv.DictReader(io.StringIO(text))}


def frame_boxes(rows, cfg, prov, mask, min_visible, min_rgb_detected) -> tuple[list[tuple[int, float, float, float, float]] | None, str]:
    """YOLO boxes for one frame, or (None, reason) if the frame must be skipped."""
    out = []
    seen_in_rgb = 0
    for r in rows:
        species = r[9].strip()
        if species in cfg["no_animal_species"]:
            continue
        if species not in cfg["species_to_class"]:
            return None, f"ignored species: {species}"
        source = prov.get((int(r[0]), int(r[1])))
        if source == "unchanged":
            return None, "box not transferred to RGB"
        x, y, w, h = map(float, r[2:6])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(HALF), x + w), min(float(HALF), y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # entirely outside the frame
        if mask is not None:
            inside = mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))].mean() / 255.0
            if inside < min_visible:  # animal mostly in the black border: drop the box, keep the frame
                continue
        seen_in_rgb += source in ("owl", "smoothed")
        cls = cfg["species_to_class"][species]
        out.append((cls, (x1 + x2) / 2 / HALF, (y1 + y2) / 2 / HALF, (x2 - x1) / HALF, (y2 - y1) / HALF))
    if prov and out and seen_in_rgb / len(out) < min_rgb_detected:
        return None, "animals not detectable in RGB"
    return out, ""


def pick_frames(valid: list[int], min_gap: int, cap: int) -> list[int]:
    """First ``cap`` frames, each at least ``min_gap`` after the previous pick (earliest first, so
    a streamed video can stop early)."""
    picked: list[int] = []
    for f in sorted(valid):
        if not picked or f - picked[-1] >= min_gap:
            picked.append(f)
            if len(picked) == cap:
                break
    return picked


def decode_frames(fh, wanted: list[int]) -> dict[int, np.ndarray]:
    """Decode the video sequentially and return the wanted 0-based frame indices (BGR)."""
    wanted_set, last, got = set(wanted), max(wanted), {}
    with av.open(fh, mode="r") as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i in wanted_set:
                got[i] = fr.to_ndarray(format="bgr24")
            if i >= last:
                break
    return got


# ---------------------------------------------------------------- one flight

def process_flight(flight: str, split: str, args, cfg, names, modality, index: ZenodoIndex) -> dict:
    t0 = time.time()
    local = args.raw is not None and (args.raw / f"{flight}_matched_processed.mp4").exists()
    src = LocalFlight(args.raw, flight) if local else ZenodoFlight(index.urls(flight))
    f_boxes, f_skipped, f_review, kept = Counter(), Counter(), [], []
    try:
        ann_name = f"{flight}_rgb_gt.txt" if modality == "rgb" else f"{flight}_gt.txt"
        ann = src.member(ann_name)
        if ann is None:
            raise FileNotFoundError(ann_name)
        rows = read_mot(ann.decode("utf-8"))
        prov_bytes = src.member(f"{flight}_provenance.csv") if modality == "rgb" else None
        prov = read_provenance(prov_bytes.decode("utf-8") if prov_bytes else None)
        mask_bytes = src.member(f"{flight}_mask_{'w' if modality == 'rgb' else 't'}.png")
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE) if mask_bytes else None

        labelled = {}
        for f, fr in rows.items():
            b, why = frame_boxes(fr, cfg, prov, mask, args.min_visible, args.min_rgb_detected)
            if b is None:
                f_skipped[why] += 1
            elif b:
                labelled[f] = b
        chosen = pick_frames(list(labelled), args.min_gap, args.max_per_flight)

        if chosen:
            fh = src.video()
            try:
                frames = decode_frames(fh, chosen)
            finally:
                fh.close()
            x0 = HALF if modality == "rgb" else 0
            for f in chosen:
                if f not in frames:
                    f_skipped["unreadable frame"] += 1
                    continue
                img = np.ascontiguousarray(frames[f][:, x0:x0 + HALF])
                stem = f"bambi{flight}_{f:08d}"
                cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labelled[f]]
                (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                for c, *_ in labelled[f]:
                    f_boxes[names[c]] += 1
                kept.append(f)
                if args.review and split == "test":
                    vis = img.copy()
                    for c, cx, cy, w, h in labelled[f]:
                        p1 = (int((cx - w / 2) * HALF), int((cy - h / 2) * HALF))
                        p2 = (int((cx + w / 2) * HALF), int((cy + h / 2) * HALF))
                        cv2.rectangle(vis, p1, p2, COLOURS[c % len(COLOURS)], 2)
                        cv2.putText(vis, names[c], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[c % len(COLOURS)], 1)
                    cv2.imwrite(str(args.out / "review" / f"{stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    f_review.append({"image": f"{stem}.jpg", "flight": flight, "frame": f, "boxes": len(labelled[f]),
                                     "verified": "", "missed_animals": "", "wrong_boxes": "", "notes": ""})
        mb = src.bytes_fetched() / 1e6 if isinstance(src, ZenodoFlight) else 0.0
    finally:
        src.close()
    return {"kept": kept, "boxes": f_boxes, "skipped": f_skipped, "review": f_review,
            "mb": round(mb, 1), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    modality = args.modality or cfg.get("modality", "rgb")
    names = cfg["names"]
    flights = eligible_flights(args.bambi_repo, modality)
    if args.flights:
        missing = [f for f in args.flights if f not in flights]
        if missing:
            sys.exit(f"not eligible for {modality} (no split or no annotations): {missing}")
        flights = {f: flights[f] for f in args.flights}
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.review:
        (args.out / "review").mkdir(parents=True, exist_ok=True)
    done_dir = args.out / ".done"  # one record per finished flight, so an interrupted run resumes
    done_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    todo = []
    for flight in sorted(flights, key=int):
        rec_file = done_dir / f"{flight}.json"
        if rec_file.exists():
            records[flight] = json.loads(rec_file.read_text(encoding="utf-8"))
        else:
            todo.append(flight)
    print(f"{len(records)} flights done earlier, {len(todo)} to go, {args.workers} at a time", flush=True)

    index = ZenodoIndex(args.bambi_repo, modality)
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process_flight, f, flights[f], args, cfg, names, modality, index): f for f in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            flight = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # network or decode failure: leave it for a re-run
                failed[flight] = f"{type(e).__name__}: {e}"
                print(f"[{n}/{len(todo)}] flight {flight}: FAILED ({failed[flight][:120]})", flush=True)
                continue
            records[flight] = rec
            (done_dir / f"{flight}.json").write_text(json.dumps(rec), encoding="utf-8")
            print(f"[{n}/{len(todo)}] flight {flight} ({flights[flight]}): {len(rec['kept'])} frames, "
                  f"{rec['mb']} MB, {rec['seconds']} s", flush=True)

    manifest: dict[str, dict[str, list[int]]] = {"train": {}, "val": {}, "test": {}}
    boxes = {s: Counter() for s in manifest}
    skipped: Counter = Counter()
    review_rows = []
    for flight in sorted(records, key=int):
        rec, split = records[flight], flights[flight]
        if rec["kept"]:
            manifest[split][flight] = rec["kept"]
        boxes[split].update(rec["boxes"])
        skipped.update(rec["skipped"])
        review_rows.extend(rec["review"])

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: [{', '.join(names)}]\n", encoding="utf-8")
    if review_rows:
        with (args.out / "review" / "test_checklist.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(review_rows[0]))
            w.writeheader()
            w.writerows(review_rows)
    summary = {
        "source": "BAMBI (Praschl et al., NeurIPS 2026 Datasets Track), CC-BY-4.0; "
                  + ("RGB boxes: owl-transferred layer (machine-generated)" if modality == "rgb" else "thermal boxes: base layer"),
        "modality": modality,
        "split_unit": "flight (official BAMBI flight_metadata/splits.json)",
        "frame_rule": {"key_frames_only": True, "first_n_per_flight": args.max_per_flight, "min_gap_frames": args.min_gap,
                       "min_visible": args.min_visible, "min_rgb_detected": args.min_rgb_detected},
        "names": names,
        "counts": {s: {"flights_with_frames": len(manifest[s]), "images": sum(map(len, manifest[s].values())),
                       "boxes": dict(boxes[s])} for s in manifest},
        "skipped_frames": dict(skipped),
        "failed_flights": failed,
        "frames": manifest,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("counts", "skipped_frames", "failed_flights")}, indent=2))
    if failed:
        print(f"{len(failed)} flight(s) failed; run the same command again to retry them.")


if __name__ == "__main__":
    main()
