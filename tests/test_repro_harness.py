"""Lightweight tests: no GPU, no fine-tuned weights."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge_pi5"))
sys.path.insert(0, str(ROOT / "tier2"))
sys.path.insert(0, str(ROOT / "train"))

import capture_preprocess as edge  # noqa: E402
from dedup import iou  # noqa: E402
from evaluate_faithfulness import average_drop, heatmap_box_iou, salient_mask_roi  # noqa: E402
from infer_pipeline import reshape_transform_vit, unwrap_state_dict, vit_target_layer  # noqa: E402


class TestExamples(unittest.TestCase):
    def test_waid_examples_exist(self):
        ex = ROOT / "examples"
        jpgs = list(ex.glob("*.jpg")) + list(ex.glob("*.png"))
        self.assertGreaterEqual(len(jpgs), 3, "examples/ must ship frames for quick-start")

    def test_examples_pass_qc(self):
        self.assertIsNotNone(edge.cv2)
        for path in sorted((ROOT / "examples").iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = edge.cv2.imread(str(path))
            self.assertIsNotNone(img, path.name)
            ok, reason, _ = edge.qc_ok(img, 60.0, 12.0, 245.0)
            self.assertTrue(ok, f"{path.name} failed QC ({reason})")


class TestEdgeQuickStart(unittest.TestCase):
    def test_preprocess_writes_outbox(self):
        self.assertIsNotNone(edge.cv2)
        with tempfile.TemporaryDirectory() as tmp:
            outbox = Path(tmp) / "outbox"
            buffer = Path(tmp) / "buffer"
            args = type("A", (), {})()
            args.outbox = outbox
            args.buffer = buffer
            args.max_side = 1280
            args.blur_thr = 60.0
            args.dark_thr = 12.0
            args.bright_thr = 245.0
            args.lat = None
            args.lon = None
            args.thermal = False
            img = edge.cv2.imread(str(next((ROOT / "examples").glob("waid_*.jpg"))))
            self.assertIsNotNone(img)
            edge.process_bgr(img, args, source="test")
            self.assertTrue(list(outbox.glob("*.jpg")))
            self.assertTrue(list(outbox.glob("*.json")))


class TestDedup(unittest.TestCase):
    def test_iou_identical(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_iou_disjoint(self):
        self.assertEqual(iou([0, 0, 1, 1], [5, 5, 6, 6]), 0.0)


class TestFaithfulnessMath(unittest.TestCase):
    def test_average_drop(self):
        self.assertAlmostEqual(average_drop(1.0, 0.721), 27.9, places=1)

    def test_top20_iou_is_keep_rate(self):
        hm = np.zeros((10, 10), np.float32)
        hm.ravel()[:20] = 1.0
        mask = salient_mask_roi(hm, 0.2)
        self.assertEqual(int(mask.sum()), 20)
        self.assertAlmostEqual(heatmap_box_iou(hm, 0.2), 0.2, places=5)


class TestInferHelpers(unittest.TestCase):
    def test_unwrap_module_prefix(self):
        sd = unwrap_state_dict({"module.visual.weight": 1})
        self.assertIn("visual.weight", sd)

    def test_vit_target_layer(self):
        class D:
            pass

        m = D()
        m.visual = D()
        m.visual.transformer = D()
        block = D()
        block.ln_1 = object()
        m.visual.transformer.resblocks = [block]
        self.assertIs(vit_target_layer(m), block.ln_1)

    def test_reshape_transform_vit(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        t = torch.randn(1, 197, 8)
        out = reshape_transform_vit(t)
        self.assertEqual(tuple(out.shape), (1, 8, 14, 14))


if __name__ == "__main__":
    unittest.main()
