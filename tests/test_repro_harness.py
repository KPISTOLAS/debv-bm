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
from evaluate_faithfulness import (  # noqa: E402
    apply_mask_in_bbox,
    average_drop,
    cluster_bootstrap,
    heatmap_box_iou,
    random_mask_like,
    salient_mask_roi,
)
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


class TestFaithfulnessControls(unittest.TestCase):
    def test_random_mask_matches_area(self):
        mask = np.zeros((12, 9), bool)
        mask.ravel()[:22] = True
        r = random_mask_like(mask, np.random.default_rng(0))
        self.assertEqual(int(r.sum()), 22)

    def test_drop_is_channel_order_agnostic(self):
        img = np.random.default_rng(1).integers(0, 255, (8, 8, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), bool)
        mask[0, 0] = True
        a = apply_mask_in_bbox(img, (2, 2, 6, 6), mask, "drop")
        b = apply_mask_in_bbox(img[..., ::-1].copy(), (2, 2, 6, 6), mask, "drop")[..., ::-1]
        np.testing.assert_array_equal(a, b)

    def test_cluster_bootstrap_resamples_images(self):
        rows = [{"image": "a", "v": 1.0}, {"image": "a", "v": 1.0}, {"image": "b", "v": 0.0}]
        ci = cluster_bootstrap(rows, lambda r: float(np.mean([x["v"] for x in r])), 500, 0)
        self.assertEqual(ci[0], 0.0)
        self.assertEqual(ci[1], 1.0)


class TestCamWeights(unittest.TestCase):
    def test_gradcampp_matches_pytorch_grad_cam(self):
        try:
            import torch
            from pytorch_grad_cam import GradCAMPlusPlus
        except ImportError:
            self.skipTest("torch / grad-cam not installed")
        from yolo_cam import cam_weights

        rng = np.random.default_rng(0)
        g = rng.normal(size=(1, 6, 5, 5)).astype(np.float32)
        a = np.abs(rng.normal(size=(1, 6, 5, 5))).astype(np.float32)
        ref = GradCAMPlusPlus.get_cam_weights(None, None, None, None, a, g)[0]
        ours = cam_weights(torch.from_numpy(g[0]), torch.from_numpy(a[0]), "gradcampp").numpy()
        np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-6)


class TestGroupedBootstrap(unittest.TestCase):
    def test_group_patterns(self):
        try:
            from eval_folder_classifier import GROUP_PATTERNS, group_ids
        except ImportError:
            self.skipTest("classifier deps not installed")
        pf = [Path("Pinus_halepensis-C8-115_1_128.jpg"), Path("Pinus_halepensis-C8-115_1_148.jpg"),
              Path("Pinus_halepensis-C8-264_0_3.jpg")]
        self.assertEqual(list(group_ids(pf, GROUP_PATTERNS["pureforest"])), [0, 0, 1])
        mc = [Path("mascherargb_immagine53_oggetto_0classe-acer.jpg"), Path("mascherargb_immagine53_oggetto_1classe-oak.jpg"),
              Path("mascherargb_immagine6_oggetto_0classe-acer.jpg")]
        ids = group_ids(mc, GROUP_PATTERNS["marche"])
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])


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
