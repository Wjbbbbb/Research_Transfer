import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from qwen_vl.dualgeo.config import FastAdapterConfig
from qwen_vl.dualgeo.curriculum import CurriculumOrder, build_curriculum
from qwen_vl.dualgeo.fast import FastSpatialAdapter
from qwen_vl.dualgeo.metrics import brier_score, expected_calibration_error, risk_coverage_curve
from qwen_vl.dualgeo.splits import split_mindcube_samples, write_mindcube_splits
from qwen_vl.dualgeo.types import Route


class M1FastPrimitivesTest(unittest.TestCase):
    def test_fast_adapter_has_expected_shapes_and_no_geometry_modules(self):
        adapter = FastSpatialAdapter(FastAdapterConfig(hidden_size=8, bottleneck_size=4, num_predicates=3))
        state = adapter(
            torch.randn(2, 5, 8),
            torch.randn(2, 4, 8),
            visual_mask=torch.tensor([[True, True, False, False, False], [True, True, True, True, True]]),
            question_mask=torch.ones(2, 4, dtype=torch.bool),
        )
        self.assertEqual(state.question_state.shape, (2, 8))
        self.assertEqual(state.spatial_state.shape, (2, 8))
        self.assertEqual(state.risk_logit.shape, (2,))
        self.assertEqual(state.predicate_logits.shape, (2, 3))
        self.assertFalse(any("geometry" in name.lower() or "vggt" in name.lower() for name, _ in adapter.named_modules()))

    def test_fast_adapter_rejects_empty_mask_rows(self):
        adapter = FastSpatialAdapter(FastAdapterConfig(hidden_size=4, bottleneck_size=2))
        with self.assertRaisesRegex(ValueError, "visual row"):
            adapter(torch.randn(1, 2, 4), torch.randn(1, 2, 4), visual_mask=torch.zeros(1, 2, dtype=torch.bool))

    def test_curriculum_orders_are_opposites_with_shared_consolidation(self):
        self_first = build_curriculum(CurriculumOrder.SELF_THEN_SEEK)
        seek_first = build_curriculum("seek_then_self")
        self.assertEqual((self_first.phase_one, self_first.phase_two), (Route.SELF, Route.SEEK))
        self.assertEqual((seek_first.phase_one, seek_first.phase_two), (Route.SEEK, Route.SELF))
        self.assertEqual(self_first.consolidation_routes, seek_first.consolidation_routes)

    def test_calibration_metrics(self):
        probability = torch.tensor([0.1, 0.2, 0.9, 0.8])
        error = torch.tensor([0, 0, 1, 1])
        self.assertGreaterEqual(brier_score(probability, error).item(), 0)
        self.assertGreaterEqual(expected_calibration_error(probability, error, bins=2).item(), 0)
        coverage, risk = risk_coverage_curve(probability, error)
        self.assertTrue(torch.allclose(coverage, torch.tensor([0.25, 0.5, 0.75, 1.0])))
        self.assertTrue(torch.allclose(risk, torch.tensor([0.0, 0.0, 1 / 3, 0.5])))

    def test_mindcube_split_keeps_a_frame_group_together_and_refuses_overwrite(self):
        samples = [
            {"id": "a", "images": ["MindCube/around/g1/1.png", "MindCube/around/g1/2.png"]},
            {"id": "b", "images": ["MindCube/around/g1/1.png", "MindCube/around/g1/3.png"]},
            {"id": "c", "images": ["MindCube/around/g2/1.png"]},
        ]
        split_samples, _ = split_mindcube_samples(samples, seed=7)
        locations = {sample["id"]: split for split, rows in split_samples.items() for sample in rows}
        self.assertEqual(locations["a"], locations["b"])
        with TemporaryDirectory() as directory:
            source = Path(directory) / "mindcube.json"
            source.write_text(json.dumps(samples), encoding="utf-8")
            destination = Path(directory) / "splits"
            manifest = write_mindcube_splits(source, destination, seed=7)
            self.assertEqual(sum(manifest["samples"].values()), 3)
            with self.assertRaises(FileExistsError):
                write_mindcube_splits(source, destination, seed=7)
