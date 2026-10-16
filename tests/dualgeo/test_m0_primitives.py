import json
import unittest

from qwen_vl.dualgeo.cache import append_records, read_records
from qwen_vl.dualgeo.collect import normalize_answer
from qwen_vl.dualgeo.inputs import validate_m0_frames
from qwen_vl.dualgeo.profiling import profile_call
from qwen_vl.dualgeo.types import CounterfactualRecord, InferenceProfile, Route


def make_record(route=Route.SELF):
    return CounterfactualRecord(
        sample_id="example-001",
        route=route,
        frame_ids=(0, 8, 16, 24),
        answer="left",
        loss=0.2,
        correct=True,
        profile=InferenceProfile(wall_time_ms=1.0, peak_memory_mb=None),
        checkpoint_id="unit-test",
    )


class M0PrimitivesTest(unittest.TestCase):
    def test_counterfactual_cache_round_trip(self):
        with self.subTest("round trip"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                path = f"{directory}/records.jsonl"
                self.assertEqual(append_records(path, [make_record(), make_record(Route.SEEK)]), 2)
                self.assertEqual(list(read_records(path)), [make_record(), make_record(Route.SEEK)])

    def test_counterfactual_cache_rejects_external_feature_leakage(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/records.jsonl"
            payload = make_record().to_dict()
            payload["vggt_features"] = "forbidden"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "unknown=.*vggt_features"):
                list(read_records(path))

    def test_profile_call_returns_result_and_nonnegative_duration(self):
        result, profile = profile_call(lambda left, right: left + right, 2, 3)
        self.assertEqual(result, 5)
        self.assertGreaterEqual(profile.wall_time_ms, 0)

    def test_profile_call_accepts_an_explicit_cuda_device(self):
        result, profile = profile_call(lambda: "ok", cuda_device="cuda:1")
        self.assertEqual(result, "ok")
        self.assertTrue(profile.peak_memory_mb is None or profile.peak_memory_mb >= 0)

    def test_m0_frame_protocol_requires_four_ordered_frames(self):
        validate_m0_frames(["a", "b", "c", "d"], [0, 8, 16, 24])
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            validate_m0_frames(["a"], [0])
        with self.assertRaisesRegex(ValueError, "ascending"):
            validate_m0_frames(["a", "b", "c", "d"], [8, 0, 16, 24])

    def test_m0_exact_match_normalizer_is_whitespace_and_case_invariant(self):
        self.assertEqual(normalize_answer("  LEFT\nSIDE "), "left side")
