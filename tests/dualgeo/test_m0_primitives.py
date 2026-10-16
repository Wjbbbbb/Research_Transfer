import json
import unittest

from qwen_vl.dualgeo.cache import append_records, read_records
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
