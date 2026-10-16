import unittest

from qwen_vl.dualgeo.training import route_for_sample
from qwen_vl.dualgeo.types import Route


class M1TrainingContractsTest(unittest.TestCase):
    def test_explicit_route_schedules_are_constant(self):
        self.assertEqual(route_for_sample("a", "self", 0), Route.SELF)
        self.assertEqual(route_for_sample("a", "seek", 0), Route.SEEK)

    def test_joint_schedule_is_deterministic_and_uses_both_routes(self):
        first = [route_for_sample(f"sample-{index}", "balanced_hash", 7) for index in range(64)]
        second = [route_for_sample(f"sample-{index}", "balanced_hash", 7) for index in range(64)]
        self.assertEqual(first, second)
        self.assertIn(Route.SELF, first)
        self.assertIn(Route.SEEK, first)

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            route_for_sample("a", "random", 0)
