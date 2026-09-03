"""Pure rank-routing tests for asynchronous CP contraction lanes."""

import unittest

try:
    from utils import _lane_communication_groups, _lane_gather_routing
except ImportError:
    _lane_communication_groups = None
    _lane_gather_routing = None


@unittest.skipIf(_lane_gather_routing is None, "PyTorch runtime is unavailable")
class LaneGatherRoutingTests(unittest.TestCase):
    def test_cp4_to_cp1_routes_one_temporal_lane_to_each_rank(self):
        ranks = [8, 9, 10, 11]
        for local_rank in range(4):
            destinations, destination_lane, receive_sources = (
                _lane_gather_routing(ranks, local_rank, 1)
            )
            self.assertEqual(destinations, tuple(ranks))
            self.assertEqual(destination_lane, local_rank)
            self.assertEqual(receive_sources, tuple(ranks))

    def test_cp4_to_cp2_forms_two_distributed_destination_lanes(self):
        ranks = [0, 1, 2, 3]
        expected = {
            0: ((0, 2), 0, (0, 1)),
            1: ((0, 2), 0, (2, 3)),
            2: ((1, 3), 1, (0, 1)),
            3: ((1, 3), 1, (2, 3)),
        }
        for local_rank, route in expected.items():
            self.assertEqual(
                _lane_gather_routing(ranks, local_rank, 2),
                route,
            )

    def test_rejects_non_contraction(self):
        with self.assertRaisesRegex(ValueError, "CP contraction"):
            _lane_gather_routing([0, 1], 0, 2)

    def test_lane_communication_group_sizes(self):
        self.assertEqual(
            _lane_communication_groups(4, 1),
            ((0,), (1,), (2,), (3,)),
        )
        self.assertEqual(
            _lane_communication_groups(4, 2),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(
            _lane_communication_groups(4, 4),
            ((0, 1, 2, 3),),
        )

    def test_lane_communication_group_rejects_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _lane_communication_groups(4, 0)
        with self.assertRaisesRegex(ValueError, "must divide"):
            _lane_communication_groups(4, 3)


if __name__ == "__main__":
    unittest.main()
