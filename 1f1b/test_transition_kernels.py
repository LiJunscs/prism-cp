"""CPU-oracle tests for fused TMF transition layout transforms."""

import unittest

try:
    import torch

    from transition_kernels import (
        assemble_gather_recv,
        join_scatter_recv,
        pack_gather_send,
        pack_scatter_send,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the local CPU environment")
class TransitionKernelOracleTests(unittest.TestCase):
    def test_scatter_send_pack_matches_zigzag_reference(self):
        source = torch.arange(16 * 2 * 3).reshape(16, 2, 3)
        actual = pack_scatter_send(source, 4, use_triton=False)
        view = source.view(8, 2, 2, 3)
        expected = torch.stack(
            [torch.cat([view[i], view[7 - i]], dim=0) for i in range(4)]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_scatter_receive_join_matches_batch_concatenation(self):
        received = torch.arange(4 * 4 * 2 * 3).reshape(4, 4, 2, 3)
        actual = join_scatter_recv(received, use_triton=False)
        expected = torch.cat(tuple(received.unbind(0)), dim=1)
        self.assertTrue(torch.equal(actual, expected))

    def test_gather_send_pack_matches_batch_lane_reference(self):
        source = torch.arange(4 * 8 * 3).reshape(4, 8, 3)
        actual = pack_gather_send(source, 4, use_triton=False)
        expected = torch.stack(
            [source[:, i * 2 : (i + 1) * 2].contiguous() for i in range(4)]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_gather_receive_assembly_restores_global_zigzag_order(self):
        received = torch.arange(4 * 4 * 2 * 3).reshape(4, 4, 2, 3)
        actual = assemble_gather_recv(received, use_triton=False)
        expected_view = torch.empty(8, 2, 2, 3, dtype=received.dtype)
        for piece in range(4):
            shard = received[piece].view(2, 2, 2, 3)
            expected_view[piece].copy_(shard[0])
            expected_view[7 - piece].copy_(shard[1])
        self.assertTrue(torch.equal(actual, expected_view.view(16, 2, 3)))


if __name__ == "__main__":
    unittest.main()
