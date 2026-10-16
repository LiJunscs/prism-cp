"""Directed CPU checks for vector content and whole-update packet identity."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from joint_cp.inputs import build_manifest
from two_dimensional_inputs import materialize_span
from two_dimensional_packet import TwoDimensionalPacketSource, vector_fields


class PacketTest(unittest.TestCase):
    def test_vector_lcg_matches_direct_reference(self):
        for case, seed, lengths in [("U1M", 2026101401, [1048576] * 16),
                                     ("mixed", 2026101402, [2048, 4096, 16384])]:
            manifest = build_manifest(lengths, case, 16, seed)
            for sample in manifest["samples"][:3]:
                n = sample["length"]
                positions = torch.tensor([0, 1, 254, 255, 256, 257, n // 2 - 1, n // 2, n - 1])
                actual = vector_fields(manifest, sample["sample_id"], positions)
                for key, reference_key in [("tokens", "tokens"), ("labels", "labels"),
                                            ("position_ids", "positions"), ("loss_mask", "loss_mask")]:
                    expected = [materialize_span(manifest, sample["sample_id"], int(p), int(p) + 1)[reference_key][0]
                                for p in positions]
                    self.assertEqual(actual[key].tolist(), expected)
                self.assertEqual(actual["tokens"].device.type, "cpu")
            with self.assertRaises(ValueError):
                vector_fields(manifest, sample["sample_id"], [n])
            empty = vector_fields(manifest, sample["sample_id"], [])
            self.assertTrue(all(value.numel() == 0 for value in empty.values()))

    def test_packet_order_identity_and_reference_signature(self):
        manifest = build_manifest([131072] * 128, "U128", 16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps(manifest))
            source = TwoDimensionalPacketSource(path, 3)
        self.assertEqual(len(source), 3)
        self.assertEqual(source.samples[0], source.samples[1])
        self.assertEqual(source.samples[0], source.samples[2])
        self.assertEqual([s["local_sample_id"] for s in source.samples[0]], list(range(128)))
        self.assertEqual([s["stable_sample_id"] for s in source.samples[0]], [s["sample_id"] for s in manifest["samples"]])
        self.assertEqual(sum(s["length_tokens"] for s in source.samples[0]), 16777216)
        with patch.object(source, "fields", return_value={}) as mock:
            packet = source.item(1, 32768, 0)
            self.assertEqual(len(packet), 128)
            self.assertEqual([call.args[1] for call in mock.call_args_list], [s["sample_id"] for s in manifest["samples"]])
        first = source.samples[0][0]
        n, sample_id = first["length_tokens"], first["stable_sample_id"]
        actual = source.fields(n, sample_id, 32768, 0, positions=torch.arange(20), seed=1)
        reference = materialize_span(manifest, sample_id, 0, 20)
        self.assertEqual(actual["tokens"].tolist(), reference["tokens"])
        self.assertEqual(actual["labels"].tolist(), reference["labels"])
        self.assertEqual(actual["loss_mask"].tolist(), reference["loss_mask"])
        with self.assertRaises(IndexError):
            source.item(3, 32768, 0)
        with self.assertRaises(ValueError):
            source.fields(n, sample_id, 100, 0, positions=torch.arange(20))


if __name__ == "__main__":
    unittest.main()
