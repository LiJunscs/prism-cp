"""Whole-update CPU packets using this round's frozen LCG64 content rule."""

from functools import lru_cache
import json
from pathlib import Path

from mixed_global_batch import packet_collate
from two_dimensional_inputs import (
    INCREMENT, MODULUS, MULTIPLIER, SAMPLE_STRIDE, _advance, validate_manifest,
)


def _signed(value):
    value %= MODULUS
    return value if value < MODULUS // 2 else value - MODULUS


def _affine(steps):
    increment = _advance(0, steps)
    return (_advance(1, steps) - increment) % MODULUS, increment


def _coefficients(count, steps):
    """Vector affine jump coefficients; signed int64 wrap is modulo 2**64."""
    import torch
    indices = torch.arange(count, dtype=torch.int64)
    multipliers = torch.ones(count, dtype=torch.int64)
    increments = torch.zeros(count, dtype=torch.int64)
    multiplier, increment = _affine(steps)
    bit = 1
    while bit < count:
        mask = (indices & bit) != 0
        multipliers = torch.where(mask, multipliers * _signed(multiplier), multipliers)
        increments = torch.where(mask, increments * _signed(multiplier) + _signed(increment), increments)
        increment = increment * (multiplier + 1) % MODULUS
        multiplier = multiplier * multiplier % MODULUS
        bit <<= 1
    return multipliers, increments


@lru_cache(maxsize=1)
def _low_coefficients():
    return _coefficients(256, 1)


def vector_fields(manifest, sample_id, positions, device="cpu"):
    """Generate arbitrary sample positions with O(N) tensor work.

    A 256-entry low-offset table and O(max_position/256) high-offset table avoid
    a Python token loop and avoid O(N log N) full-size tensor operations. No
    CUDA generation is performed; CPU fields are copied to requested device.
    """
    import torch
    sample = next((s for s in manifest["samples"] if s["sample_id"] == sample_id), None)
    if sample is None:
        raise ValueError("unknown stable sample ID")
    positions = torch.as_tensor(positions, dtype=torch.int64, device="cpu").reshape(-1)
    if positions.numel() and (bool((positions < 0).any()) or bool((positions >= sample["length"]).any())):
        raise ValueError("sample position outside length")
    rule = manifest["content_rule"]
    initial = _advance(rule["initial_state"], sample_id * SAMPLE_STRIDE)
    offsets = positions + 1
    high = offsets // 256
    low = offsets % 256
    high_multiplier, high_increment = _coefficients(int(high.max()) + 1 if high.numel() else 1, 256)
    low_multiplier, low_increment = _low_coefficients()
    states = (_signed(initial) * high_multiplier[high] + high_increment[high]) * low_multiplier[low] + low_increment[low]
    tokens = ((states >> 32) & 0xFFFFFFFF) % rule["vocab_size"]
    next_states = states * _signed(MULTIPLIER) + _signed(INCREMENT)
    labels = ((next_states >> 32) & 0xFFFFFFFF) % rule["vocab_size"]
    return dict(tokens=tokens.to(device), labels=labels.to(device),
                position_ids=positions.to(device),
                loss_mask=(positions < sample["loss_prefix_tokens"]).float().to(device))


class TwoDimensionalPacketSource:
    def __init__(self, manifest_path, updates):
        if type(updates) is not int or updates <= 0:
            raise ValueError("updates must be a positive integer")
        self.input_manifest = json.loads(Path(manifest_path).read_text())
        validate_manifest(self.input_manifest)
        sample_rows = [dict(local_sample_id=index, logical_sample_id=sample["sample_id"],
                            stable_sample_id=sample["sample_id"], sample_uid=sample["sample_uid"],
                            length_tokens=sample["length"])
                       for index, sample in enumerate(self.input_manifest["samples"])]
        self.samples = [[dict(sample) for sample in sample_rows] for _ in range(updates)]
        self.manifest = dict(self.input_manifest)
        self.manifest["streams"] = [dict(id="TWO_DIMENSIONAL", updates=[
            dict(update=index + 1, input_tokens=self.input_manifest["input_tokens"],
                 loss_tokens=self.input_manifest["loss_tokens"], recipe_ids=[],
                 logical_sample_ids=[sample["stable_sample_id"] for sample in sample_rows],
                 repeat_same_frozen_samples=True)
            for index in range(updates)])]

    def __len__(self):
        return len(self.samples)

    def fields(self, n, logical_id, vocab, eod, device="cpu", positions=None, seed=1234):
        """Legacy PacketIntegrity.fields signature, using stable sample IDs."""
        import torch
        sample = next((s for s in self.input_manifest["samples"] if s["sample_id"] == logical_id), None)
        if sample is None or sample["length"] != n or vocab != self.input_manifest["content_rule"]["vocab_size"]:
            raise ValueError("integrity reference differs from frozen sample contract")
        if positions is None:
            positions = torch.arange(n, dtype=torch.int64)
        elif hasattr(positions, "device"):
            positions = positions.cpu()
        # eod/legacy seed intentionally do not override the frozen content rule.
        return vector_fields(self.input_manifest, logical_id, positions, device)

    def item(self, index, vocab, eod):
        import torch
        if not 0 <= index < len(self):
            raise IndexError("whole-update packet exhausted")
        return [dict(self.fields(sample["length_tokens"], sample["stable_sample_id"], vocab, eod),
                     original_seq_len=torch.tensor([sample["length_tokens"]], dtype=torch.int32),
                     padded_seq_len=torch.tensor([sample["length_tokens"]], dtype=torch.int32))
                for sample in self.samples[index]]

    def install_integrity_reference(self):
        """Reuse PacketIntegrity coverage/F-B checks with this content rule."""
        import mixed_packet_integrity
        mixed_packet_integrity.fields = self.fields


def install(manifest_path, updates, world=16):
    """Install packet dataset/loader hooks in the existing training process."""
    from megatron.training.datasets.varlen_dataset import MockPackedCPCaseDataset
    from megatron.training.datasets import data_samplers
    from megatron.training import training
    source = TwoDimensionalPacketSource(manifest_path, updates)
    if source.input_manifest.get("world",16) != world:
        raise ValueError("Packet manifest world differs from launch")
    MockPackedCPCaseDataset.__getitem__ = lambda self, index: source.item(index, self.vocab_size, self.eod)
    MockPackedCPCaseDataset.__len__ = lambda self: len(source)
    original = data_samplers.build_pretraining_data_loader

    def loader(dataset, consumed_samples):
        from megatron.training import get_args
        from megatron.core import mpu
        args = get_args()
        if args.eval_iters != 0 or args.global_batch_size != 1 or args.micro_batch_size != 1:
            raise ValueError("two-dimensional packets require eval=0 and batch sizes=1 packet")
        if mpu.get_data_parallel_world_size() != 1 or mpu.get_context_parallel_world_size() != world:
            raise ValueError("packet source requires DP1 and the declared context world")
        if dataset is None or len(dataset) == 0:
            return None
        result = original(dataset, consumed_samples)
        if result is not None:
            result.collate_fn = packet_collate
        return result

    data_samplers.build_pretraining_data_loader = loader
    if hasattr(training, "build_pretraining_data_loader"):
        training.build_pretraining_data_loader = loader
    source.install_integrity_reference()
    return source
