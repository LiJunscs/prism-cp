"""Opt-in Megatron executor for the complete joint IR, including MLP placement.

Reuses existing mixed GQA/GDN kernels and the unchanged layer/residual forward.
Each attention layer can change sample grouping, CP and rank placement. Ingress
ownership is restored before the GPT head, preserving labels/loss semantics.
"""

from copy import copy
from contextlib import contextmanager
from functools import wraps
import inspect

from .ir import Plan, validate_plan
from .transport import Transport


def check_mlp_balance_config(block):
    """The migrated dense tokenwise MLP cannot change stochastic/recompute work."""
    config = block.config
    if config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1:
        raise ValueError("MLP balance requires TP=PP=1")
    if config.recompute_granularity is not None or getattr(config, "sequence_parallel", False):
        raise ValueError("MLP balance requires no recomputation or sequence parallelism")
    if any(getattr(config, name, 0) != 0 for name in ("hidden_dropout", "attention_dropout", "activation_dropout")):
        raise ValueError("MLP balance requires dropout=0")
    if getattr(config, "num_moe_experts", None) or getattr(config, "mlp_chunks_for_training", 1) != 1:
        raise ValueError("MLP balance supports one dense MLP call per layer")
    for layer in block.layers:
        if getattr(layer, "hidden_dropout", 0) != 0 or any(
            getattr(layer, name, False) for name in
            ("recompute_mlp", "recompute_input_layernorm", "recompute_pre_mlp_layernorm")
        ):
            raise ValueError("MLP balance forbids layer dropout or recomputation")


@contextmanager
def mlp_placement(mlp, transfer, attention, placement):
    original = mlp.forward

    def forward(*args, **kwargs):
        if (len(args) > 1 and args[1] is not None) or kwargs.get("per_token_scale") is not None:
            raise ValueError("Joint MLP placement supports dense unscaled tokens only")
        options = dict(kwargs)
        hidden = args[0] if args else options["hidden_states"]
        hidden = transfer(hidden, attention, placement)
        if args:
            result = original(hidden, *args[1:], **options)
        else:
            options["hidden_states"] = hidden
            result = original(**options)
        output, bias = result
        return transfer(output, placement, attention), bias

    mlp.forward = forward
    try:
        yield
    finally:
        mlp.forward = original


def install(settings, replays, batch_provider):
    """Called before distributed initialization; provider(consume) returns batch/lengths."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.mixed_context_parallel import MixedCPSegment, _validate_training
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.models.gpt.gpt_model import GPTModel

    if getattr(TransformerBlock, "_two_dimensional_installed", False):
        raise RuntimeError("A two-dimensional executor is already installed")
    plans = [Plan.from_dict(row["joint_plan"]) for row in replays]
    if not plans:
        raise ValueError("Joint execution requires a compiled replay")
    group_ranks = set()
    for plan, row in zip(plans, replays):
        validate_plan(plan, dict(enumerate(row["expected_lengths"])))
        for batch in plan.microbatches:
            for layout in (batch.ingress, *(x for pair in batch.layers for x in pair)):
                group_ranks.update(g.ranks for g in layout.groups)
    group_ranks.update((rank,) for rank in range(plans[0].world))
    groups, world_group = {}, {}
    original_initialize = parallel_state.initialize_model_parallel

    @wraps(original_initialize)
    def initialize(*args, **kwargs):
        result = original_initialize(*args, **kwargs)
        if any(plan.world != dist.get_world_size() for plan in plans):
            raise ValueError("Joint replay world differs from distributed world")
        # Bootstrap only; kernels and transport receive explicit handles.
        world_group["handle"] = dist.group.WORLD
        for ranks in sorted(group_ranks, key=lambda r: (len(r), r)):
            group = dist.new_group(list(ranks))
            if dist.get_rank() in ranks:
                groups[ranks] = group
        return result

    parallel_state.initialize_model_parallel = initialize
    original_gpt = GPTModel.forward
    signature = inspect.signature(original_gpt)

    @wraps(original_gpt)
    def gpt_forward(self, *args, **kwargs):
        if self.vocab_size != settings["vocab_size"]:
            raise ValueError("Joint model vocabulary differs from cost/memory profile")
        if self.config.params_dtype != torch.bfloat16:
            raise ValueError("Joint storage model and executor require BF16 parameters")
        bound = signature.bind(self, *args, **kwargs)
        params = bound.arguments.get("packed_seq_params")
        if params is None:
            raise ValueError("Joint replay requires packed training inputs")
        batch, lengths = batch_provider(False)
        # A later layout can bring a longer sequence onto this rank. Build RoPE
        # for the global microbatch maximum before GPT prepares its embeddings.
        params = copy(params)
        maximum = max(lengths[s] for g in batch.ingress.groups for s in g.samples)
        params.max_seqlen_q = params.max_seqlen_kv = maximum
        bound.arguments["packed_seq_params"] = params
        return original_gpt(*bound.args, **bound.kwargs)

    GPTModel.forward = gpt_forward
    transports = {}

    def forward(block, hidden, params, rotary, padding):
        batch, lengths = batch_provider(True)
        world = plans[0].world
        rank = dist.get_rank(world_group["handle"])
        pattern = "".join("G" if linear else "*" for linear in block._packed_sequence_linear_pattern)
        if any(plan.pattern != pattern for plan in plans) or len(block.layers) != len(pattern):
            raise ValueError("Joint plan and actual transformer pattern differ")
        check_mlp_balance_config(block)
        for field, value in settings["model_contract"].items():
            if getattr(block.config, field, None) != value:
                raise ValueError(f"Joint model contract mismatch: {field}")
        if params.cp_partition_mode != "zigzag" or params.pad_between_seqs is not False:
            raise ValueError("Joint ingress requires unpadded zigzag THD")
        if block.config.gdn_dynamic_context_parallel or block.config.packed_sequence_cp_size is not None:
            raise ValueError("Joint IR must be the only layer placement policy")
        ingress_group = next(g for g in batch.ingress.groups if rank in g.ranks)
        if (tuple(dist.get_process_group_ranks(params.cp_group)) != ingress_group.ranks
                or params.mixed_cp_segments
                or params.cu_seqlens_q.diff().tolist() != [lengths[s] for s in ingress_group.samples]
                or hidden.shape[0] != batch.ingress.tokens(lengths, world)[rank]):
            raise ValueError("Actual input ownership differs from compiled joint ingress")
        key = tuple(sorted(lengths.items()))
        if key not in transports:
            transports.clear()
            transports[key] = Transport(lengths, world, world_group["handle"])
        transfer = transports[key]
        metadata = {}

        def view(layout):
            if layout not in metadata:
                group = next(g for g in layout.groups if rank in g.ranks)
                boundaries = [0]
                for sample in group.samples:
                    boundaries.append(boundaries[-1] + lengths[sample])
                cu = torch.tensor(boundaries, dtype=torch.int32, device=hidden.device)
                maximum = max(lengths[s] for s in group.samples)
                inner = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                    cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu,
                    max_seqlen_q=maximum, max_seqlen_kv=maximum,
                    local_cp_size=len(group.ranks), cp_group=groups[group.ranks],
                    cp_partition_mode=layout.mode, pad_between_seqs=False)
                n = boundaries[-1] // len(group.ranks)
                outer = copy(inner)
                local = torch.tensor([0, n], dtype=torch.int32, device=hidden.device)
                outer.cu_seqlens_q = outer.cu_seqlens_kv = local
                outer.cu_seqlens_q_padded = outer.cu_seqlens_kv_padded = local
                outer.cp_group, outer.local_cp_size = groups[(rank,)], 1
                outer.mixed_cp_segments = (MixedCPSegment(0, n, inner),)
                mask = torch.zeros((1, n), dtype=torch.bool, device=hidden.device)
                metadata[layout] = outer, mask
            return metadata[layout]

        previous = batch.ingress
        for layer, (attention, placement) in zip(block.layers, batch.layers):
            hidden = transfer(hidden, previous, attention)
            layer_params, mask = view(attention)
            _validate_training(block, hidden, layer_params)
            with mlp_placement(layer.mlp, transfer, attention, placement):
                hidden, _ = layer(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rotary,
                                  packed_seq_params=layer_params, padding_mask=mask)
            previous = attention
        return transfer(hidden, previous, batch.ingress)

    TransformerBlock._parent_contiguous_forward = forward
    TransformerBlock._two_dimensional_installed = True
