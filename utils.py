# utils.py
"""Common utilities for Attention/GDN CP benchmarks."""

import argparse
import os
import random
import sys
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist

from transition_kernels import (
    assemble_gather_recv,
    join_scatter_recv,
    pack_gather_send,
    pack_scatter_send,
)

sys.path.insert(0, "/workspace/repos/Megatron-LM")
warnings.filterwarnings("ignore", category=UserWarning)


def add_common_benchmark_args(parser):
    """Add model, backend, GDN, and timing arguments shared by benchmarks."""
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument(
        "--attention-backend",
        default="flash",
        choices=["flash", "fused", "unfused", "auto"],
    )
    parser.add_argument("--gdn-key-head-dim", type=int, default=128)
    parser.add_argument("--gdn-value-head-dim", type=int, default=128)
    parser.add_argument("--gdn-num-key-heads", type=int, default=32)
    parser.add_argument("--gdn-num-value-heads", type=int, default=32)
    parser.add_argument("--gdn-conv-kernel-dim", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    return parser


def parse_benchmark_args(extra_args_provider=None, argv=None, description=None):
    """Parse common arguments plus arguments supplied by one benchmark entrypoint.

    Args:
        extra_args_provider: Optional callable that receives and returns the
            shared ``ArgumentParser`` after adding entrypoint-specific flags.
        argv: Optional argument sequence. ``None`` reads from ``sys.argv``.
        description: Optional command-line description.
    """
    parser = argparse.ArgumentParser(description=description)
    add_common_benchmark_args(parser)
    if extra_args_provider is not None:
        updated_parser = extra_args_provider(parser)
        if updated_parser is not None:
            parser = updated_parser
    return parser.parse_args(argv)


def make_layer_types(num_attention_layers, num_gdn_layers, seed):
    """Build a reproducibly randomized schedule with exact module counts."""
    if num_attention_layers < 0 or num_gdn_layers < 0:
        raise ValueError("Layer counts must be non-negative.")
    layer_types = ["attention"] * num_attention_layers + ["gdn"] * num_gdn_layers
    if not layer_types:
        raise ValueError("At least one layer is required.")
    random.Random(seed).shuffle(layer_types)
    return layer_types


def make_layer_plan(layer_types, args, mode):
    """Return ``(layer_type, cp_size)``; DP is implicitly ``world_size / CP``."""
    if mode == "fixed":
        fixed_cp = args.fixed_cp_size or max(args.attn_cp_size, args.gdn_cp_size)
        return [(layer_type, fixed_cp) for layer_type in layer_types]
    return [
        (layer_type, args.attn_cp_size if layer_type == "attention" else args.gdn_cp_size)
        for layer_type in layer_types
    ]


def validate_plan(layer_plan, args, world_size):
    """Validate the implicit DP x CP layouts handled by the P2P autograd path."""
    if args.global_batch_size < 1:
        raise ValueError("global_batch_size must be positive.")

    for _, cp in layer_plan:
        if cp < 1 or world_size % cp:
            raise ValueError(f"CP size {cp} must divide world_size={world_size}.")
        if args.seq_length % (2 * cp) or args.seq_length // cp < 128:
            raise ValueError(f"Invalid seq_length={args.seq_length} for CP={cp}.")
        dp = world_size // cp
        if args.global_batch_size % dp:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} must be divisible by "
                f"DP={dp} for CP={cp} and world_size={world_size}."
            )

    transitions = [
        (src_cp, dst_cp)
        for (_, src_cp), (_, dst_cp) in zip(layer_plan, layer_plan[1:])
        if src_cp != dst_cp
    ]
    if transitions:
        first_dp = world_size // layer_plan[0][1]
        local_batch = args.global_batch_size // first_dp
        for src_cp, dst_cp in transitions:
            large_cp, small_cp = max(src_cp, dst_cp), min(src_cp, dst_cp)
            if large_cp % small_cp:
                raise ValueError(f"CP transition {src_cp}->{dst_cp} requires nested CP sizes.")
            ratio = large_cp // small_cp
            if dst_cp < src_cp:
                if local_batch % ratio:
                    raise ValueError(
                        f"DP{world_size // src_cp}-CP{src_cp} -> "
                        f"DP{world_size // dst_cp}-CP{dst_cp} requires local microbatch "
                        f"{local_batch} to be divisible by {ratio}."
                    )
                local_batch //= ratio
            else:
                local_batch *= ratio

        if DEBUG_CP:
            supported = {(1, 4), (4, 1), (1, 8), (8, 1), (2, 4), (4, 2), (2, 8), (8, 2), (4, 8), (8, 4)}
            unsupported = [f"{src}->{dst}" for src, dst in transitions if (src, dst) not in supported]
            max_cp = max(cp for _, cp in layer_plan)
            if unsupported:
                raise ValueError(
                    "DEBUG_CP uses fixed global-rank direction tables. It requires "
                    "world_size == max CP and supports only 1<->4, 2<->4, 2<->8, and 4<->8."
                )


def describe_plan(layer_plan):
    """Return compact layer, layout, and transition descriptions."""
    world_size = dist.get_world_size()
    schedule = "-".join("A" if layer_type == "attention" else "G" for layer_type, _ in layer_plan)
    layout_plan = "-".join(f"DP{world_size // cp}-CP{cp}" for _, cp in layer_plan)
    transitions = sum(
        src_cp != dst_cp for (_, src_cp), (_, dst_cp) in zip(layer_plan, layer_plan[1:])
    )
    return schedule, layout_plan, transitions


def build_stack(args, layer_plan, tp_group, cp_cache):
    """Build an Attention/GDN stack with CP transitions between layouts."""
    layers = []
    for layer_type, cp in layer_plan:
        cp_group = cp_cache[cp][0]
        if layer_type == "attention":
            fn = build_attn_te(
                args.hidden_size,
                args.num_attention_heads,
                args.num_query_groups,
                cp,
                tp_group,
                cp_group,
            )
        else:
            fn = build_gdn_te(
                args.hidden_size,
                args.num_attention_heads,
                args.num_query_groups,
                cp,
                tp_group,
                cp_group,
                args,
            )
        layers.append((layer_type, cp, fn))

    def forward(x):
        cur_cp = layers[0][1]
        for _, cp, fn in layers:
            if cp != cur_cp:
                x = transmition_cp_size(x, cur_cp, cp, cp_cache)
                cur_cp = cp
            x = fn(x)
        return x

    return forward, layers


class BenchmarkGradBuffer:
    """Accumulate benchmark parameter gradients into one contiguous buffer."""

    def __init__(self, params, sync_group, sync_size):
        if not params:
            raise ValueError("The benchmark stack has no trainable parameters.")

        device = params[0].device
        dtype = params[0].dtype
        for param in params:
            if param.device != device or param.dtype != dtype:
                raise ValueError(
                    "BenchmarkGradBuffer requires all parameters to share one device and dtype."
                )

        self.params = params
        self.sync_group = sync_group
        self.sync_size = sync_size
        self.grad_data = torch.zeros(
            sum(param.numel() for param in params), device=device, dtype=dtype
        )
        self._grad_accs = []
        self._hook_handles = []

        offset = 0
        for param in self.params:
            next_offset = offset + param.numel()
            param.main_grad = self.grad_data[offset:next_offset].view_as(param)
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            handle = grad_acc.register_hook(self._make_param_hook(param))
            self._grad_accs.append(grad_acc)
            self._hook_handles.append(handle)
            offset = next_offset

    @staticmethod
    def _make_param_hook(param):
        def param_hook(*unused):
            if param.grad is not None:
                param.main_grad.add_(param.grad.data)
                param.grad = None

        return param_hook

    def zero_grad(self):
        """Clear the whole accumulation buffer before one backward pass."""
        self.grad_data.zero_()
        for param in self.params:
            param.grad = None

    def all_reduce(self):
        """Synchronize and average every parameter gradient in one collective."""
        if self.sync_size > 1:
            dist.all_reduce(self.grad_data, group=self.sync_group)
            self.grad_data.div_(self.sync_size)

    def close(self):
        """Remove temporary benchmark hooks so completed plans can release memory."""
        for handle in self._hook_handles:
            handle.remove()
        for param in self.params:
            param.grad = None
            if hasattr(param, "main_grad"):
                del param.main_grad
        self._hook_handles.clear()
        self._grad_accs.clear()


def benchmark_parameters(layers, world_size):
    """Collect trainable parameters and install their contiguous grad buffer."""
    params = []
    seen = set()
    for _, _, fn in layers:
        for module in getattr(fn, "_benchmark_modules", ()):
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
    return BenchmarkGradBuffer(params, dist.group.WORLD, world_size)


def synchronize_parameters(params):
    """Make the benchmark start from one replicated model state."""
    for param in params:
        dist.broadcast(param.data, src=0)

# ---------- Environment & Flags ----------
_ATTN_BACKEND_ENV = {
    "flash":   {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"},
    "fused":   {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"},
    "unfused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "1"},
    "auto":    {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "1"},
}

# Keep the known 8-GPU direction tables enabled while the general rank mapping
# is under correctness investigation. Set this environment variable to 0 to
# exercise the general nested-CP path instead.
DEBUG_CP = os.getenv("MEGATRON_BENCH_DEBUG_CP_TABLES", "1") == "1"
DIRECTION_2_4 = {
    "send": [[0, 1], [2, 3], [0, 1], [2, 3], [4, 5], [6, 7], [4, 5], [6, 7]],
    "recv": [[0, 2], [0, 2], [1, 3], [1, 3], [4, 6], [4, 6], [5, 7], [5, 7]]
}
DIRECTION_2_8 = {
    "send": [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7],
             [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]],
    "recv": [[0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6],
             [1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7]],
}
DIRECTION_4_8 = {
    "send": [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1], [2, 3], [4, 5], [6, 7]],
    "recv": [[0, 4], [0, 4], [1, 5], [1, 5], [2, 6], [2, 6], [3, 7], [3, 7]],
}
DIRECTION_4_1 = {
    "send": [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6 ,7], [4, 5, 6 ,7], [4, 5, 6 ,7], [4, 5, 6 ,7]],
    "recv": [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6 ,7], [4, 5, 6 ,7], [4, 5, 6 ,7], [4, 5, 6 ,7]],
}
DIRECTION_8_1 = {
    "send": [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
    "recv": [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
}
DIRECTION = None

os.environ["NVTE_DEBUG"] = "0"
os.environ["NVTE_DEBUG_LEVEL"] = "0"


# ---------- Distributed setup ----------
def setup_dist():
    if dist.is_initialized():
        return
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("NCCL setup requires at least one CUDA device.")
    device_index = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device_index %= device_count
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    process_group_timeout = timedelta(minutes=1)
    if "RANK" in os.environ:
        dist.init_process_group(
            "nccl",
            device_id=device,
            timeout=process_group_timeout,
        )
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(
            "nccl",
            rank=0,
            world_size=1,
            device_id=device,
            timeout=process_group_timeout,
        )

    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        get_cuda_rng_tracker,
        _MODEL_PARALLEL_RNG_TRACKER_NAME,
        _DATA_PARALLEL_RNG_TRACKER_NAME,
    )
    seed = 1234
    initialize_rng_tracker(use_te_rng_tracker=True)
    tracker = get_cuda_rng_tracker()
    tracker.reset()
    torch.cuda.manual_seed(seed)
    tracker.add(_DATA_PARALLEL_RNG_TRACKER_NAME, seed)
    tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed + 2718 + dist.get_rank())


# ---------- CP/TP group creation ----------
def make_cp_tp_groups(cp_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size >= cp_size and world_size % cp_size == 0

    cp_group = None
    for start in range(0, world_size, cp_size):
        ranks = list(range(start, start + cp_size))
        grp = dist.new_group(ranks)
        if rank in ranks:
            cp_group = grp
    tp_group = None
    for i in range(world_size):
        grp = dist.new_group([i])
        if i == rank:
            tp_group = grp
    return tp_group, cp_group


def make_cp_group_cache(cp_sizes):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cache = {}
    for cp in sorted(set(cp_sizes)):
        for start in range(0, world_size, cp):
            ranks = list(range(start, start + cp))
            grp = dist.new_group(ranks)
            if rank in ranks:
                cache[cp] = (grp, ranks)
    return cache


# ---------- Sequence partitioning (Megatron style) ----------
def partition_for_cp(x_full, cp_group):
    def get_pretrain_batch_on_this_cp_rank(
        batch: dict[str, torch.Tensor], cp_group: torch.distributed.ProcessGroup
    ):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)

        METADATA_KEYS = (
            'cu_seqlens',
            'cu_seqlens_padded',
            'max_seqlen',
            'local_cp_size',
            'hybrid_cp_group',
        )

        if cp_size > 1:
            for key, val in batch.items():
                if key in METADATA_KEYS or val is None:
                    continue

                seq_dim = 2 if key == 'attention_mask' else 1

                # reshape: [..., seq_len, ...] -> [..., 2*cp_size, chunk_len, ...]
                val = val.view(
                    *val.shape[:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[seq_dim + 1:],
                )

                # zigzag 选取：用 narrow + cat 替代 index_select，避免元素数 > 2^31 时 CUDA kernel 配置溢出
                idx_a = cp_rank
                idx_b = 2 * cp_size - cp_rank - 1
                val = torch.cat(
                    [val.narrow(seq_dim, idx_a, 1), val.narrow(seq_dim, idx_b, 1)],
                    dim=seq_dim,
                )

                # 合并回 [..., 2*chunk_len, ...]
                val = val.view(*val.shape[:seq_dim], -1, *val.shape[seq_dim + 2:])
                batch[key] = val

        return batch
    x_bsh = x_full.transpose(0, 1).contiguous()
    batch = {'hidden_states': x_bsh, 'cu_seqlens': None}
    batch = get_pretrain_batch_on_this_cp_rank(batch, cp_group=cp_group)
    return batch['hidden_states'].transpose(0, 1).contiguous()


# ----------- Get Splited Pretrain Batch On this CP Rank -----------
def get_pretrain_batch_on_this_cp_rank(seq_len, batch_size, hidden_size, cp_group, device="cuda", dtype=torch.bfloat16):
    cp_size = torch.distributed.get_world_size(cp_group)
    x_split = torch.randn(seq_len // cp_size, batch_size, hidden_size, device=device, dtype=dtype)
    return x_split 

# ---------- CP transition helpers ----------
def _scatter_sequence_redistribute(
    seq_local,
    src_cp_size,
    dst_cp_size,
    cp_group_cache,
    *,
    use_triton,
):
    """Synchronous CP expansion core shared by eager and async-adjoint paths."""
    global DEBUG_CP, DIRECTION
    multiple = dst_cp_size // src_cp_size

    dst_ranks = cp_group_cache[dst_cp_size][1]
    src_ranks = cp_group_cache[src_cp_size][1]
    cur_rank = cp_group_cache[src_cp_size][0].rank()
    cur_global = src_ranks[cur_rank]
    dst_group_start = (cur_global // dst_cp_size) * dst_cp_size
    dst_local_rank = cur_global - dst_group_start

    send_chunks = pack_scatter_send(
        seq_local,
        multiple,
        use_triton=use_triton,
    )
    send_targets = []
    for i in range(multiple):
        if DEBUG_CP:
            dst_g = DIRECTION["send"][cur_global][i]
        else:
            dst_g = dst_ranks[cur_rank * multiple + i]
        send_targets.append(dst_g)

    recv_chunks = torch.empty_like(send_chunks)
    ops = []
    for i, (chunk, dst_g) in enumerate(zip(send_chunks, send_targets)):
        if dst_g != cur_global:
            ops.append(dist.P2POp(dist.isend, chunk, dst_g))

        if DEBUG_CP:
            src_g = DIRECTION["recv"][cur_global][i]
        else:
            src_g = dst_ranks[i * src_cp_size + (dst_local_rank // multiple)]
        if src_g != cur_global:
            ops.append(dist.P2POp(dist.irecv, recv_chunks[i], src_g))
        else:
            self_send_idx = send_targets.index(cur_global)
            recv_chunks[i].copy_(send_chunks[self_send_idx])

    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    return join_scatter_recv(recv_chunks, use_triton=use_triton)


def _gather_sequence_redistribute(
    seq_local,
    src_cp_size,
    dst_cp_size,
    cp_group_cache,
    *,
    use_triton,
):
    """Synchronous CP contraction core with fused local layout transforms."""
    global DEBUG_CP, DIRECTION
    multiple = src_cp_size // dst_cp_size

    src_ranks = cp_group_cache[src_cp_size][1]
    cur_rank = cp_group_cache[src_cp_size][0].rank()
    cur_global = src_ranks[cur_rank]
    src_group_start = src_ranks[0]
    src_local_rank = cur_global - src_group_start
    send_column = src_local_rank // multiple
    recv_column = src_local_rank % dst_cp_size

    send_chunks = pack_gather_send(
        seq_local,
        multiple,
        use_triton=use_triton,
    )
    recv_chunks = torch.empty_like(send_chunks)
    ops = []
    for i, chunk in enumerate(send_chunks):
        if DEBUG_CP:
            dst_g = DIRECTION["recv"][cur_global][i]
        else:
            dst_g = src_ranks[i * dst_cp_size + send_column]
        if dst_g != cur_global:
            ops.append(dist.P2POp(dist.isend, chunk, dst_g))

        if DEBUG_CP:
            src_g = DIRECTION["send"][cur_global][i]
        else:
            src_g = src_ranks[recv_column * multiple + i]
        if src_g != cur_global:
            ops.append(dist.P2POp(dist.irecv, recv_chunks[i], src_g))
        else:
            self_batch_group = src_local_rank // dst_cp_size
            recv_chunks[i].copy_(send_chunks[self_batch_group])

    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    return assemble_gather_recv(recv_chunks, use_triton=use_triton)


class _ScatterSequenceForCPTransition(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        seq_local,
        src_cp_size,
        dst_cp_size,
        cp_group_cache,
        use_triton,
    ):
        ctx.dst_cp_size = dst_cp_size
        ctx.src_cp_size = src_cp_size
        ctx.cp_group_cache = cp_group_cache
        ctx.use_triton = use_triton
        return _scatter_sequence_redistribute(
            seq_local,
            src_cp_size,
            dst_cp_size,
            cp_group_cache,
            use_triton=use_triton,
        )

    @staticmethod
    def backward(ctx, grad):
        return _GatherSequenceForCPTransition.apply(
            grad,
            ctx.dst_cp_size,
            ctx.src_cp_size,
            ctx.cp_group_cache,
            ctx.use_triton,
        ), None, None, None, None


class _GatherSequenceForCPTransition(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        seq_local,
        src_cp_size,
        dst_cp_size,
        cp_group_cache,
        use_triton,
    ):
        ctx.src_cp_size = src_cp_size
        ctx.dst_cp_size = dst_cp_size
        ctx.cp_group_cache = cp_group_cache
        ctx.use_triton = use_triton
        return _gather_sequence_redistribute(
            seq_local,
            src_cp_size,
            dst_cp_size,
            cp_group_cache,
            use_triton=use_triton,
        )

    @staticmethod
    def backward(ctx, grad):
        return _ScatterSequenceForCPTransition.apply(
            grad,
            ctx.dst_cp_size,
            ctx.src_cp_size,
            ctx.cp_group_cache,
            ctx.use_triton,
        ), None, None, None, None


class _AsyncLaneGatherAutograd(torch.autograd.Function):
    """Attach an already-enqueued lane gather to its local producer graphs."""

    @staticmethod
    def forward(
        ctx,
        received_output,
        src_cp_size,
        dst_cp_size,
        cp_group_cache,
        use_triton,
        forward_state,
        *lane_inputs,
    ):
        ctx.src_cp_size = src_cp_size
        ctx.dst_cp_size = dst_cp_size
        ctx.cp_group_cache = cp_group_cache
        ctx.use_triton = use_triton
        ctx.forward_state = forward_state
        ctx.input_count = len(lane_inputs)
        return received_output

    @staticmethod
    def backward(ctx, grad_output):
        packed_grad = _scatter_sequence_redistribute(
            grad_output,
            ctx.dst_cp_size,
            ctx.src_cp_size,
            ctx.cp_group_cache,
            use_triton=ctx.use_triton,
        )
        if packed_grad.size(1) % ctx.input_count:
            raise RuntimeError(
                f"Transition gradient batch {packed_grad.size(1)} cannot split "
                f"across {ctx.input_count} lane inputs."
            )
        lane_batch = packed_grad.size(1) // ctx.input_count
        lane_gradients = tuple(
            chunk.contiguous()
            for chunk in packed_grad.split(lane_batch, dim=1)
        )
        ctx.forward_state = None
        return (None, None, None, None, None, None, *lane_gradients)


def _lane_gather_routing(src_ranks, src_local_rank, dst_cp_size):
    """Return deterministic send/receive routing for one contraction rank."""
    src_cp_size = len(src_ranks)
    if src_cp_size <= dst_cp_size or src_cp_size % dst_cp_size:
        raise ValueError(
            f"Lane routing requires nested CP contraction, got "
            f"{src_cp_size}->{dst_cp_size}."
        )
    if not 0 <= src_local_rank < src_cp_size:
        raise ValueError(
            f"Source local rank {src_local_rank} is outside CP{src_cp_size}."
        )
    multiple = src_cp_size // dst_cp_size
    send_column = src_local_rank // multiple
    recv_column = src_local_rank % dst_cp_size
    destination_lane = src_local_rank // dst_cp_size
    destinations = tuple(
        src_ranks[lane * dst_cp_size + send_column]
        for lane in range(multiple)
    )
    receive_sources = tuple(
        src_ranks[recv_column * multiple + piece]
        for piece in range(multiple)
    )
    return destinations, destination_lane, receive_sources


def _lane_communication_groups(multiple, lane_group_size):
    """Partition contraction lanes into equally sized NCCL enqueue groups."""
    if multiple < 1:
        raise ValueError(f"Lane multiple must be positive, got {multiple}.")
    if lane_group_size < 1:
        raise ValueError(
            f"Transition lane group size must be positive, got {lane_group_size}."
        )
    if multiple % lane_group_size:
        raise ValueError(
            f"Transition lane group size {lane_group_size} must divide "
            f"contraction multiple {multiple}."
        )
    return tuple(
        tuple(range(start, start + lane_group_size))
        for start in range(0, multiple, lane_group_size)
    )


class AsyncLaneGatherTransition:
    """Enqueue CP-contraction lane groups while later producers still compute.

    ``lane_group_size=1`` preserves the original per-lane behavior. Larger
    values delay an enqueue until that many consecutive producer lanes are
    ready, allowing experiments that trade a shorter overlap window for fewer,
    larger NCCL groups. ``finish`` waits only at the consumer boundary and then
    performs the fused receive-layout write. The backward intentionally retains
    the validated synchronous CP-expansion adjoint; backward/inter-cohort
    overlap is a separate optimization.
    """

    def __init__(
        self,
        src_cp_size,
        dst_cp_size,
        cp_group_cache,
        stream,
        *,
        use_triton=True,
        lane_group_size=1,
    ):
        if src_cp_size <= dst_cp_size or src_cp_size % dst_cp_size:
            raise ValueError(
                f"Async lane gather requires nested CP contraction, got "
                f"{src_cp_size}->{dst_cp_size}."
            )
        self.src_cp_size = src_cp_size
        self.dst_cp_size = dst_cp_size
        self.cp_group_cache = cp_group_cache
        self.stream = stream
        self.use_triton = use_triton
        self.multiple = src_cp_size // dst_cp_size
        self.lane_group_size = lane_group_size
        self.lane_groups = _lane_communication_groups(
            self.multiple,
            self.lane_group_size,
        )

        src_ranks = cp_group_cache[src_cp_size][1]
        cur_rank = cp_group_cache[src_cp_size][0].rank()
        self.cur_global = src_ranks[cur_rank]
        self.src_ranks = src_ranks
        self.src_local_rank = self.cur_global - src_ranks[0]
        (
            self.destinations,
            self.destination_lane,
            self.receive_sources,
        ) = _lane_gather_routing(
            src_ranks,
            self.src_local_rank,
            dst_cp_size,
        )

        self.inputs = []
        self.communication_inputs = []
        self.pending_communication = []
        self.enqueued_group_count = 0
        self.works = []
        self.received = None
        self.output = None
        self.done_event = None

    def _enqueue_pending_group(self):
        if not self.pending_communication:
            return
        expected_lanes = self.lane_groups[self.enqueued_group_count]
        actual_lanes = tuple(lane for lane, _ in self.pending_communication)
        if actual_lanes != expected_lanes:
            raise RuntimeError(
                f"Pending transition lanes {actual_lanes} do not match "
                f"communication group {expected_lanes}."
            )

        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready_event)
            ops = []
            for lane, communication_tensor in self.pending_communication:
                destination = self.destinations[lane]
                if destination != self.cur_global:
                    ops.append(
                        dist.P2POp(dist.isend, communication_tensor, destination)
                    )

                if lane == self.destination_lane:
                    self.received = communication_tensor.new_empty(
                        self.multiple,
                        *communication_tensor.shape,
                    )
                    for piece, source in enumerate(self.receive_sources):
                        if source == self.cur_global:
                            self.received[piece].copy_(communication_tensor)
                        else:
                            ops.append(
                                dist.P2POp(
                                    dist.irecv,
                                    self.received[piece],
                                    source,
                                )
                            )

            if ops:
                self.works.extend(dist.batch_isend_irecv(ops))

        for _, communication_tensor in self.pending_communication:
            communication_tensor.record_stream(self.stream)
        self.pending_communication.clear()
        self.enqueued_group_count += 1

    def enqueue(self, tensor):
        lane = len(self.inputs)
        if lane >= self.multiple:
            raise RuntimeError(
                f"Transition already has {self.multiple} lanes; "
                f"cannot enqueue lane {lane}."
            )
        if tensor.dim() != 3:
            raise ValueError(f"Expected an SBH lane tensor, got shape {tensor.shape}.")
        if self.inputs and tensor.shape != self.inputs[0].shape:
            raise ValueError(
                f"Lane shape {tuple(tensor.shape)} differs from first lane "
                f"{tuple(self.inputs[0].shape)}."
            )
        self.inputs.append(tensor)
        communication_tensor = tensor.contiguous()
        self.communication_inputs.append(communication_tensor)
        self.pending_communication.append((lane, communication_tensor))
        if len(self.pending_communication) == self.lane_group_size:
            self._enqueue_pending_group()
        return lane

    def finish(self):
        if len(self.inputs) != self.multiple:
            raise RuntimeError(
                f"Incomplete async transition: "
                f"{len(self.inputs)}/{self.multiple} lanes."
            )
        if self.pending_communication:
            pending = tuple(lane for lane, _ in self.pending_communication)
            raise RuntimeError(f"Unflushed transition lanes at finish: {pending}.")
        if self.enqueued_group_count != len(self.lane_groups):
            raise RuntimeError(
                f"Incomplete transition communication groups: "
                f"{self.enqueued_group_count}/{len(self.lane_groups)}."
            )
        if self.received is None:
            raise RuntimeError("This rank did not materialize its destination lane.")
        # ProcessGroupNCCL does not expose its completion event.  Defer waits
        # until every producer lane has been launched; by this point the P2P
        # operations have had the entire remaining producer window to run.
        with torch.cuda.stream(self.stream):
            for work in self.works:
                work.wait()
            self.output = assemble_gather_recv(
                self.received,
                use_triton=self.use_triton,
            )
            self.done_event = torch.cuda.Event()
            self.done_event.record(self.stream)
        consumer_stream = torch.cuda.current_stream()
        consumer_stream.wait_event(self.done_event)
        self.output.record_stream(consumer_stream)
        return _AsyncLaneGatherAutograd.apply(
            self.output,
            self.src_cp_size,
            self.dst_cp_size,
            self.cp_group_cache,
            self.use_triton,
            self,
            *self.inputs,
        )


def _repeat_scatter_sequence_p2p(
    seq_local,
    src_cp_size,
    dst_cp_size,
    cp_group_cache,
    *,
    average,
):
    """P2P CP expansion for repeat mode.

    ``src_cp_size < dst_cp_size`` means DP contracts. Repeat mode keeps the
    local batch unchanged, so source DP replicas inside one destination CP
    group contain duplicate logical samples. Each source rank splits its
    Megatron-zigzag sequence shard exactly as the normal scatter does; every
    destination rank receives the corresponding shard from all duplicate
    source DP replicas and either sums or averages them.
    """
    if dst_cp_size <= src_cp_size or dst_cp_size % src_cp_size:
        raise ValueError(
            f"Repeat scatter requires nested CP expansion, got "
            f"{src_cp_size}->{dst_cp_size}."
        )

    # Autograd may supply an expanded, zero-stride gradient (for example from
    # tensor.sum()).  Materialize it before reshaping or handing derived
    # buffers to NCCL P2P operations.
    seq_local = seq_local.contiguous()
    seq_len, batch, hidden = seq_local.shape
    multiple = dst_cp_size // src_cp_size
    if seq_len % (2 * multiple):
        raise ValueError(
            f"Local sequence length {seq_len} is not divisible by "
            f"2*repeat_ratio={2 * multiple} for CP{src_cp_size}->CP{dst_cp_size}."
        )

    src_ranks = cp_group_cache[src_cp_size][1]
    dst_ranks = cp_group_cache[dst_cp_size][1]
    cur_global = dist.get_rank()
    src_local_rank = src_ranks.index(cur_global)
    dst_local_rank = dst_ranks.index(cur_global)

    seq_view = seq_local.view(
        2 * multiple,
        seq_len // (2 * multiple),
        batch,
        hidden,
    )
    send_chunks = []
    send_targets = []
    for piece in range(multiple):
        chunk = torch.cat(
            [seq_view[piece], seq_view[2 * multiple - piece - 1]],
            dim=0,
        ).contiguous()
        dst_global = dst_ranks[src_local_rank * multiple + piece]
        send_chunks.append(chunk)
        send_targets.append(dst_global)

    ops = []
    recv_chunks = [None] * multiple
    for replica in range(multiple):
        chunk = send_chunks[replica]
        dst_global = send_targets[replica]
        if dst_global != cur_global:
            ops.append(dist.P2POp(dist.isend, chunk, dst_global))

        src_global = dst_ranks[
            replica * src_cp_size + (dst_local_rank // multiple)
        ]
        if src_global != cur_global:
            received = torch.empty_like(chunk)
            ops.append(dist.P2POp(dist.irecv, received, src_global))
        else:
            self_piece = send_targets.index(cur_global)
            received = send_chunks[self_piece]
        recv_chunks[replica] = received

    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    reduced = recv_chunks[0].clone()
    for received in recv_chunks[1:]:
        reduced.add_(received)
    if average and multiple > 1:
        reduced.div_(multiple)
    return reduced


def _repeat_gather_sequence_p2p(
    seq_local,
    src_cp_size,
    dst_cp_size,
    cp_group_cache,
    *,
    scale=1.0,
):
    """P2P CP contraction for repeat mode.

    ``src_cp_size > dst_cp_size`` expands DP. Instead of splitting the local
    batch as the normal redistribution gather does, the same batch is sent to
    every new DP replica. Received sequence shards are reassembled in the exact
    Megatron zigzag order expected by ``dst_cp_size``.
    """
    if src_cp_size <= dst_cp_size or src_cp_size % dst_cp_size:
        raise ValueError(
            f"Repeat gather requires nested CP contraction, got "
            f"{src_cp_size}->{dst_cp_size}."
        )

    # This helper sends the input tensor itself.  NCCL P2P requires a stable,
    # dense buffer; backward gradients from reductions can be expanded views
    # with zero strides, which otherwise lead to an illegal memory access.
    seq_local = seq_local.contiguous()
    seq_len, batch, hidden = seq_local.shape
    multiple = src_cp_size // dst_cp_size
    if seq_len % 2:
        raise ValueError(
            f"Local sequence length {seq_len} must be even for repeat gather."
        )

    src_ranks = cp_group_cache[src_cp_size][1]
    cur_global = dist.get_rank()
    src_local_rank = src_ranks.index(cur_global)
    send_column = src_local_rank // multiple
    recv_column = src_local_rank % dst_cp_size

    send_targets = [
        src_ranks[replica * dst_cp_size + send_column]
        for replica in range(multiple)
    ]

    ops = []
    recv_chunks = [None] * multiple
    for replica, dst_global in enumerate(send_targets):
        if dst_global != cur_global:
            ops.append(dist.P2POp(dist.isend, seq_local, dst_global))

        src_global = src_ranks[recv_column * multiple + replica]
        if src_global != cur_global:
            received = torch.empty_like(seq_local)
            ops.append(dist.P2POp(dist.irecv, received, src_global))
        else:
            received = seq_local
        recv_chunks[replica] = received
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    half = seq_len // 2
    out = seq_local.new_empty(
        2 * multiple,
        half,
        batch,
        hidden,
    )
    for piece, received in enumerate(recv_chunks):
        received = received.view(2, half, batch, hidden)
        out[piece].copy_(received[0])
        out[2 * multiple - piece - 1].copy_(received[1])
    out = out.view(seq_len * multiple, batch, hidden)
    if scale != 1.0:
        out.mul_(scale)
    return out


class _RepeatGatherSequenceForCPTransition(torch.autograd.Function):
    """Replicate one logical DP sample when CP contracts and DP expands."""

    @staticmethod
    def forward(ctx, seq_local, src_cp_size, dst_cp_size, cp_group_cache):
        ctx.src_cp_size = src_cp_size
        ctx.dst_cp_size = dst_cp_size
        ctx.cp_group_cache = cp_group_cache
        return _repeat_gather_sequence_p2p(
            seq_local,
            src_cp_size,
            dst_cp_size,
            cp_group_cache,
        )

    @staticmethod
    def backward(ctx, grad):
        return _repeat_scatter_sequence_p2p(
            grad,
            ctx.dst_cp_size,
            ctx.src_cp_size,
            ctx.cp_group_cache,
            average=False,
        ), None, None, None


class _RepeatScatterSequenceForCPTransition(torch.autograd.Function):
    """Average duplicate DP replicas when CP expands and DP contracts."""

    @staticmethod
    def forward(ctx, seq_local, src_cp_size, dst_cp_size, cp_group_cache):
        ctx.src_cp_size = src_cp_size
        ctx.dst_cp_size = dst_cp_size
        ctx.cp_group_cache = cp_group_cache
        ctx.multiple = dst_cp_size // src_cp_size
        return _repeat_scatter_sequence_p2p(
            seq_local,
            src_cp_size,
            dst_cp_size,
            cp_group_cache,
            average=True,
        )

    @staticmethod
    def backward(ctx, grad):
        return _repeat_gather_sequence_p2p(
            grad,
            ctx.dst_cp_size,
            ctx.src_cp_size,
            ctx.cp_group_cache,
            scale=1.0 / ctx.multiple,
        ), None, None, None


def transmition_cp_size(
    seq,
    src_cp,
    dst_cp,
    cache,
    strategy="redistribute",
    *,
    use_triton=True,
):
    """Change CP layout using redistribution or spatial-repeat semantics."""
    if src_cp == dst_cp:
        return seq

    large_cp, small_cp = max(src_cp, dst_cp), min(src_cp, dst_cp)
    if large_cp % small_cp:
        raise ValueError(
            f"CP transition {src_cp}->{dst_cp} requires nested CP sizes."
        )

    if strategy == "repeat":
        if dst_cp > src_cp:
            return _RepeatScatterSequenceForCPTransition.apply(
                seq, src_cp, dst_cp, cache
            )
        return _RepeatGatherSequenceForCPTransition.apply(
            seq, src_cp, dst_cp, cache
        )

    if strategy != "redistribute":
        raise ValueError(
            f"Unknown CP transition strategy {strategy!r}; expected "
            "'redistribute' or 'repeat'."
        )

    if DEBUG_CP:
        set_transition_direction(src_cp, dst_cp)
    if dst_cp > src_cp:
        return _ScatterSequenceForCPTransition.apply(
            seq,
            src_cp,
            dst_cp,
            cache,
            use_triton,
        )
    return _GatherSequenceForCPTransition.apply(
        seq,
        src_cp,
        dst_cp,
        cache,
        use_triton,
    )


# ---------- Global FLOPs formulas ----------
def attn_flops_global(batch, seq_len, hidden, n_heads, n_kv, dp_size=1):
    """Total forward FLOPS for one attention layer, all GPUs combined."""
    global_batch = batch * dp_size
    head_dim = hidden // n_heads
    kv_dim = head_dim * n_kv
    # QKV proj (all tokens)
    proj = 2 * global_batch * seq_len * hidden * (hidden + 2 * kv_dim)
    # Core attention (full seq_len x seq_len)
    core = 2 * global_batch * seq_len * seq_len * hidden * 0.5
    # Out proj
    out = 2 * global_batch * seq_len * hidden * hidden
    return proj + core + out


def gdn_flops_global(batch, seq_len, hidden, qk_head_dim, v_head_dim,
                     n_qk_heads, n_v_heads, conv_kernel, dp_size=1):
    """Total forward FLOPS for one GDN layer, all GPUs combined."""
    global_batch = batch * dp_size
    tokens = global_batch * seq_len
    qk_dim = qk_head_dim * n_qk_heads
    v_dim = v_head_dim * n_v_heads
    tflops =  2 * tokens * (
        hidden * (2 * qk_dim + 2 * v_dim + 2 * n_v_heads) +
        conv_kernel * (2 * qk_dim + v_dim) +
        n_v_heads * (v_head_dim ** 2) * 4 +
        hidden * v_dim
    )
    return tflops


# ---------- TE module builders ----------
def build_attn_te(hidden, n_heads, n_kv, cp_size, tp_group, cp_group):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear, TEDotProductAttention, TERowParallelLinear,
    )

    config = TransformerConfig(
        num_layers=1, hidden_size=hidden, num_attention_heads=n_heads,
        num_query_groups=n_kv, context_parallel_size=cp_size,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        bf16=True, params_dtype=torch.bfloat16,
        activation_func=torch.nn.functional.silu,
        attention_dropout=0.0, sequence_parallel=False, normalization="RMSNorm",
    )
    pg = ProcessGroupCollection(tp=tp_group, cp=cp_group)
    head_dim = hidden // n_heads

    qkv_proj = TELayerNormColumnParallelLinear(
        input_size=hidden, output_size=(n_heads + 2 * n_kv) * head_dim,
        config=config, init_method=config.init_method, gather_output=False,
        bias=False, skip_bias_add=False, is_expert=False,
        tp_comm_buffer_name='qkv', tp_group=tp_group,
    ).cuda()

    core_attn = TEDotProductAttention(
        config=config, layer_number=1, attn_mask_type=AttnMaskType.causal,
        attention_type="self", cp_comm_type="p2p" if cp_size > 1 else None,
        pg_collection=pg,
    ).cuda()

    out_proj = TERowParallelLinear(
        input_size=hidden, output_size=hidden, config=config,
        init_method=config.output_layer_init_method, bias=False,
        skip_bias_add=False, is_expert=False, input_is_parallel=True,
        tp_comm_buffer_name='proj', tp_group=tp_group,
    ).cuda()

    def forward(x):
        s, b, _ = x.shape
        qkv, _ = qkv_proj(x)
        q = qkv[..., :n_heads * head_dim].view(s, b, n_heads, head_dim)
        k = qkv[..., n_heads * head_dim:(n_heads + n_kv) * head_dim].view(s, b, n_kv, head_dim)
        v = qkv[..., (n_heads + n_kv) * head_dim:].view(s, b, n_kv, head_dim)
        attn_out = core_attn(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.causal)
        out, _ = out_proj(attn_out)
        return out

    # The benchmark owns these modules so it can model the data-parallel
    # gradient synchronization performed by a real training iteration.
    forward._benchmark_modules = (qkv_proj, core_attn, out_proj)
    return forward


def build_gdn_te(hidden, n_heads, n_kv, cp_size, tp_group, cp_group, args):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear, TERowParallelLinear, TENorm,
    )

    config = TransformerConfig(
        num_layers=1, hidden_size=hidden, num_attention_heads=n_heads,
        num_query_groups=n_kv, context_parallel_size=cp_size,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        bf16=True, params_dtype=torch.bfloat16,
        activation_func=torch.nn.functional.silu, sequence_parallel=False,
        normalization="RMSNorm", experimental_attention_variant="gated_delta_net",
        linear_attention_freq=9999,
        linear_conv_kernel_dim=args.gdn_conv_kernel_dim,
        linear_key_head_dim=args.gdn_key_head_dim,
        linear_value_head_dim=args.gdn_value_head_dim,
        linear_num_key_heads=args.gdn_num_key_heads,
        linear_num_value_heads=args.gdn_num_value_heads,
    )
    pg = ProcessGroupCollection(tp=tp_group, cp=cp_group)
    sub = GatedDeltaNetSubmodules(
        in_proj=ModuleSpec(module=TELayerNormColumnParallelLinear),
        out_norm=ModuleSpec(module=TENorm),
        out_proj=ModuleSpec(module=TERowParallelLinear),
    )
    module = GatedDeltaNet(config=config, submodules=sub, layer_number=1, pg_collection=pg).cuda()

    def forward(x):
        out, _ = module(x, attention_mask=None)
        return out

    forward._benchmark_modules = (module,)
    return forward


# ---------- Timing utilities ----------
def timed_fwd(fn, x, warmup, iters):
    for _ in range(warmup):
        with torch.no_grad():
            fn(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            fn(x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def timed_bwd(fn, x, warmup, iters, zero_grad=None, post_backward=None):
    for _ in range(warmup):
        if zero_grad is not None:
            zero_grad()
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        out.sum().backward()
        if post_backward is not None:
            post_backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if zero_grad is not None:
            zero_grad()
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out.sum().backward()
        if post_backward is not None:
            post_backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def timed_fwd_bwd(fn, x, warmup, iters, zero_grad=None, post_backward=None):
    for _ in range(warmup):
        if zero_grad is not None:
            zero_grad()
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        out.sum().backward()
        if post_backward is not None:
            post_backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if zero_grad is not None:
            zero_grad()
        if x.grad is not None:
            x.grad = None
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(x)
        out.sum().backward()
        if post_backward is not None:
            post_backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def set_transition_direction(src_cp, dst_cp):
    global DEBUG_CP, DIRECTION_2_4, DIRECTION_2_8, DIRECTION_4_8, DIRECTION_4_1, DIRECTION_8_1, DIRECTION
    if (src_cp == 2 and dst_cp == 4) or (src_cp == 4 and dst_cp == 2):
        DIRECTION = DIRECTION_2_4
    elif (src_cp == 2 and dst_cp == 8) or (src_cp == 8 and dst_cp == 2):
        DIRECTION = DIRECTION_2_8
    elif (src_cp == 4 and dst_cp == 8) or (src_cp == 8 and dst_cp == 4):
        DIRECTION = DIRECTION_4_8
    elif (src_cp == 4 and dst_cp == 1) or (src_cp == 1 and dst_cp == 4):
        DIRECTION = DIRECTION_4_1
    elif (src_cp == 8 and dst_cp == 1) or (src_cp == 1 and dst_cp == 8):
        DIRECTION = DIRECTION_8_1
    else:
        raise ValueError(f"Unsupported CP transition: {src_cp} -> {dst_cp}")
