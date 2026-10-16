"""Replay the live native algorithm on CPU without importing Megatron/CUDA."""

import ast
import copy
import os
from functools import lru_cache
from math import ceil, log2
from pathlib import Path

MEGATRON_ROOT = Path(os.environ.get("MEGATRON_ROOT", Path(__file__).resolve().parents[3] / "Megatron-LM"))
SOURCE = MEGATRON_ROOT / "megatron/core/datasets/data_schedule_utils.py"


def load_native(beta_tokens: float = 0.0, source: Path = SOURCE):
    """Return the native function, changing only workload and cap for A/B ablation.

    beta_tokens=0 compiles the unchanged AST. The weighted variant preserves
    sorting, greedy ties, minimum CP, leftovers and the entire fill procedure.
    """
    raw = source.read_bytes()
    tree = ast.parse(raw)
    names = {
        "dcp_gpus_needed",
        "next_hdp_group_packing_aware",
        "_DYNAMIC_CP_WORKLOAD_CAP_DELTA",
    }
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in names
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    if len(nodes) != 3:
        raise ValueError("Native scheduler source changed: review the CPU extraction")
    nodes = copy.deepcopy(nodes)
    if beta_tokens:
        if beta_tokens < 0:
            raise ValueError("beta must be nonnegative")
        edits = 0
        scheduler = next(
            n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "next_hdp_group_packing_aware"
        )
        for node in ast.walk(scheduler):
            if isinstance(node, ast.FunctionDef) and node.name == "workload":
                node.body = ast.parse(f"return (seq_len * seq_len + {beta_tokens!r} * seq_len) / cp_size").body
                edits += 1
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cap" for t in node.targets):
                node.value = ast.parse(
                    f"(float(local_tall) + {beta_tokens!r}) * float(max_seq_len_per_rank) * (1.0 + _DYNAMIC_CP_WORKLOAD_CAP_DELTA)",
                    mode="eval",
                ).body
                edits += 1
        if edits != 2:
            raise ValueError("Native workload/cap structure changed: review weighted ablation")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = dict(lru_cache=lru_cache, ceil=ceil, log2=log2)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["next_hdp_group_packing_aware"], "omitted_no_whole_file_hash"


def replay(
    samples: list[tuple[int, int]],
    world: int,
    cap: int,
    min_cp: int = 1,
    beta_tokens: float = 0.0,
) -> list[list[list[int]]]:
    """Replay every microbatch, checking that each iteration consumes samples."""
    function, _ = load_native(beta_tokens)
    remaining = sorted(samples, key=lambda pair: pair[1], reverse=True)
    batches = []
    while remaining:
        _, leftovers, _, ranks = function(remaining, world, cap, min_cp)
        if len(leftovers) >= len(remaining):
            raise ValueError("Native scheduler made no progress")
        batches.append(ranks)
        remaining = leftovers
    return batches
