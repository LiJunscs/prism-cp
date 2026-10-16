"""Full optimizer-shard comparisons for three Adam steps, outside timing claims."""

import json
import os
from pathlib import Path

from numerical_contracts import load_contract


class TrainingNumerics:
    """Compare all local dW, FP32 master weights and Adam state tensors.

    Repartitioned logits cannot be compared by microbatch ordinal. This gate
    instead compares globally synchronized optimizer shards on the unchanged
    parameter/optimizer domain. Y/dX per-sample equivalence remains separate.
    """

    def __init__(self, output: Path):
        self.role = os.environ.get("SEQUENCE_CP_NUMERICS", "off")
        if self.role not in ("off", "write", "read"):
            raise ValueError("Unknown numerical role")
        self.folder = Path(os.environ.get("SEQUENCE_CP_REFERENCE", "/tmp/prism_sequence_cp_reference"))
        self.output = output
        self.step = 0
        self.installed = False
        self.diagnostic = os.environ.get("SEQUENCE_CP_DIAGNOSTIC") == "1"
        self.check_initial = os.environ.get("SEQUENCE_CP_CHECK_INIT") == "1"
        self.failed = False
        self.metadata = {}
        self.contract = load_contract()

    def check(self, name: str, values: dict) -> None:
        import torch

        if self.role == "off" or self.step > 3:
            return
        rank = os.environ["RANK"]
        path = self.folder / f"rank{rank}_step{self.step}_{name}.pt"
        if not values:
            raise ValueError("Empty full-tensor gate: " + name)
        if self.role == "write":
            self.folder.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in values.items()}, path)
            return
        reference = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if values.keys() != reference.keys():
            raise ValueError("Optimizer shard tensor names differ")
        phase = "initial_weights" if name.endswith("initial_weights") else name.rsplit("_", 1)[-1]
        limits = self.contract["optimizer"][phase]
        relative_limit, absolute_limit = limits["relative"], limits["absolute"]
        worst, max_abs, count, passed = 0.0, 0.0, 0, True
        worst_key = ""
        tensors = []
        for key, tensor in values.items():
            actual, expected = tensor.detach().reshape(-1), reference[key].reshape(-1)
            if actual.shape != expected.shape:
                raise ValueError("Optimizer shard shapes differ")
            error2, norm2, absolute, sign_flips = 0.0, 0.0, 0.0, 0
            finite = True
            for start in range(0, actual.numel(), 2**20):
                a = actual[start : start + 2**20].float().cpu().double()
                b = expected[start : start + 2**20].double()
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                delta = a - b
                error2 += float(delta.square().sum())
                norm2 += float(b.square().sum())
                absolute = max(absolute, float(delta.abs().max()) if delta.numel() else 0.0)
                sign_flips += int(((a * b) < 0).sum())
            relative = (error2 / max(norm2, 1e-30)) ** 0.5
            tensor_pass = finite and (relative <= relative_limit or absolute <= absolute_limit)
            passed &= tensor_pass
            tensors.append(
                dict(
                    key=key,
                    **self.metadata.get(key.split("/")[0], {}),
                    elements=actual.numel(),
                    relative_l2=relative,
                    max_abs=absolute,
                    reference_rms=(norm2 / max(1, actual.numel())) ** 0.5,
                    error_rms=(error2 / max(1, actual.numel())) ** 0.5,
                    sign_flips=sign_flips,
                    passed=tensor_pass,
                )
            )
            if relative > worst:
                worst, worst_key = relative, key
            max_abs = max(max_abs, absolute)
            count += actual.numel()
        self.output.mkdir(parents=True, exist_ok=True)
        row = dict(
            step=self.step,
            name=name,
            passed=passed,
            elements=count,
            relative_l2=worst,
            max_abs=max_abs,
            worst_key=worst_key,
            relative_limit=relative_limit,
            absolute_limit=absolute_limit,
            tensors=tensors,
            numerical_profile=self.contract["profile"],
        )
        with (self.output / f"numerics_rank{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if not passed:
            self.failed = True
            if not self.diagnostic:
                raise RuntimeError(
                    "Sequence CP full optimizer tensor gate failed: "
                    + str({k: v for k, v in row.items() if k != "tensors"})
                )

    def install(self, optimizer, model=None) -> None:
        """Wrap raw Adam after Megatron's parameter-gradient synchronization."""
        import torch

        if self.installed or self.role == "off":
            return
        self.installed = True
        named = {
            id(p): f"chunk{j}.{name}" for j, chunk in enumerate(model or []) for name, p in chunk.named_parameters()
        }
        optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        for index, wrapper in enumerate(optimizers):
            raw = wrapper.optimizer
            original = raw.step
            names = {}
            for original_attr, shard_attr in (
                ("model_float16_groups", "shard_fp32_from_float16_groups"),
                ("model_fp32_groups", "shard_fp32_groups"),
            ):
                for originals, shards in zip(getattr(wrapper, original_attr, []), getattr(wrapper, shard_attr, [])):
                    for parameter, shard in zip(originals, shards):
                        info = dict(parameter=named.get(id(parameter), "unknown"), model_shape=list(parameter.shape))
                        if hasattr(wrapper, "_get_model_param_range_map"):
                            interval = wrapper._get_model_param_range_map(parameter)["param"]
                            info.update(shard_start=interval.start, shard_end=interval.end)
                        names[id(shard)] = info
            params = [p for group in raw.param_groups for p in group["params"]]
            self.metadata = {str(i): names.get(id(p), dict(parameter="unknown")) for i, p in enumerate(params)}
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / f"parameters_rank{os.environ['RANK']}_optimizer{index}.json").write_text(
                json.dumps(self.metadata, indent=2) + "\n"
            )
            if self.check_initial:
                ordinal = self.step
                self.step = 0
                self.check(f"optimizer{index}_initial_weights", {str(i): p for i, p in enumerate(params)})
                self.step = ordinal

            def step(*args, _raw=raw, _original=original, _index=index, **kwargs):
                params = [p for group in _raw.param_groups for p in group["params"]]
                prefix = f"optimizer{_index}_"
                self.check(prefix + "gradients", {str(i): p.grad for i, p in enumerate(params) if p.grad is not None})
                result = _original(*args, **kwargs)
                self.check(prefix + "weights", {str(i): p for i, p in enumerate(params)})
                self.check(
                    prefix + "states",
                    {
                        f"{i}/{k}": v if torch.is_tensor(v) else torch.as_tensor(v)
                        for i, p in enumerate(params)
                        for k, v in _raw.state[p].items()
                        if torch.is_tensor(v) or isinstance(v, (int, float))
                    },
                )
                return result

            raw.step = step

    def finish_step(self) -> None:
        """In a one-step diagnostic, collect all errors then enforce the same gate."""
        if self.diagnostic and self.role == "read":
            import torch
            import torch.distributed as dist

            flag = torch.tensor(int(self.failed), device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                raise RuntimeError("Sequence CP diagnostic collected all tensors; unchanged numerical gate failed")
