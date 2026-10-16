"""Reconstruct logical samples to check Y/dX across changing CP and packing."""

import json
import os
import sys
from pathlib import Path

from numerical_contracts import load_contract


class LogicalNumerics:
    """One-step, TP=PP=1 checks; external packed layout must be native zigzag."""

    def __init__(self, state: dict, output: Path):
        self.state, self.output = state, output
        self.role = os.environ.get("SEQUENCE_CP_LOGICAL", "off")
        if self.role not in ("off", "write", "read"):
            raise ValueError("Unknown logical numerical role")
        self.reference = Path(os.environ.get("SEQUENCE_CP_LOGICAL_REFERENCE", "/tmp/prism_r2_logical_unused"))
        self.active = False
        self.installed = False
        self.parts = {}
        self.ordinal = 0
        self.group = None
        self.step = 0
        self.contract = load_contract()

    def begin(self, model, step: int) -> None:
        """Install model hooks once, activate only inside the training step."""
        if self.role == "off" or step > 1:
            return
        self.active, self.step = True, step
        self.parts, self.ordinal = {}, 0
        if self.installed:
            return
        if len(model) != 1:
            raise ValueError("Logical diagnostic requires one model chunk")
        core = model[0]
        while hasattr(core, "module"):
            core = core.module

        def embedding(module, inputs, kwargs, output):
            if not self.active:
                return
            rank = int(os.environ["RANK"])
            from mixed_plan import local_execution_groups
            self.group = local_execution_groups(self.state, self.ordinal, rank)
            self.ordinal += 1
            group = self.group
            tokens = kwargs.get("input_ids", inputs[0] if inputs else None)
            positions = kwargs.get("position_ids", inputs[1] if len(inputs) > 1 else None)
            if tokens is None or positions is None:
                raise ValueError("Embedding hook requires tokens and position IDs")
            self.capture("tokens", tokens.reshape(-1, 1), group)
            self.capture("positions", positions.reshape(-1, 1), group)
            self.capture("embedding_Y", output, group)
            output.register_hook(lambda grad: self.capture("embedding_dX", grad, group))

        def decoder(module, inputs, output):
            if self.active:
                self.capture("decoder_Y", output[0] if isinstance(output, tuple) else output, self.group)

        core.embedding.register_forward_hook(embedding, with_kwargs=True)
        core.decoder.register_forward_hook(decoder)
        original_loss = core.compute_language_model_loss

        def token_loss(labels, logits):
            result = original_loss(labels, logits)
            if self.active:
                self.capture("labels", labels.reshape(-1, 1), self.group)
                self.capture("token_loss", result.reshape(-1, 1), self.group)
            return result

        core.compute_language_model_loss = token_loss
        main = sys.modules["__main__"]
        original_masked_loss = main.loss_func

        def masked_loss(loss_mask, output_tensor, *args, **kwargs):
            if self.active:
                self.capture("loss_mask", loss_mask.reshape(-1, 1), self.group)
            return original_masked_loss(loss_mask, output_tensor, *args, **kwargs)

        main.loss_func = masked_loss
        self.installed = True

    def capture(self, name: str, value, group) -> None:
        """Save each CP shard with explicit logical row positions and sample ID."""
        import torch

        if not self.active:
            return
        if isinstance(group, tuple):
            value = value.reshape(-1, value.shape[-1])
            cursor = 0
            for part in group:
                rows = sum(self.state["lengths"][i] for i in part.ids) // len(part.ranks)
                self.capture(name, value[cursor:cursor + rows], part)
                cursor += rows
            if cursor != value.shape[0]:
                raise ValueError("Mixed logical rows do not match context segments")
            return
        cp, rank = len(group.ranks), group.ranks.index(int(os.environ["RANK"]))
        flat = value.detach().reshape(-1, value.shape[-1]).cpu()
        expected_rows = sum(self.state["lengths"][i] for i in group.ids) // cp
        if flat.shape[0] != expected_rows:
            raise ValueError(f"Logical {name} row shape does not match planned native ownership")
        cursor = 0
        for i in group.ids:
            n = self.state["lengths"][i]
            width = n // (2 * cp)
            positions = torch.cat(
                (
                    torch.arange(rank * width, (rank + 1) * width),
                    torch.arange(n - (rank + 1) * width, n - rank * width),
                )
            )
            rows = n // cp
            self.parts.setdefault(name, []).append(
                dict(sample_id=i, positions=positions, value=flat[cursor : cursor + rows].clone())
            )
            cursor += rows

    def finish(self) -> None:
        """Check exact data/ownership and the launch's frozen Y/dX limits."""
        import torch
        import torch.distributed as dist

        if not self.active:
            return
        self.active = False
        required = {
            "tokens",
            "positions",
            "labels",
            "loss_mask",
            "embedding_Y",
            "decoder_Y",
            "embedding_dX",
            "token_loss",
        }
        if set(self.parts) != required:
            raise ValueError("Incomplete logical numerical fields")
        rank = int(os.environ["RANK"])
        if self.role == "write":
            self.reference.mkdir(parents=True, exist_ok=True)
            torch.save(self.parts, self.reference / f"logical_rank{rank}_step{self.step}.pt")
            return
        references = [
            torch.load(
                self.reference / f"logical_rank{r}_step{self.step}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            for r in range(dist.get_world_size())
        ]
        rows = []
        failed = False
        exact_fields = {"tokens", "positions", "labels", "loss_mask", "embedding_Y"}
        for field, pieces in self.parts.items():
            by_sample = {}
            for reference in references:
                for part in reference[field]:
                    by_sample.setdefault(part["sample_id"], []).append(part)
            for piece in pieces:
                sample = piece["sample_id"]
                width = piece["value"].shape[-1]
                n = self.state["lengths"][sample]
                expected = torch.empty((n, width), dtype=piece["value"].dtype)
                seen = torch.zeros(n, dtype=torch.int32)
                for source in by_sample[sample]:
                    seen[source["positions"]] += 1
                    expected[source["positions"]] = source["value"]
                if not bool((seen == 1).all()):
                    raise ValueError("Logical reference has missing/duplicate ownership")
                expected = expected[piece["positions"]]
                actual = piece["value"]
                exact = torch.equal(actual, expected)
                a, b = actual.float(), expected.float()
                relative = float((a - b).norm() / b.norm().clamp_min(1e-12))
                absolute = float((a - b).abs().max())
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                threshold = self.contract["logical"][field]
                passed = finite and (exact if field in exact_fields else relative <= threshold)
                failed |= not passed
                rows.append(
                    dict(
                        step=self.step,
                        field=field,
                        sample_id=sample,
                        local_rows=len(actual),
                        relative_l2=relative,
                        max_abs=absolute,
                        exact=exact,
                        finite=finite,
                        passed=passed,
                        threshold=threshold,
                        numerical_profile=self.contract["profile"],
                    )
                )
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / f"logical_rank{rank}.jsonl").open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        flag = torch.tensor(int(failed), device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise RuntimeError("Logical sequence Y/dX/data numerical gate failed")
