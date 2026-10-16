"""Versioned BF16 numerical contracts; exact data and finite checks remain."""

import json
import math
import os
from pathlib import Path


def contract(profile: str = "strict_v1", architecture: str = "gdngqa_gpt") -> dict:
    """Return a contract frozen before a launch, with architecture-aware limits."""
    if profile not in ("strict_v1", "cp_bf16_v2", "cp_bf16_exploratory_v3") or architecture not in ("gdngqa_gpt", "gqa_gpt"):
        raise ValueError("Unknown numerical profile or architecture")
    if profile == "cp_bf16_exploratory_v3":
        result = contract("cp_bf16_v2", architecture)
        result.update(profile=profile,
                      scope="User-authorized exploratory performance admission; three Adam steps; provisional numerics, no strict-equivalence or convergence claim",
                      authorization="User on 2026-10-14 requested relaxed precision thresholds and continuation unless clearly erroneous",
                      basis="Frozen exploratory limits; original cp_bf16_v2 failures retained separately")
        result["optimizer"]["gradients"] = dict(relative=0.15, absolute=1e-6)
        result["optimizer"]["weights"] = dict(relative=0.01, absolute=1e-6)
        result["optimizer"]["states"] = dict(relative=0.25, absolute=1e-6)
        result["logical"].update(decoder_Y=0.05, embedding_dX=0.10, token_loss=0.03)
        return result
    relaxed = profile == "cp_bf16_v2"
    y, dx, dw = (0.03, 0.04, 0.06) if architecture == "gdngqa_gpt" else (0.02, 0.03, 0.05)
    return dict(
        profile=profile,
        architecture=architecture,
        scope="development BF16 CP/packing comparison, first three Adam steps; no long-run convergence claim",
        authorization=("User requested relaxed CP tolerances on 2026-10-09" if relaxed else "original diagnostic"),
        basis=(
            "Y/dX/dW match existing distributed_bench.py d3g1/g4 limits; optimizer limits are separate"
            if relaxed
            else "historical strict one-dimensional scheduler diagnostic"
        ),
        optimizer={
            "initial_weights": dict(relative=0.0 if relaxed else 1e-5, absolute=0.0 if relaxed else 1e-7),
            "gradients": dict(relative=dw if relaxed else 0.005, absolute=1e-6),
            "weights": dict(relative=0.005 if relaxed else 1e-5, absolute=1e-6 if relaxed else 1e-7),
            "states": dict(relative=0.10 if relaxed else 0.005, absolute=1e-6),
        },
        logical={
            "tokens": 0.0,
            "positions": 0.0,
            "labels": 0.0,
            "loss_mask": 0.0,
            "embedding_Y": 0.0,
            "decoder_Y": y if relaxed else 0.005,
            "embedding_dX": dx if relaxed else 0.005,
            "token_loss": 0.01 if relaxed else 0.005,
        },
        finite_required=True,
        exact_data_and_ownership=True,
    )


def load_contract() -> dict:
    """Load the launch snapshot, or retain legacy strict behavior for direct use."""
    path = os.environ.get("SEQUENCE_CP_NUMERICAL_CONTRACT")
    data = json.loads(Path(path).read_text()) if path else contract()
    expected = contract(data["profile"], data["architecture"])
    # Saved contracts must match a named version, avoiding silent per-run tuning.
    if data != expected:
        raise ValueError("Numerical contract differs from its frozen named version")
    values = [v for limits in data["optimizer"].values() for v in limits.values()] + list(data["logical"].values())
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Numerical limits must be finite and nonnegative")
    return data
