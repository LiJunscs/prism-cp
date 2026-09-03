"""Pure execution-plan logic for cohort-aware TMF 1F1B."""

import math
from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class LayerSpec:
    layer_type: str
    cp_size: int


@dataclass(frozen=True)
class PipelineBoundary:
    boundary_id: int
    source_stage: int
    destination_stage: int
    source_cp: int
    destination_cp: int
    temporal_width: int
    wire_item_shape: tuple


@dataclass(frozen=True)
class TMFExecutionPlan:
    stage_plans: tuple
    stage_cohort_widths: tuple
    cohort_width: int
    logical_microbatches: int
    pipeline_cohorts: int
    boundaries: tuple


def _lcm(values):
    return reduce(math.lcm, values, 1)


def _parse_layer_token(token):
    try:
        layer_name, cp_text = (part.strip() for part in token.split(":", 1))
        cp_size = int(cp_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid layer token {token!r}; expected TYPE:CP, for example gdn:1."
        ) from exc
    aliases = {
        "a": "attention",
        "attn": "attention",
        "attention": "attention",
        "g": "gdn",
        "gdn": "gdn",
    }
    try:
        layer_type = aliases[layer_name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown layer type {layer_name!r}; expected attention/attn/a or gdn/g."
        ) from exc
    if cp_size < 1:
        raise ValueError(f"CP size must be positive in layer token {token!r}.")
    return LayerSpec(layer_type, cp_size)


def parse_stage_layer_patterns(specification, pp_size, layers_per_stage, default_cp):
    """Parse ``stage;stage`` and ``TYPE:CP,TYPE:CP`` layer patterns."""
    if specification is None:
        return tuple(
            tuple(LayerSpec("attention", default_cp) for _ in range(layers_per_stage))
            for _ in range(pp_size)
        )

    stage_texts = [stage.strip() for stage in specification.split(";")]
    if len(stage_texts) != pp_size:
        raise ValueError(
            f"--stage-layer-patterns defines {len(stage_texts)} stages, "
            f"but pipeline_parallel_size={pp_size}."
        )
    stage_plans = []
    for stage, stage_text in enumerate(stage_texts):
        if not stage_text:
            raise ValueError(f"Stage {stage} has an empty layer pattern.")
        tokens = [token.strip() for token in stage_text.split(",")]
        if any(not token for token in tokens):
            raise ValueError(f"Stage {stage} contains an empty layer token.")
        stage_plans.append(tuple(_parse_layer_token(token) for token in tokens))
    return tuple(stage_plans)


def temporal_width(logical_microbatch_size, stage_mesh_size, cp_size):
    """Minimum number of logical microbatches integral under ``DP=mesh/CP``."""
    if logical_microbatch_size < 1:
        raise ValueError("logical_microbatch_size must be positive.")
    if cp_size < 1 or stage_mesh_size % cp_size:
        raise ValueError(
            f"CP{cp_size} must be positive and divide stage mesh {stage_mesh_size}."
        )
    dp_size = stage_mesh_size // cp_size
    return dp_size // math.gcd(dp_size, logical_microbatch_size)


def local_payload_shape(
    seq_length,
    logical_microbatch_size,
    hidden_size,
    stage_mesh_size,
    cp_size,
    logical_width,
):
    """Shape of one payload representing ``logical_width`` microbatches."""
    if seq_length % (2 * cp_size):
        raise ValueError(
            f"seq_length={seq_length} must be divisible by 2*CP{cp_size}."
        )
    dp_size = stage_mesh_size // cp_size
    numerator = logical_microbatch_size * logical_width
    if numerator % dp_size:
        raise ValueError(
            f"logical batch {logical_microbatch_size} * width {logical_width} "
            f"is not divisible by stage-local DP={dp_size} at CP{cp_size}."
        )
    return (seq_length // cp_size, numerator // dp_size, hidden_size)


def _validate_nested_cp(source_cp, destination_cp, context):
    large_cp = max(source_cp, destination_cp)
    small_cp = min(source_cp, destination_cp)
    if large_cp % small_cp:
        raise ValueError(
            f"{context} CP{source_cp}->CP{destination_cp} requires nested CP sizes."
        )


def build_execution_plan(
    stage_plans,
    *,
    pp_size,
    stage_mesh_size,
    logical_microbatches,
    logical_microbatch_size,
    seq_length,
    hidden_size,
    cross_stage_policy="forbid",
):
    """Validate the initial TMF scope and freeze all cohort/boundary metadata.

    The initial implementation deliberately requires temporal width 1 at every
    PP boundary. Cross-stage CP remeshing is represented in the boundary schema
    but rejected until a boundary-cohort protocol is implemented.
    """
    if pp_size < 1 or stage_mesh_size < 1:
        raise ValueError("PP size and stage mesh size must be positive.")
    if logical_microbatches < 1:
        raise ValueError("logical_microbatches must be positive.")
    if len(stage_plans) != pp_size:
        raise ValueError(
            f"Expected {pp_size} stage plans, received {len(stage_plans)}."
        )
    if cross_stage_policy != "forbid":
        raise NotImplementedError(
            "Cross-stage CP transition policy is reserved but not implemented; "
            "use cross_stage_policy='forbid'."
        )

    stage_widths = []
    for stage, stage_plan in enumerate(stage_plans):
        if not stage_plan:
            raise ValueError(f"Stage {stage} must contain at least one layer.")
        widths = []
        for layer_index, layer in enumerate(stage_plan):
            if stage_mesh_size % layer.cp_size:
                raise ValueError(
                    f"stage={stage} layer={layer_index} CP{layer.cp_size} must "
                    f"divide stage mesh {stage_mesh_size}."
                )
            if seq_length % (2 * layer.cp_size) or seq_length // layer.cp_size < 128:
                raise ValueError(
                    f"Invalid seq_length={seq_length} for stage={stage} "
                    f"layer={layer_index} CP{layer.cp_size}."
                )
            widths.append(
                temporal_width(
                    logical_microbatch_size, stage_mesh_size, layer.cp_size
                )
            )
        for source, destination in zip(stage_plan, stage_plan[1:]):
            _validate_nested_cp(
                source.cp_size,
                destination.cp_size,
                f"stage={stage}",
            )
        if widths[0] != 1 or widths[-1] != 1:
            raise ValueError(
                f"Initial TMF adapter requires stage {stage} entry/exit temporal "
                f"width 1, got entry={widths[0]} and exit={widths[-1]}. "
                "Place a width-1 CP block at both stage boundaries."
            )
        stage_widths.append(_lcm(widths))

    cohort_width = _lcm(stage_widths)
    if logical_microbatches % cohort_width:
        raise ValueError(
            f"logical microbatches M={logical_microbatches} must be divisible by "
            f"TMF cohort width K={cohort_width}."
        )

    boundaries = []
    for source_stage in range(pp_size - 1):
        destination_stage = source_stage + 1
        source_cp = stage_plans[source_stage][-1].cp_size
        destination_cp = stage_plans[destination_stage][0].cp_size
        if source_cp != destination_cp:
            raise ValueError(
                "Initial TMF adapter forbids CP transitions across PP stages: "
                f"stage {source_stage} exits CP{source_cp}, while stage "
                f"{destination_stage} enters CP{destination_cp}."
            )
        boundary_width = temporal_width(
            logical_microbatch_size, stage_mesh_size, source_cp
        )
        if boundary_width != 1:
            raise ValueError(
                f"PP boundary {source_stage}->{destination_stage} requires temporal "
                f"width {boundary_width}; the initial adapter requires width 1."
            )
        boundaries.append(
            PipelineBoundary(
                boundary_id=source_stage,
                source_stage=source_stage,
                destination_stage=destination_stage,
                source_cp=source_cp,
                destination_cp=destination_cp,
                temporal_width=boundary_width,
                wire_item_shape=local_payload_shape(
                    seq_length,
                    logical_microbatch_size,
                    hidden_size,
                    stage_mesh_size,
                    source_cp,
                    1,
                ),
            )
        )

    return TMFExecutionPlan(
        stage_plans=tuple(tuple(plan) for plan in stage_plans),
        stage_cohort_widths=tuple(stage_widths),
        cohort_width=cohort_width,
        logical_microbatches=logical_microbatches,
        pipeline_cohorts=logical_microbatches // cohort_width,
        boundaries=tuple(boundaries),
    )


def format_stage_plan(stage_plan):
    names = {"attention": "A", "gdn": "G"}
    return ",".join(
        f"{names[layer.layer_type]}@CP{layer.cp_size}" for layer in stage_plan
    )
