"""Deterministic content primitives shared by explicit joint input manifests."""

MODULUS = 1 << 64
MULTIPLIER = 6364136223846793005
INCREMENT = 1442695040888963407
SAMPLE_STRIDE = 1 << 32


def validate_manifest(manifest):
    from joint_cp.inputs import validate_manifest as validate_joint
    validate_joint(manifest)


def _advance(state: int, count: int) -> int:
    """Jump a linear congruential generator by count steps in O(log count)."""
    if count < 0:
        raise ValueError("negative stream offset")
    multiplier, increment = MULTIPLIER, INCREMENT
    acc_multiplier, acc_increment = 1, 0
    while count:
        if count & 1:
            acc_multiplier = acc_multiplier * multiplier % MODULUS
            acc_increment = (acc_increment * multiplier + increment) % MODULUS
        increment = increment * (multiplier + 1) % MODULUS
        multiplier = multiplier * multiplier % MODULUS
        count >>= 1
    return (acc_multiplier * state + acc_increment) % MODULUS


def materialize_span(manifest: dict, sample_id: int, start: int, stop: int) -> dict:
    """Generate only the requested causal sample slice, as plain CPU lists."""
    sample = next((s for s in manifest["samples"] if s["sample_id"] == sample_id), None)
    if sample is None or not 0 <= start <= stop <= sample["length"]:
        raise ValueError("invalid sample or span bounds")
    rule = manifest["content_rule"]
    state = _advance(rule["initial_state"], sample_id * SAMPLE_STRIDE + start + 1)
    stream_tokens = []
    for _ in range(stop - start + 1):
        stream_tokens.append((state >> 32) % rule["vocab_size"])
        state = (state * MULTIPLIER + INCREMENT) % MODULUS
    return {"tokens": stream_tokens[:-1], "labels": stream_tokens[1:],
            "positions": list(range(start, stop)),
            "loss_mask": [int(position < sample["loss_prefix_tokens"])
                          for position in range(start, stop)]}
