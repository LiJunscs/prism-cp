"""Explicit variable-size joint experiment manifests with deterministic content."""

from two_dimensional_inputs import MODULUS, MULTIPLIER, INCREMENT, SAMPLE_STRIDE


def build_manifest(lengths, case_id, world, seed=1234):
    manifest = dict(schema="joint_cp_inputs_v1", case_id=case_id, world=world, seed=seed,
        variant="frozen_lengths", grouping_constraint=False, alignment_tokens=2*world,
        backend_alignments={"joint_attention":2*world}, input_tokens=sum(lengths),
        loss_tokens=sum(n//2 for n in lengths),
        content_rule=dict(name="lcg64_next_token_v1",vocab_size=32768,initial_state=seed,
            modulus=MODULUS,multiplier=MULTIPLIER,increment=INCREMENT,sample_stride=SAMPLE_STRIDE,
            token="upper_32_bits_mod_vocab",label="token_at_position_plus_one_in_same_sample_stream",
            position="zero_based_sample_position",loss_mask="position < length // 2"),
        samples=[dict(sample_id=i,sample_uid=f"{case_id}:{i:06d}",length=n,original_length=n,
                      loss_prefix_tokens=n//2) for i,n in enumerate(lengths)])
    validate_manifest(manifest)
    return manifest


def validate_manifest(m):
    if m.get("schema") != "joint_cp_inputs_v1" or m.get("variant") != "frozen_lengths":
        raise ValueError("Unknown joint input schema")
    world=m.get("world")
    if world not in (8,16,32) or type(m.get("seed")) is not int or not 0 <= m["seed"] < MODULUS:
        raise ValueError("Invalid joint world or seed")
    if m.get("grouping_constraint") is not False or m.get("alignment_tokens") != 2*world:
        raise ValueError("Joint grouping/alignment contract changed")
    if m.get("backend_alignments") != {"joint_attention":2*world}:
        raise ValueError("Joint backend alignment changed")
    samples=m.get("samples",[])
    if not samples or [s["sample_id"] for s in samples] != list(range(len(samples))):
        raise ValueError("Joint input must contain ascending unique sample IDs")
    for s in samples:
        n=s["length"]
        if (type(n) is not int or n<=0 or n%(2*world) or n>=SAMPLE_STRIDE-1
                or s["original_length"]!=n or s["loss_prefix_tokens"]!=n//2
                or s["sample_uid"]!=f"{m['case_id']}:{s['sample_id']:06d}"):
            raise ValueError("Joint sample length, mask or identity changed")
    rule=m["content_rule"]
    expected=dict(name="lcg64_next_token_v1",vocab_size=32768,initial_state=m["seed"],
        modulus=MODULUS,multiplier=MULTIPLIER,increment=INCREMENT,sample_stride=SAMPLE_STRIDE,
        token="upper_32_bits_mod_vocab",label="token_at_position_plus_one_in_same_sample_stream",
        position="zero_based_sample_position",loss_mask="position < length // 2")
    if rule!=expected:
        raise ValueError("Joint deterministic content rule changed")
    if m["input_tokens"]!=sum(s["length"] for s in samples) or m["loss_tokens"]!=sum(s["loss_prefix_tokens"] for s in samples):
        raise ValueError("Joint token total changed")


def packet_launch_metadata(manifest, world, capacity):
    """Native mock fixture metadata; real items come from the packet source."""
    from two_dimensional_inputs import validate_manifest as validate_any
    validate_any(manifest)
    if world != manifest.get("world",16) or capacity<=0:
        raise ValueError("Packet launch world/capacity mismatch")
    if max(s["length"] for s in manifest["samples"]) > world*capacity:
        raise ValueError("Ingress allocation cannot hold the longest sample")
    return (1,world*capacity,manifest["input_tokens"]//world,manifest["input_tokens"],
            manifest["loss_tokens"],str(2*world),dict(mode="packed_cp_case",lengths=[2*world],prompt_lengths=[world]))
