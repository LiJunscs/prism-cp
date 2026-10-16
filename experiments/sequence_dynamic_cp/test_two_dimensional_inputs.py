"""Explicit frozen lengths preserve identities, content and prefix masks."""

import copy

import pytest

from joint_cp.inputs import build_manifest
from two_dimensional_inputs import INCREMENT, MODULUS, MULTIPLIER, _advance, materialize_span, validate_manifest


def test_manifest_totals_alignment_and_identity():
    lengths = [224 * 1024] * 8 + [4096] * 64
    manifest = build_manifest(lengths, 'L224_S4', 16)
    validate_manifest(manifest)
    assert manifest['input_tokens'] == 2 * 1024**2
    assert manifest['loss_tokens'] == 1024**2
    assert [s['sample_id'] for s in manifest['samples']] == list(range(72))
    assert manifest == build_manifest(lengths, 'L224_S4', 16)
    with pytest.raises(ValueError):
        build_manifest([33], 'invalid_alignment', 16)


def test_seed_content_span_and_prefix():
    first = build_manifest([4096, 16384], 'mixed', 16, seed=2026101401)
    second = build_manifest([4096, 16384], 'mixed', 16, seed=2026101402)
    assert materialize_span(first, 0, 0, 64)['tokens'] != materialize_span(second, 0, 0, 64)['tokens']
    whole = materialize_span(first, 0, 0, 64)
    for field in whole:
        assert whole[field] == materialize_span(first, 0, 0, 17)[field] + materialize_span(first, 0, 17, 64)[field]
    assert whole['labels'][:-1] == whole['tokens'][1:]
    assert materialize_span(first, 0, 2046, 2050)['loss_mask'] == [1, 1, 0, 0]
    assert materialize_span(first, 0, 4096, 4096)['tokens'] == []
    with pytest.raises(ValueError):
        materialize_span(first, 0, 0, 4097)


def test_jump_and_semantic_rejections():
    state = 2026101401
    for steps in range(100):
        assert _advance(2026101401, steps) == state
        state = (state * MULTIPLIER + INCREMENT) % MODULUS
    manifest = build_manifest([4096, 8192], 'mixed', 16)
    for field, value in [('sample_id', 0), ('loss_prefix_tokens', 4097), ('length', 8194)]:
        broken = copy.deepcopy(manifest)
        broken['samples'][1][field] = value
        with pytest.raises(ValueError):
            validate_manifest(broken)
