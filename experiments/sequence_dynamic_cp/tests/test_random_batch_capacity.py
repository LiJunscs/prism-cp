"""Finite stochastic comparisons must not average per-batch speedups."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from random_batch_capacity_analysis import capacity_shortlist, integrate


def test_capacity_rule_preserves_fractional_non_power_of_two_thresholds():
    values = capacity_shortlist([248 * 1024] * 8 + [4096] * 64)
    assert 15872 in values  # 15.5Ki, the exact CP16 threshold.
    assert 31 * 1024 in values
    assert all(v % 32 == 0 and v * 16 >= 248 * 1024 for v in values)


def test_expected_time_and_tokens_choose_one_global_capacity_and_keep_n_fallback():
    def row(name, time, peak, cap=None):
        return dict(id=name, capacity_tokens=cap, scores={"test": dict(
            predicted_update_ms=time, peak_gib=peak, model_fits=True)})

    cases = [dict(case="a", input_tokens=100, simple_capacity_tokens=[16384, 24576],
                  native=[row("N_K16", 100, 10, 16384), row("N_K24", 80, 20, 24576)],
                  joint=[row("J_a", 40, 15)]),
             dict(case="b", input_tokens=900, simple_capacity_tokens=[16384, 24576],
                  native=[row("N_K16", 200, 30, 16384), row("N_K24", 240, 40, 24576)],
                  joint=[row("J_b", 220, 25)])]
    result = integrate(cases, {"a": 0.5, "b": 0.5}, "test", "physical")
    assert result["fixed_capacity_tokens"] == 16384
    assert result["fixed"]["expected_update_ms"] == 150
    assert result["simple"]["expected_update_ms"] == 140
    assert result["selected"]["expected_update_ms"] == 120
    assert result["selection_gain_vs_simple_pct"] == pytest.approx(100 / 6)
    assert result["selected"]["effective_tokens_per_second"] == pytest.approx(500000 / 120)
    assert result["selected"]["worst_case_model_peak_gib"] == 30
    assert result["joint_selected_probability"] == 0.5
    assert integrate(list(reversed(cases)), {"a": 0.5, "b": 0.5}, "test", "physical") == result
