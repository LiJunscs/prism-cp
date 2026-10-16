"""Admission preferences must treat native and joint candidates symmetrically."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_performance_policy import select, tradeoff


def test_threshold_uses_throughput_and_is_inclusive():
    result = tradeoff(110, 100, 100, 105)
    assert result["accepted"]
    assert result["throughput_gain_pct"] == pytest.approx(10)
    assert not tradeoff(110, 100, 100.01, 105)["accepted"]


def test_no_ten_percent_cap_and_physical_guard_wins():
    assert tradeoff(130, 100, 100, 112)["accepted"]
    assert not tradeoff(130, 100, 100, 112, fits=False)["accepted"]
    assert not tradeoff(130, 100, 100, 112, min_gain_per_memory_growth=3)["accepted"]


def test_common_reference_is_not_reset_and_kind_does_not_change_gate():
    rows = [dict(id="N_K16", kind="native", time=100, peak=100, fits=True),
            dict(id="N_K20", kind="native", time=90, peak=105, fits=True),
            dict(id="joint", kind="joint", time=85, peak=110, fits=True)]
    result = select(rows, "N_K16")
    assert result["selected"] == "N_K20"
    assert not result["candidates"][2]["decision"]["accepted"]
    rows[2]["kind"] = "native"
    assert select(rows, "N_K16")["candidates"][2]["decision"] == result["candidates"][2]["decision"]


def test_slower_lower_memory_does_not_replace_reference():
    rows = [dict(id="N_K16", kind="native", time=100, peak=100, fits=True),
            dict(id="joint", kind="joint", time=110, peak=90, fits=True)]
    assert select(rows, "N_K16")["selected"] == "N_K16"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_multiplier_fails(bad):
    with pytest.raises(ValueError):
        tradeoff(100, 100, 90, 105, min_gain_per_memory_growth=bad)
