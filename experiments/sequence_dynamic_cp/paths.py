"""Shared paths for current CP execution and retained calibration evidence."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COEFFICIENTS = ROOT / "experiments/performance_model/results/cp1_20261009_p1_v2/stack_coefficients.json"
