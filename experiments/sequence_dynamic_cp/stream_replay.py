"""Finite-value check for the full training result."""

import math


def finite_result(value):
    import torch
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite_result(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_result(v) for v in value)
    return True
