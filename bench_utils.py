"""Backward-compatible module alias for :mod:`utils`."""

import sys

import utils

sys.modules[__name__] = utils
