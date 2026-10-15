#!/usr/bin/env python
"""Enforce the frozen runtime, then invoke the hash-locked evaluator runner."""

from __future__ import print_function

import os
import sys


EXPECTED_PYTHON = (3, 7, 9)
EXPECTED_NUMPY = "1.17.3"

if sys.version_info[:3] != EXPECTED_PYTHON:
    raise SystemExit(
        "revision-fashion-test-v1 requires CPython 3.7.9; got %s"
        % ".".join(str(value) for value in sys.version_info[:3])
    )

try:
    import numpy as np
except ImportError:
    raise SystemExit("NumPy is missing; install the pinned requirements file")

if np.__version__ != EXPECTED_NUMPY:
    raise SystemExit(
        "revision-fashion-test-v1 requires NumPy %s; got %s"
        % (EXPECTED_NUMPY, np.__version__)
    )

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "target_dilution"))

import run_revision_eval


if __name__ == "__main__":
    run_revision_eval.main()
