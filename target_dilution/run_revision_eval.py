#!/usr/bin/env python
"""Run the frozen revision evaluator without touching earlier phase outputs."""

from __future__ import print_function

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cf_safeblock.revision_eval import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT,
    run_development_smoke,
    run_final_test,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("smoke-dev", "score-test"),
        help="smoke-dev uses repr_select only; score-test runs the locked test matrix",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-test", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke-dev":
        run_development_smoke(args.data_root)
    else:
        run_final_test(
            args.data_root,
            args.output,
            os.path.join(ROOT, "src", "cf_safeblock", "revision_eval.py"),
            os.path.abspath(__file__),
            resume=args.resume_test,
        )


if __name__ == "__main__":
    main()
