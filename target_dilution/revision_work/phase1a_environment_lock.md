# Phase 1a — Execution environment clarification

Date frozen: 2026-10-14, before implementation and before any final-test scoring.
Applies to: `revision-fashion-test-v1` only.

Use the original Fashion qualification runtime: CPython 3.7.9 and NumPy 1.17.3. Both are installed locally and match the versions recorded for the original Phase 1 qualification environment. Do not use the CIFAR Phase 5 Python 3.12 / NumPy 2.5.3 environment for the Fashion head evaluation.

This clarification does not change the data, training, conditions, seeds, metrics, or analysis rule in `phase1_protocol_frozen.md`. It fixes the runtime before the evaluator is implemented and before final-test access. Record the exact runtime and the SHA-256 of all evaluator source files in the final-test input lock.
