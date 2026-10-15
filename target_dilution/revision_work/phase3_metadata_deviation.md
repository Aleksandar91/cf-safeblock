# Phase 3 — Post-scoring metadata deviation

Date discovered: 2026-10-14

The scoring run completed and wrote 100 JSON records with split=repr_select. This label came from a metadata bug: _load_final_test loaded final-test features and labels, but move_target_fraction returned a new bundle without the auxiliary evaluation_indices marker. _score uses the marker's presence to select the split label. It used the unchanged evaluation arrays for the actual predictions.

The scoring process and generated records are preserved. The analysis audit checked that every saved source prediction array has exactly the official class-0 final-test indices in the existing manifest (1,000 indices per run), and every transition table has matching clean/attack margins. The effective evaluation split is therefore final_test; only the serialized split field is incorrect.

This issue was found after scoring. In accordance with the frozen protocol, the scored records were not overwritten and the test matrix was not rerun. The correction is documented in fashion_test_v1/primary_results.md and fashion_test_v1/analysis_audit.json. Any code cleanup must be a new version and must not rescore this test set.
