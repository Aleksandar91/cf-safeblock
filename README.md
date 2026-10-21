# Score archive and held-out Fashion evaluation

This repository contains the scored records behind the reported tables and the code that checks the locked Fashion-MNIST test evaluation. It does not contain a manuscript, review notes, or Word files. The official Fashion-MNIST image archives are not stored here; the preparation script can download them.

`SHA256SUMS` lists the SHA-256 of every archived file except this README, with two spaces before the path.

## Development scores

Each `verdict.json` includes the per-seed records. A reported mean is the arithmetic mean of the matching seed field.

On a flip row the fields are `intended_asr`, `source_recall`, `sink_class`, and `sink_share`. On a clean row, where no label was rewritten, the fraction predicted as class 1 is stored as `predicted_as_class_1`. Per-class recall is `per_class_recall`, with class 0 at index 0.

| File | Condition recorded by that file |
| --- | --- |
| `phase1/results/verdict.json` | Fashion-MNIST pair 0 to 1, label flip, missing-class scale 0, five move fractions |
| `phase2/results/verdict.json` | Ten Fashion-MNIST pairs, label flip, missing-class scale 0 |
| `phase3/results/verdict.json` | MNIST pair 0 to 1, label flip, missing-class scale 0 |
| `phase3/pv_results/verdict.json` | Four-class Apple slice, label flip, shared regime and target monopoly |
| `phase4_rho1/fashion_dose/verdict.json` | Fashion-MNIST pair 0 to 1, label flip, ordinary softmax, five move fractions |
| `phase4_rho1/fashion_pairs/verdict.json` | Ten Fashion-MNIST pairs, label flip, ordinary softmax |
| `phase4_rho1/mnist/verdict.json` | MNIST pair 0 to 1, label flip, ordinary softmax |
| `clean_baseline/verdict.json` | Fashion-MNIST, no label flip, scale 0 and scale 1, move fractions 0 and 1 |
| `clean_baseline_mnist/verdict.json` | MNIST pair 0 to 1, no label flip, scale 0 and scale 1, move fractions 0 and 1 |
| `centralized_head/verdict.json` | Fashion-MNIST pooled client-training features, one linear head, no clients, no label flip |

`phase1/results/verdict.json` does not repeat the scale inside each seed row. The condition is the one named in this table. The later flip files carry `honest_rho`. The two clean files carry `rho` and `"label_flip": false`. The centralized file is one head on the pooled sample. The MNIST clean file and the centralized file were added after the 5 October archive. The eight files above them are unchanged.

## Locked Fashion-MNIST test evaluation

The later evaluation of pair 0 to 1 uses all 10,000 official test images. Its records are under `target_dilution/revision_work/fashion_test_v1/`:

- `runs/`: 100 per-seed JSON records
- `source_predictions/`: the class-0 prediction arrays used for paired transitions
- `transitions/`: 50 paired clean-to-attack transition tables
- `summary.json`, `analysis_audit.json`, and `primary_results.md`
- `input_lock.json`: hashes of the protocol, runtime source, manifest, trunk, and environment

The guide is `target_dilution/revision_work/reproduction/DATA_AND_CODE.md`. The class-role manifest is `data/fashion-mnist/manifest.csv`. The frozen trunk asset is `target_dilution/revision_work/reproduction/assets/fashion_phi_trunk.bin`.

Every run JSON serializes `split` as `repr_select`. That label is wrong. The saved source-prediction indices match the 1,000 class-0 identifiers marked `final_test` in the manifest, and the records were left unchanged. The discrepancy is recorded in `primary_results.md`, `analysis_audit.json`, and `target_dilution/revision_work/phase3_metadata_deviation.md`.

To check the archived test tables without training, run the analysis script from the repository root. It writes a new `reanalysis/` directory and refuses to overwrite one that already exists. It does not rescore the test set.

```text
python target_dilution/analyze_revision_eval.py
```

Reconstructing the inputs requires CPython 3.7.9 and NumPy 1.17.3, as locked in `input_lock.json`. From the repository root:

```text
python -m pip install -r target_dilution/revision_work/reproduction/requirements-fashion-test-v1.txt
python target_dilution/revision_work/reproduction/prepare_inputs.py --download-missing
python target_dilution/revision_work/reproduction/run_revision_eval_locked.py smoke-dev
```

`prepare_inputs.py` checks the locked hashes, restores the trunk only when the working copy is absent, and downloads only missing official archives. The smoke command uses the development pool and compares the evaluator with stored anchors. The archived 100-head test matrix is already in `fashion_test_v1/`. Do not rerun it to regenerate the stored tables. The guarded scoring command is documented in the guide and refuses to overwrite those records.

## Shared-target allocation check

`Multi-target-LF/` records a later check on the development pool `repr_select`. Client 0 still owns every class-0 head-training image, and no target-class image is moved. The three attack allocations send those class-0 labels to class 1, split them between classes 1 and 6, or cycle them across classes 1–9. Clean heads are included. There are 40 per-seed JSON records under `Multi-target-LF/runs/`. The protocol is `Multi-target-LF/protocol.md`, and the runner is `Multi-target-LF/run_split.py`. This directory does not contain a manuscript.
