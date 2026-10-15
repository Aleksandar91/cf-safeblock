# Data and code guide: Fashion-MNIST held-out evaluation

## Scope

This package documents and reproduces the narrow Fashion-MNIST study for the
linear-head system in `revision-fashion-test-v1`. It contains no manuscript,
review, Word file, or rendered figure. The source script for the revision
figures is included and reads only the archived score records. The trained
Fashion trunk is included as `assets/fashion_phi_trunk.bin`; official image
archives are downloaded from the Fashion-MNIST source and are not copied into
this package.

## Dataset roles

`data/fashion-mnist/manifest.csv` assigns official Fashion-MNIST images to the
following disjoint roles. The official training split contains 6,000 images per
class, of which 4,750 per class become federated head-training data.

| Role | Images per class | Use |
|---|---:|---|
| `server_probe` | 50 | Locked server-side probe; not used by the revision score |
| `attacker_surrogate` | 200 | Reserved surrogate pool; not used by this evaluation |
| `development / repr_train` | 800 | Fits the frozen representation trunk |
| `development / repr_select` | 200 | Selects the trunk epoch and supplies earlier exploratory scores |
| `client_train` | 4,750 | Federated linear-head training; all class-0 rows belong to client 0 |
| `final_test` | 1,000 | Official test examples per class; used only by the frozen revision evaluation |

The `repr_select` pool was used to select the trunk epoch. The final-test
evaluation is a held-out evaluation of that already selected pipeline; it is
not an independently preregistered replication. The manifest is tracked. The
four official IDX archives and the local trunk under `data/` are ignored by
Git; `prepare_inputs.py` restores the exact trunk from this package and can
download only missing archives when requested.

## Model and training implementation

The trunk is a 784-256-128 ReLU MLP trained on `repr_train`, selected by
cross-entropy on `repr_select`, and then frozen. Input pixels are converted to
float64 and divided by 255. The output head used to select the trunk is
discarded. The federated model is a ten-class linear head trained for 30 rounds
with 10 clients, one local epoch per client per round, Adam (learning rate
`1e-3`, batch size 32, moments reset each local epoch), and sample-weighted
FedAvg. The attack begins after round 10: client 0 changes its local class-0
labels to class 1 and uses scale `rho=1`; honest clients retain the tested
scale. The exact constants and deterministic partition/move operation are in
the frozen source files listed by `fashion_test_v1/input_lock.json`.

## Code map

| File | Purpose |
|---|---|
| `src/cf_safeblock/revision_eval.py` | Frozen evaluator used for all 100 held-out heads |
| `target_dilution/run_revision_eval.py` | Development parity smoke and explicit scoring entry point |
| `target_dilution/revision_work/reproduction/run_revision_eval_locked.py` | Checks CPython and NumPy versions before invoking the hash-locked runner |
| `target_dilution/analyze_revision_eval.py` | Checks counts, class metrics, source indices, transition margins, and creates the primary results table |
| `target_dilution/revision_work/reproduction/make_revision_figures.py` | Regenerates the categorical development pair panel, corrected Table 6 schematic, and held-out Fashion dose/transition plot from archived scores |
| `target_dilution/revision_work/fashion_test_v1/input_lock.json` | Hashes of protocol, runtime source, data, model, and environment used for scoring |
| `target_dilution/revision_work/fashion_test_v1/runs/` | 100 per-seed clean/attack JSON records |
| `target_dilution/revision_work/fashion_test_v1/source_predictions/` | Source-class prediction arrays and official image indices for paired transitions |
| `target_dilution/revision_work/fashion_test_v1/transitions/` | 50 paired clean-to-attack transition tables |
| `target_dilution/revision_work/fashion_test_v1/primary_results.md` | Main results table and metadata-deviation disclosure |
| `target_dilution/revision_work/phase1_protocol_frozen.md` | Frozen protocol and analysis rules |
| `target_dilution/revision_work/phase1a_environment_lock.md` | Runtime lock fixed before scoring |

The score JSON field `split` is incorrectly serialized as `repr_select` in all
100 records. The evaluator loaded the official final-test arrays before a
partition helper dropped an auxiliary index marker used only for that label.
All 100 saved source-prediction index arrays match exactly the manifest's
1,000 class-0 `final_test` indices; transition margins and score counts also
pass the integrity audit. The JSONs and evaluator were preserved unchanged.
`analysis_audit.json` and `phase3_metadata_deviation.md` record the evidence.

## Reproduction steps

From the repository root, create a CPython 3.7.9 environment and install
the pinned NumPy version:

```text
python -m pip install -r target_dilution/revision_work/reproduction/requirements-fashion-test-v1.txt
python target_dilution/revision_work/reproduction/prepare_inputs.py --download-missing
python target_dilution/revision_work/reproduction/run_revision_eval_locked.py smoke-dev
python target_dilution/analyze_revision_eval.py
python target_dilution/revision_work/reproduction/make_revision_figures.py
```

`prepare_inputs.py` verifies the protocol and runtime-source hashes, the
tracked manifest and trunk metadata, restores the
frozen trunk only if its working copy is absent, and downloads only missing
archives after the explicit flag. It never replaces an existing archive or
checkpoint. The smoke command uses only `repr_select` and compares the evaluator
with stored development anchors. The analysis command validates the archived
held-out matrix and writes report and audit sidecars to a new `reanalysis/`
folder. It refuses to overwrite an existing reanalysis. The figure command
writes only new revision-specific figure names under
`target_dilution/manuscript/figures/`; it does not overwrite the original
development figures.

The original 100-cell held-out score is already archived. The guarded scoring
entry point is `python target_dilution/revision_work/reproduction/run_revision_eval_locked.py score-test`; it writes
only to `target_dilution/revision_work/fashion_test_v1/`, refuses to overwrite,
and requires an explicit resume flag for an interrupted matrix. Do not rerun
the final-test matrix as part of reproducing the archived paper tables; use the
stored per-seed outputs and analysis. Any new test-set scoring is a separately
dated study amendment under the frozen protocol.

## Push scope

Candidate reproducibility files for a user-controlled Git push are the source
and revision scripts, pinned environment file, phase records, manifest,
held-out run JSON/NPZ/transition outputs, this guide, and the 1.8 MB frozen
trunk asset. The local manuscript and all contents of
`target_dilution/manuscript/` are outside this package and must not be added to
the release. No Git operation was performed by this preparation.
