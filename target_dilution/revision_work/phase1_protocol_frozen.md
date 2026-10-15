# Phase 1 — Frozen Fashion held-out evaluation protocol

Protocol version: `revision-fashion-test-v1`
Freeze date: 2026-10-14
Purpose: independent evaluation of the already selected primary Fashion-MNIST experiment.

## Relationship to earlier work

This is a new, explicitly dated evaluation amendment. It does not edit or reinterpret the locked Phase 1–5 protocol, any earlier JSON, the CIFAR-10 verdict, or the October 3 Low/High cell. The earlier Fashion results remain development-set results. This amendment authorizes the first scoring use of the official Fashion-MNIST `final_test` images for this new revision only.

The trunk, its checkpoint, and its selected epoch are frozen as they currently stand. The trunk was selected using `repr_select`, so the final-test results are a held-out evaluation of the fixed pipeline; they are not a preregistered replication of the original development analysis. `final_test` will not be used for training, preprocessing choices, model selection, seed selection, threshold changes, or additional tuning.

## Data and locked model inputs

- Dataset: official Fashion-MNIST.
- Test pool: all 10,000 rows marked `role=final_test` in the existing manifest; expected 1,000 examples per class.
- Client-training pool and partition: exactly the existing `client_train` rows and manifest. No image or label is reassigned except the existing target-move operation below.
- Representation: existing 128-dimensional frozen trunk, loaded from `data/fashion-mnist/phi/phi_trunk.f64bin`; do not retrain, replace, or select another epoch.
- Primary ordered pair: source 0 to target 1.
- Seeds: `1101, 1207, 1309, 2101, 2207`.
- Federated head: existing implementation and constants, 10 clients, 30 rounds, warmup 10, one local epoch per client per round, Adam, batch size 32, learning rate `1e-3`, moments reset per local epoch, sample-weighted FedAvg.

Before scoring, record SHA-256 hashes for the source revision, manifest, trunk, four official Fashion IDX archives, and evaluator. If any locked input is absent or its official archive check fails, stop before evaluating the final-test examples.

## Full fixed evaluation matrix

Run the Cartesian product of:

- Scale condition: honest clients use `rho=0` or `rho=1`.
- Target move fraction: `0, 0.25, 0.50, 0.75, 1.00`, using the existing deterministic move function and its measured `M_t` values.
- Label condition: clean (no relabeling) or attack.
- Seed: all five seeds above.

This is 100 trained heads. The clean and attack head for a given scale, move fraction, and seed share the same initialization and deterministic minibatch-order seeds.

Attack implementation stays exactly as in the archived experiment: from round 10, client 0 maps its local class-0 labels to class 1 and uses `rho=1`; honest clients retain the specified scale. Thus at honest `rho=0` this is the documented asymmetric attacker. At honest `rho=1` the attacker override has no effect. Clean runs use the stated scale on every client and do not change labels.

No other pair, model, optimizer, round count, local epoch, scale, threshold, or attack variant is added to this evaluation.

## Evaluation and stored outcomes

Evaluate each final global head once on all final-test images. Store integer counts and per-seed values for:

- overall accuracy and recall for every class;
- for true class 0: source recall, intended ASR to class 1, complete predicted-class counts, sink class and sink share;
- for clean runs: class-1 prediction rate on true class-0 examples, plus class-1 recall;
- for every clean/attack seed pair: the exact 10-by-10 prediction-transition table on the same true-class-0 images, including the net change in predictions to class 1.

Retain all five per-seed records. Report their mean, sample standard deviation, range, and exact counts. Do not treat the five seeds as independent samples of clients, partitions, trunks, or images. Do not pool five copies of the same test images as if they were 5,000 independent examples. Any image-resampling interval must resample matched example IDs jointly across the clean/attack predictions and be labelled conditional on these five trained heads.

## Analysis rule

There is no binary pass/fail gate and no result-dependent rerun. Report the full matrix, including null or contrary outcomes. The independent results may narrow or change the manuscript's claims; they do not authorize changing this protocol and rescoring the same test set under another model or threshold.

A code or input mismatch discovered before any final-test scoring means stop and correct the implementation before freezing the evaluator hash. A mismatch discovered after scoring means preserve the outputs, document the deviation, and do not silently overwrite or rerun them. Any further scoring requires a separate dated amendment and a genuinely new evaluation set.

## Output and safety boundaries

- New code: `src/cf_safeblock/revision_eval.py` and a dedicated runner under `target_dilution/`.
- New outputs only: `target_dilution/revision_work/fashion_test_v1/`.
- Never overwrite prior phase results, source archives, manifest, trunk, manuscript, or Phase 5 outputs.
- No Git commit or push is part of this task.
- The reproducibility bundle will exclude `target_dilution/manuscript/**`, including the draft, reviews, figures, and Word files. Manuscript changes, if later needed, remain separate and will not be pushed by the agent.

## Phase gate

Proceed to implementation only after a development-pool dry run confirms that the evaluator reproduces the existing primary Fashion metrics and writes to the new output tree. Inspect the evaluator and its recorded hashes before the first final-test scoring pass. Once scoring begins, the analysis rule above applies.
