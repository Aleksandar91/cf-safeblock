# Multi-target label flip, locked contrast

Date locked: 2026-10-21, before any scored run in this folder.

## Question

When client 0 already owns every federated head-training image of class 0, and the target classes stay shared, does the attacker's label histogram control where true class-0 images land?

This is not a new ownership grid. No target-class images are moved. Client 0's FedAvg weight stays at the shared-target value from the existing dose.

## Fixed setup

Same Fashion-MNIST linear head as `run_flip_condition_simple`:

- frozen trunk and `client_train` partition at move fraction 0
- 10 clients, sample-weighted FedAvg, 30 rounds, warmup 10
- Adam, learning rate 1e-3, batch 32, moments reset each local epoch
- seeds 1101, 1207, 1309, 2101, 2207
- evaluation pool `repr_select`, 200 images per class
- honest scale `rho` in {0, 1}
- after warmup, client 0 uses scale 1 on the attack rounds
- honest clients keep `rho`
- clean runs never rewrite a label, and every client keeps the stated `rho`

The official test split is not scored.

## Three attack allocations

Class-0 rows on client 0 are taken in client-array order. That order is the manifest order and does not depend on the seed. Their count must be 4750.

1. `concentrate_1`: all 4750 labels become 1. This is the existing shared-target pair 0 to 1.
2. `split_1_6`: the first 2375 labels become 1 and the remaining 2375 become 6.
3. `uniform_1_9`: row `i` becomes class `(i mod 9) + 1`. Classes 1 through 7 receive 528 labels. Classes 8 and 9 receive 527.

## Gate

`concentrate_1` must reproduce the archived development seeds before the other allocations are interpreted.

At honest rho 0, intended ASR by seed must be 0.05, 0.04, 0.04, 0.04, 0.04. Source recall must be 0 and the sink must be class 6.

At honest rho 1, intended ASR by seed must be 0.055, 0.05, 0.05, 0.045, 0.05. Source recall must be 0 and the sink must be class 6.

If any of those values differ, the split runs are not a comparable extension of the existing dose.

## Metrics

For every run: source recall, full prediction histogram on true class-0 images, per-class recall, and accuracy.

For every attack seed: total variation between the attack histogram and the attacker's label allocation, and total variation between the attack histogram and the matched clean seed.

## Scope

Outputs stay in `Multi-target-LF/runs/`. This protocol does not edit the existing manuscript or any archived verdict.
