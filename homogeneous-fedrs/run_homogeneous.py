#!/usr/bin/env python
"""Homogeneous FedRS flip: the attacker keeps rho 0 and recomputes scales.

Writes only under homogeneous-fedrs/runs/. Does not edit archived verdicts.
"""

from __future__ import print_function

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
os.chdir(str(LAB))

from cf_safeblock.local_head import class_scales, initialize_head, local_shuffle_seed, train_linear_head
from cf_safeblock.single_writer_run import N_ROUNDS, WARMUP_ROUNDS, load_bundle
from cf_safeblock.target_dilution import (
    fedavg,
    move_target_fraction,
    predict,
    score_attack,
    target_monopoly_fraction,
)

OUT = Path(__file__).resolve().parent / "runs"
SEEDS = (1101, 1207, 1309, 2101, 2207)
FRACTIONS = (0.0, 1.0)
SOURCE = 0
TARGET = 1
WARMUP = WARMUP_ROUNDS


def destination_counts(predicted, mask, n_classes=10):
    counts = np.bincount(np.asarray(predicted[mask], dtype=np.int64), minlength=n_classes)
    return [int(v) for v in counts]


def train_once(bundle, seed):
    features0 = bundle["clients"][0][0]
    weight, bias = initialize_head(10, features0.shape[1], seed)
    scale_checked = False
    for round_index in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        counts = []
        for client_id, (features, labels) in enumerate(bundle["clients"]):
            round_labels = np.array(labels, copy=True)
            rho = 0.0
            if client_id == 0 and round_index >= WARMUP:
                round_labels[round_labels == SOURCE] = TARGET
                scales = class_scales(round_labels, 10, rho)
                if not scale_checked:
                    if scales[SOURCE] != 0.0 or scales[TARGET] != 1.0:
                        raise SystemExit("attacker scale after the rewrite is not the homogeneous rule")
                    if np.any(round_labels == SOURCE):
                        raise SystemExit("label 0 remains after the rewrite")
                    scale_checked = True
            trained_weight, trained_bias = train_linear_head(
                features,
                round_labels,
                weight,
                bias,
                local_shuffle_seed(seed, round_index, client_id),
                rho=rho,
            )
            local_weights.append(trained_weight)
            local_biases.append(trained_bias)
            counts.append(int(features.shape[0]))
        weight, bias = fedavg(local_weights, local_biases, counts)
    if not scale_checked:
        raise SystemExit("the flipped attacker round did not run")
    return weight, bias, counts


def score_once(bundle, weight, bias, fraction, seed, counts):
    eval_x, eval_y = bundle["evaluation"]
    predicted = predict(weight, bias, eval_x)
    mask = eval_y == SOURCE
    n_source = int(np.sum(mask))
    if n_source != 200:
        raise SystemExit("expected 200 class-0 evaluation images, got %s" % n_source)
    if str(bundle["evaluation"][0].shape[0]) != "2000":
        raise SystemExit("evaluation pool is not the 2,000-image development pool")
    metrics = score_attack(eval_y, predicted, SOURCE, TARGET)
    class1 = eval_y == TARGET
    record = {
        "seed": int(seed),
        "source": SOURCE,
        "target": TARGET,
        "move_fraction": float(fraction),
        "M_t": target_monopoly_fraction(bundle, TARGET, 0),
        "honest_rho": 0.0,
        "attacker_rho": 0.0,
        "attacker_rule": "recompute_scale_from_rewritten_labels",
        "rounds": N_ROUNDS,
        "warmup_rounds": WARMUP,
        "evaluation": "repr_select",
        "n_source_eval": n_source,
        "attacker_fedavg_weight": counts[0] / float(sum(counts)),
        "destination_counts": destination_counts(predicted, mask),
        "class1_recall": float(np.sum(predicted[class1] == TARGET) / np.sum(class1)),
        "accuracy": float(np.mean(predicted == eval_y)),
        "source_predictions": [int(v) for v in predicted[mask]],
    }
    record.update(metrics)
    return record


def comparable(record):
    kept = dict(record)
    kept.pop("seconds", None)
    return kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_bundle(os.path.join("data", "fashion-mnist"))
    owner_class0 = int(np.sum(base["clients"][0][1] == SOURCE))
    if owner_class0 != 4750:
        raise SystemExit("client 0 does not hold 4750 class-0 images")
    for fraction in FRACTIONS:
        bundle = move_target_fraction(base, TARGET, fraction)
        measured = target_monopoly_fraction(bundle, TARGET, 0)
        expected_mt = 0.1 if fraction == 0.0 else 1.0
        if abs(measured - expected_mt) > 1e-12:
            raise SystemExit("measured M_t %s is not %s" % (measured, expected_mt))
        for seed in SEEDS:
            started = time.time()
            weight, bias, counts = train_once(bundle, seed)
            first = score_once(bundle, weight, bias, fraction, seed, counts)
            weight2, bias2, counts2 = train_once(bundle, seed)
            second = score_once(bundle, weight2, bias2, fraction, seed, counts2)
            if comparable(first) != comparable(second):
                raise SystemExit("repeat mismatch at fraction %s seed %s" % (fraction, seed))
            if not np.array_equal(weight, weight2) or not np.array_equal(bias, bias2):
                raise SystemExit("repeat weights differ at fraction %s seed %s" % (fraction, seed))
            first["seconds"] = round(time.time() - started, 3)
            path = OUT / ("f%s_seed%s.json" % (("0" if fraction == 0.0 else "1"), seed))
            path.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                "f=%s seed=%s recall=%.3f asr=%.3f sink=%s share=%.3f acc=%.3f"
                % (
                    fraction,
                    seed,
                    first["source_recall"],
                    first["intended_asr"],
                    first["sink_class"],
                    first["sink_share"],
                    first["accuracy"],
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
