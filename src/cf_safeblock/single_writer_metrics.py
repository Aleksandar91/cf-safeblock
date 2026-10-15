"""Pre-registered kill rule for the single-writer curve. No training here."""

from __future__ import annotations

import numpy as np

SEEDS = (1101, 1207, 1309)
RHOS = (1.0, 0.5, 0.0)
SOURCE_CLASS = 0
TARGET_CLASS = 1


def scores(labels, predicted, n_classes: int = 10) -> dict:
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    source = labels == SOURCE_CLASS
    if not np.any(source):
        raise ValueError("evaluation set has no source-class examples")
    source_count = float(np.sum(source))
    return {
        "macro_f1": _macro_f1(labels, predicted, n_classes),
        "source_recall": float(np.sum(predicted[source] == SOURCE_CLASS) / source_count),
        "asr": float(np.sum(predicted[source] == TARGET_CLASS) / source_count),
    }


def kill_verdict(runs: list) -> dict:
    """Apply the two locked lines. ``runs`` contains the 18 curve records."""

    restored = []
    amplified = []
    rows = []
    for seed in SEEDS:
        clean_one = _find(runs, seed, 1.0, "clean")
        clean_zero = _find(runs, seed, 0.0, "clean")
        attack_one = _find(runs, seed, 1.0, "attack")
        attack_zero = _find(runs, seed, 0.0, "attack")
        gain = clean_zero["source_recall"] - clean_one["source_recall"]
        extra_drop = (clean_zero["source_recall"] - attack_zero["source_recall"]) - (
            clean_one["source_recall"] - attack_one["source_recall"]
        )
        if gain >= 0.40:
            restored.append(seed)
        if clean_zero["source_recall"] >= 0.50 and extra_drop >= 0.20:
            amplified.append(seed)
        rows.append({
            "seed": seed,
            "clean_gain": gain,
            "extra_attack_drop": extra_drop,
            "clean_rho0": clean_zero["source_recall"],
        })
    failed_restore = len(restored) < 2
    failed_amplify = len(amplified) < 2
    reasons = []
    if failed_restore:
        reasons.append("restriction_did_not_restore_clean_recall")
    if failed_amplify:
        reasons.append("flip_did_not_gain_extra_damage")
    return {
        "passed": not reasons,
        "stop_reasons": reasons,
        "restored_seeds": restored,
        "amplified_seeds": amplified,
        "seeds": rows,
    }


def _find(runs, seed, rho, condition):
    matches = [
        run for run in runs
        if run["seed"] == seed and float(run["rho"]) == float(rho) and run["condition"] == condition
    ]
    if len(matches) != 1:
        raise ValueError("expected one record for seed %s rho %s %s" % (seed, rho, condition))
    return matches[0]


def _macro_f1(labels, predicted, n_classes: int) -> float:
    scores_by_class = []
    for class_id in range(n_classes):
        true_positive = float(np.sum((predicted == class_id) & (labels == class_id)))
        false_positive = float(np.sum((predicted == class_id) & (labels != class_id)))
        false_negative = float(np.sum((predicted != class_id) & (labels == class_id)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        if precision + recall == 0.0:
            scores_by_class.append(0.0)
        else:
            scores_by_class.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores_by_class))
