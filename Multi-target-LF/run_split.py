#!/usr/bin/env python
"""Locked one-to-many label flip on the shared-target Fashion head.

Writes only under Multi-target-LF/runs/. Does not import or edit archived verdicts.
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

from cf_safeblock.local_head import initialize_head, local_shuffle_seed, train_linear_head
from cf_safeblock.single_writer_run import N_ROUNDS, WARMUP_ROUNDS, load_bundle
from cf_safeblock.target_dilution import fedavg, predict, score_attack

OUT = Path(__file__).resolve().parent / "runs"
SEEDS = (1101, 1207, 1309, 2101, 2207)
SOURCE = 0
WARMUP = WARMUP_ROUNDS
EXPECTED_N = 4750
EXPECTED_WEIGHT = 9025 / 47500.0

EXPECTED_ASR = {
    0.0: {1101: 0.05, 1207: 0.04, 1309: 0.04, 2101: 0.04, 2207: 0.04},
    1.0: {1101: 0.055, 1207: 0.05, 1309: 0.05, 2101: 0.045, 2207: 0.05},
}


def allocation(n, policy):
    if n != EXPECTED_N:
        raise ValueError("class-0 count is %s, expected %s" % (n, EXPECTED_N))
    if policy == "concentrate_1":
        labels = np.full(n, 1, dtype=np.int64)
    elif policy == "split_1_6":
        labels = np.empty(n, dtype=np.int64)
        half = n // 2
        labels[:half] = 1
        labels[half:] = 6
    elif policy == "uniform_1_9":
        labels = (np.arange(n) % 9) + 1
    else:
        raise ValueError(policy)
    return labels.astype(np.int64)


def histogram(labels, n_classes=10):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.int64)
    total = float(np.sum(counts))
    return counts, counts / total


def total_variation(p, q):
    return 0.5 * float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def per_class_recall(labels, predicted, n_classes=10):
    recalls = []
    for class_id in range(n_classes):
        mask = labels == class_id
        recalls.append(float(np.sum(predicted[mask] == class_id) / np.sum(mask)))
    return recalls


def apply_flip(labels, new_source_labels):
    rewritten = np.array(labels, copy=True)
    positions = np.flatnonzero(rewritten == SOURCE)
    if positions.shape[0] != new_source_labels.shape[0]:
        raise ValueError("flip position count does not match the locked allocation")
    rewritten[positions] = new_source_labels
    return rewritten


def train_one(bundle, seed, honest_rho, policy, assigned):
    features0 = bundle["clients"][0][0]
    weight, bias = initialize_head(10, features0.shape[1], seed)
    for round_index in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        counts = []
        for client_id, (features, labels) in enumerate(bundle["clients"]):
            round_labels = np.array(labels, copy=True)
            rho = float(honest_rho)
            if policy != "clean" and client_id == 0 and round_index >= WARMUP:
                round_labels = apply_flip(round_labels, assigned)
                rho = 1.0
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
    return weight, bias, counts


def score_run(bundle, weight, bias, policy, honest_rho, seed, assigned):
    eval_x, eval_y = bundle["evaluation"]
    predicted = predict(weight, bias, eval_x)
    mask = eval_y == SOURCE
    n_source = int(np.sum(mask))
    if n_source != 200:
        raise ValueError("expected 200 class-0 evaluation images, got %s" % n_source)
    destination_counts, destination = histogram(predicted[mask])
    scored = score_attack(eval_y, predicted, SOURCE, 1)
    recalls = per_class_recall(eval_y, predicted)
    record = {
        "policy": policy,
        "seed": int(seed),
        "honest_rho": float(honest_rho),
        "attacker_rho_after_warmup": 1.0 if policy != "clean" else float(honest_rho),
        "warmup_rounds": WARMUP,
        "rounds": N_ROUNDS,
        "evaluation": "repr_select",
        "n_source_eval": n_source,
        "source_recall": scored["source_recall"],
        "predicted_as_class_1": scored["intended_asr"],
        "sink_class": scored["sink_class"],
        "sink_share": scored["sink_share"],
        "accuracy": float(np.mean(eval_y == predicted)),
        "per_class_recall": recalls,
        "destination_counts": [int(v) for v in destination_counts],
        "destination_rate": [float(v) for v in destination],
    }
    if policy != "clean":
        label_counts, label_rate = histogram(assigned)
        record["allocation_counts"] = [int(v) for v in label_counts]
        record["allocation_rate"] = [float(v) for v in label_rate]
        record["tv_to_allocation"] = total_variation(destination, label_rate)
    return record


def run_record(bundle, seed, honest_rho, policy, assigned):
    started = time.time()
    weight, bias, counts = train_one(bundle, seed, honest_rho, policy, assigned)
    record = score_run(bundle, weight, bias, policy, honest_rho, seed, assigned)
    record["client_counts"] = counts
    record["attacker_fedavg_weight"] = counts[0] / float(sum(counts))
    record["seconds"] = round(time.time() - started, 2)
    return record


def write_record(record):
    OUT.mkdir(parents=True, exist_ok=True)
    name = "%s_rho%s_seed%s.json" % (
        record["policy"],
        ("0" if record["honest_rho"] == 0.0 else "1"),
        record["seed"],
    )
    path = OUT / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return path


def check_anchor(records):
    mismatches = []
    for record in records:
        expected = EXPECTED_ASR[record["honest_rho"]][record["seed"]]
        if abs(record["predicted_as_class_1"] - expected) > 1e-12:
            mismatches.append((record["seed"], record["honest_rho"], record["predicted_as_class_1"], expected))
        if record["source_recall"] != 0.0 or record["sink_class"] != 6:
            mismatches.append((record["seed"], record["honest_rho"], "sink", record["source_recall"], record["sink_class"]))
    return mismatches


def main():
    bundle = load_bundle("data/fashion-mnist")
    class0 = int(np.sum(bundle["clients"][0][1] == SOURCE))
    assigned = {policy: allocation(class0, policy) for policy in ("concentrate_1", "split_1_6", "uniform_1_9")}
    probe = run_record(bundle, SEEDS[0], 0.0, "concentrate_1", assigned["concentrate_1"])
    weight = probe["attacker_fedavg_weight"]
    if abs(weight - EXPECTED_WEIGHT) > 1e-12:
        raise SystemExit("attacker weight %.6f is not the shared-target weight" % weight)

    anchor = []
    for honest_rho in (0.0, 1.0):
        for seed in SEEDS:
            if honest_rho == 0.0 and seed == SEEDS[0]:
                record = probe
            else:
                record = run_record(bundle, seed, honest_rho, "concentrate_1", assigned["concentrate_1"])
            write_record(record)
            anchor.append(record)
            print(
                "anchor rho %s seed %s asr %.3f recall %.3f sink %s (%.1fs)"
                % (honest_rho, seed, record["predicted_as_class_1"], record["source_recall"], record["sink_class"], record["seconds"]),
                flush=True,
            )
    mismatches = check_anchor(anchor)
    if mismatches:
        print("ANCHOR MISMATCH", mismatches, flush=True)
        raise SystemExit(1)
    print("anchor matched the archived shared-target seeds", flush=True)

    for policy in ("clean", "split_1_6", "uniform_1_9"):
        for honest_rho in (0.0, 1.0):
            for seed in SEEDS:
                record = run_record(
                    bundle,
                    seed,
                    honest_rho,
                    policy,
                    None if policy == "clean" else assigned[policy],
                )
                write_record(record)
                print(
                    "%s rho %s seed %s recall %.3f as1 %.3f sink %s tv %s (%.1fs)"
                    % (
                        policy,
                        honest_rho,
                        seed,
                        record["source_recall"],
                        record["predicted_as_class_1"],
                        record["sink_class"],
                        ("%.3f" % record["tv_to_allocation"]) if policy != "clean" else "-",
                        record["seconds"],
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
