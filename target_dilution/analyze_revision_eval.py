#!/usr/bin/env python
"""Validate and summarize the frozen revision-fashion-test-v1 outputs."""

from __future__ import print_function

import argparse
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cf_safeblock.revision_eval import (
    MOVE_FRACTIONS,
    SCALES,
    SEEDS,
    _condition_key,
    sha256_file,
)
from cf_safeblock.fashion_io import read_manifest

DEFAULT_INPUT = os.path.join(
    ROOT, "target_dilution", "revision_work", "fashion_test_v1"
)
DEFAULT_OUTPUT = os.path.join(DEFAULT_INPUT, "reanalysis")


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _mean_sd(values):
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_and_validate(input_dir):
    records = {}
    expected = set()
    for rho in SCALES:
        for fraction in MOVE_FRACTIONS:
            for condition in ("clean", "attack"):
                for seed in SEEDS:
                    expected.add(_condition_key(rho, fraction, seed, condition))

    runs_dir = os.path.join(input_dir, "runs")
    prediction_dir = os.path.join(input_dir, "source_predictions")
    manifest = read_manifest(os.path.join(ROOT, "data", "fashion-mnist", "manifest.csv"))
    expected_source_indices = np.asarray(sorted(
        row["official_index"] for row in manifest
        if row["role"] == "final_test" and row["class_id"] == 0
    ), dtype=np.int64)
    if expected_source_indices.size != 1000:
        raise AssertionError("manifest does not identify 1000 final-test source examples")
    stored_split_values = set()
    run_files = [name for name in os.listdir(runs_dir) if name.endswith(".json")]
    if len(run_files) != 100:
        raise AssertionError("expected 100 run records, got %s" % len(run_files))

    for name in run_files:
        key = name[:-5]
        if key not in expected or key in records:
            raise AssertionError("unexpected or duplicate run key: " + key)
        row = _read_json(os.path.join(runs_dir, name))
        if row["protocol_version"] != "revision-fashion-test-v1":
            raise AssertionError("run has unexpected protocol version")
        stored_split_values.add(row["split"])
        if row["split"] != "repr_select":
            raise AssertionError("unexpected split metadata; review before analysis")
        if row["n_total"] != 10000 or row["n_source"] != 1000:
            raise AssertionError("unexpected final-test size")
        if row["accuracy_correct"] != sum(item["correct"] for item in row["per_class"]):
            raise AssertionError("overall and per-class correct counts differ")
        if row["accuracy"] != row["accuracy_correct"] / float(row["n_total"]):
            raise AssertionError("overall accuracy does not match counts")
        if len(row["per_class"]) != 10 or any(item["n"] != 1000 for item in row["per_class"]):
            raise AssertionError("expected 1000 final-test examples per class")
        if sum(row["source_prediction_counts"]) != 1000:
            raise AssertionError("source prediction counts do not sum to 1000")
        source_counts = row["source_prediction_counts"]
        if source_counts[0] != row["source_correct"]:
            raise AssertionError("source recall count mismatch")
        if source_counts[1] != row["intended_count"]:
            raise AssertionError("intended ASR count mismatch")
        if abs(row["source_recall"] - source_counts[0] / 1000.0) > 1e-12:
            raise AssertionError("source recall does not match integer count")
        if abs(row["intended_asr"] - source_counts[1] / 1000.0) > 1e-12:
            raise AssertionError("intended ASR does not match integer count")
        if row["condition"] == "clean" and row["attacker_rho"] is not None:
            raise AssertionError("clean record has attacker scale")
        if row["condition"] == "attack" and row["attacker_rho"] != 1.0:
            raise AssertionError("attacker override differs from locked protocol")

        npz_path = os.path.join(prediction_dir, key + ".npz")
        with np.load(npz_path) as saved:
            indices = saved["indices"].astype(np.int64)
            predictions = saved["predictions"].astype(np.int64)
        if indices.size != 1000 or predictions.size != 1000:
            raise AssertionError("expected 1000 paired source predictions")
        if not np.array_equal(indices, expected_source_indices):
            raise AssertionError("stored source predictions do not use final-test source IDs")
        if not np.array_equal(
            np.bincount(predictions, minlength=10), np.asarray(source_counts)
        ):
            raise AssertionError("stored source predictions disagree with run counts")
        records[key] = row

    if set(records) != expected:
        raise AssertionError("run matrix is incomplete")
    return records, stored_split_values


def _validate_transitions(input_dir, records):
    transition_dir = os.path.join(input_dir, "transitions")
    files = [name for name in os.listdir(transition_dir) if name.endswith(".json")]
    if len(files) != 50:
        raise AssertionError("expected 50 paired transition tables")
    transition_stats = {}
    for rho in SCALES:
        for fraction in MOVE_FRACTIONS:
            for seed in SEEDS:
                key = _condition_key(rho, fraction, seed, "paired")
                row = _read_json(os.path.join(transition_dir, key + ".json"))
                transition = row["transition"]
                table = np.asarray(
                    transition["from_prediction_rows_to_attack_prediction_columns"],
                    dtype=np.int64,
                )
                if table.shape != (10, 10) or int(np.sum(table)) != 1000:
                    raise AssertionError("transition table shape/count mismatch: " + key)
                clean_key = _condition_key(rho, fraction, seed, "clean")
                attack_key = _condition_key(rho, fraction, seed, "attack")
                clean = records[clean_key]
                attack = records[attack_key]
                if not np.array_equal(np.sum(table, axis=1), clean["source_prediction_counts"]):
                    raise AssertionError("transition row margins differ from clean counts")
                if not np.array_equal(np.sum(table, axis=0), attack["source_prediction_counts"]):
                    raise AssertionError("transition column margins differ from attack counts")
                entered = int(np.sum(table[:, 1]) - table[1, 1])
                exited = int(np.sum(table[1, :]) - table[1, 1])
                if entered != transition["entered_target"] or exited != transition["exited_target"]:
                    raise AssertionError("transition entry/exit counts mismatch")
                if entered - exited != attack["intended_count"] - clean["intended_count"]:
                    raise AssertionError("paired transition net differs from ASR counts")
                transition_stats[(rho, fraction, seed)] = transition
    return transition_stats


def _development_attack_mean(rho, fraction):
    if rho == 0.0:
        directory = os.path.join(ROOT, "target_dilution", "phase1", "results")
    else:
        directory = os.path.join(
            ROOT, "target_dilution", "phase4_rho1", "fashion_dose"
        )
    values = []
    for seed in SEEDS:
        name = "mt_%s_seed_%s.json" % (("%.2f" % fraction).replace(".", "p"), seed)
        row = _read_json(os.path.join(directory, name))
        values.append(float(row["intended_asr"]))
    return float(statistics.mean(values))


def _analysis(records, transition_stats, input_dir):
    table_rows = []
    for rho in SCALES:
        for fraction in MOVE_FRACTIONS:
            clean_rows = [
                records[_condition_key(rho, fraction, seed, "clean")] for seed in SEEDS
            ]
            attack_rows = [
                records[_condition_key(rho, fraction, seed, "attack")] for seed in SEEDS
            ]
            transitions = [transition_stats[(rho, fraction, seed)] for seed in SEEDS]
            clean_asr = [row["intended_asr"] for row in clean_rows]
            attack_asr = [row["intended_asr"] for row in attack_rows]
            paired_delta = [
                attack["intended_asr"] - clean["intended_asr"]
                for clean, attack in zip(clean_rows, attack_rows)
            ]
            entry_counts = [row["entered_target"] for row in transitions]
            exit_counts = [row["exited_target"] for row in transitions]
            sinks = [row["sink_class"] for row in attack_rows]
            sink_counts = {}
            for sink in sinks:
                sink_counts[str(sink)] = sink_counts.get(str(sink), 0) + 1
            mt = attack_rows[0]["measured_Mt"]
            table_rows.append({
                "rho": rho,
                "move_fraction": fraction,
                "measured_Mt": mt,
                "clean_source_recall": _mean_sd([row["source_recall"] for row in clean_rows]),
                "attack_source_recall": _mean_sd([row["source_recall"] for row in attack_rows]),
                "clean_target_rate": _mean_sd(clean_asr),
                "attack_intended_asr": _mean_sd(attack_asr),
                "paired_delta_asr": _mean_sd(paired_delta),
                "clean_sink_share": _mean_sd([row["sink_share"] for row in clean_rows]),
                "attack_sink_share": _mean_sd([row["sink_share"] for row in attack_rows]),
                "target_entries_per_seed": _mean_sd(entry_counts),
                "target_exits_per_seed": _mean_sd(exit_counts),
                "clean_accuracy": _mean_sd([row["accuracy"] for row in clean_rows]),
                "attack_accuracy": _mean_sd([row["accuracy"] for row in attack_rows]),
                "clean_target_recall": _mean_sd([row["target_class_recall"] for row in clean_rows]),
                "attack_target_recall": _mean_sd([row["target_class_recall"] for row in attack_rows]),
                "attack_sink_classes_by_seed": sinks,
                "attack_sink_seed_counts": sink_counts,
                "development_attack_asr": _development_attack_mean(rho, fraction),
            })
    return table_rows


def _write_report(path, rows, split_metadata):
    lines = [
        "# Fashion-MNIST held-out evaluation: primary pair 0 to 1",
        "",
        "All rates use the official 10,000-image final test set (1,000 per class).",
        "Means and sample SDs are across five head initializations/minibatch orders",
        "conditional on one fixed trunk, partition, and test set. The five seeds",
        "are not five independently sampled test populations.",
        "",
        "Clean target rate is the fraction of true class-0 images predicted as class 1",
        "without a label flip. Paired delta is the seed-matched attack rate minus",
        "that clean rate. Entries/exits count actual class-0 image prediction",
        "transitions into/out of class 1 between paired clean and attacked heads.",
        "ASR is intended-target ASR on attacked rows.",
        "",
        "| rho | move f | measured Mt | clean source recall | attack source recall | clean target rate | attack ASR | paired delta ASR | clean sink share | attack sink share | entries | exits | clean acc. | attack acc. | clean class-1 recall | attack class-1 recall | attack sink classes | dev attack ASR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    def fmt(value):
        return "%.3f +/- %.3f" % (value["mean"], value["sample_sd"])
    for row in rows:
        lines.append(
            "| %.1f | %.2f | %.6f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.3f |"
            % (
                row["rho"], row["move_fraction"], row["measured_Mt"],
                fmt(row["clean_source_recall"]), fmt(row["attack_source_recall"]),
                fmt(row["clean_target_rate"]), fmt(row["attack_intended_asr"]),
                fmt(row["paired_delta_asr"]), fmt(row["clean_sink_share"]),
                fmt(row["attack_sink_share"]), fmt(row["target_entries_per_seed"]),
                fmt(row["target_exits_per_seed"]), fmt(row["clean_accuracy"]),
                fmt(row["attack_accuracy"]), fmt(row["clean_target_recall"]),
                fmt(row["attack_target_recall"]),
                ",".join(str(value) for value in row["attack_sink_classes_by_seed"]),
                row["development_attack_asr"],
            )
        )
    lines.extend([
        "",
        "These results are the complete pre-specified matrix. No threshold was",
        "applied after seeing them and no cell was rerun. Differences from the",
        "development pool are descriptive, not a basis for excluding outcomes.",
        "",
        "Metadata note: the run JSONs store split=repr_select because the existing",
        "target-move helper drops the evaluation_indices marker from its returned",
        "bundle. The frozen runner loads final_test features/labels before applying",
        "that helper; all saved source-prediction indices exactly match the 1,000",
        "class-0 IDs marked final_test in the manifest. Thus this is a mislabeled",
        "split field, not a development-set evaluation. Original run JSONs were",
        "preserved unchanged. Stored split values: %s." % sorted(split_metadata),
        "",
    ])
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    records, split_metadata = _load_and_validate(input_dir)
    transitions = _validate_transitions(input_dir, records)
    rows = _analysis(records, transitions, input_dir)
    audit = {
        "protocol_version": "revision-fashion-test-v1",
        "n_run_records": len(records),
        "n_transition_records": len(transitions),
        "all_checks_passed": True,
        "n_final_test_images_per_run": 10000,
        "n_source_images_per_run": 1000,
        "stored_split_metadata": sorted(split_metadata),
        "effective_split": "final_test",
        "split_metadata_deviation": {
            "present": True,
            "cause": "move_target_fraction drops evaluation_indices; split label tests for that marker",
            "source_indices_match_final_test_manifest": True,
            "original_run_records_modified": False,
        },
        "report_rows": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    audit_path = os.path.join(output_dir, "analysis_audit.json")
    report_path = os.path.join(output_dir, "primary_results.md")
    if os.path.exists(audit_path) or os.path.exists(report_path):
        raise IOError("refusing to overwrite an existing analysis output in " + output_dir)
    _write_json(audit_path, audit)
    _write_report(report_path, rows, split_metadata)
    print("integrity checks passed: 100 runs, 100 source-prediction files, 50 transitions")
    print("wrote " + report_path)
    print("wrote " + audit_path)
    print("")
    with open(report_path, "r") as handle:
        print(handle.read())


if __name__ == "__main__":
    main()
