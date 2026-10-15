"""Revision-only held-out evaluation for the Fashion-MNIST primary pair."""

from __future__ import print_function

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys

import numpy as np

from cf_safeblock.fashion_io import read_idx, read_manifest
from cf_safeblock.fashion_split import (
    FINAL_TEST_FILES,
    N_CLASSES,
    N_CLIENTS,
    OFFICIAL_MD5,
)
from cf_safeblock.local_head import (
    initialize_head,
    local_shuffle_seed,
    train_linear_head,
)
from cf_safeblock.representation import featurize, load_canonical_trunk
from cf_safeblock.single_writer_run import N_ROUNDS, WARMUP_ROUNDS, load_bundle
from cf_safeblock.target_dilution import (
    fedavg,
    move_target_fraction,
    target_monopoly_fraction,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROTOCOL_PATH = os.path.join(REPO_ROOT, "target_dilution", "revision_work", "phase1_protocol_frozen.md")
ENVIRONMENT_LOCK_PATH = os.path.join(REPO_ROOT, "target_dilution", "revision_work", "phase1a_environment_lock.md")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data", "fashion-mnist")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "target_dilution", "revision_work", "fashion_test_v1")
SEEDS = (1101, 1207, 1309, 2101, 2207)
MOVE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
SCALES = (0.0, 1.0)
SOURCE = 0
TARGET = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _input_lock(data_root, module_path, runner_path):
    raw_dir = os.path.join(data_root, "raw")
    archive_records = {}
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz") + tuple(FINAL_TEST_FILES):
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            raise IOError("required Fashion archive is missing: " + path)
        actual_md5 = md5_file(path)
        if actual_md5 != OFFICIAL_MD5[name]:
            raise ValueError("official Fashion archive checksum mismatch: " + name)
        archive_records[name] = {
            "md5": actual_md5,
            "sha256": sha256_file(path),
        }

    manifest_path = os.path.join(data_root, "manifest.csv")
    trunk_path = os.path.join(data_root, "phi", "phi_trunk.f64bin")
    identity_path = os.path.join(data_root, "phi", "phi_identity.json")
    normalization_path = os.path.join(data_root, "phi", "normalization.txt")
    requirements_path = os.path.join(REPO_ROOT, "requirements.txt")
    runtime_sources = (
        "src/cf_safeblock/fashion_io.py",
        "src/cf_safeblock/fashion_split.py",
        "src/cf_safeblock/local_head.py",
        "src/cf_safeblock/representation.py",
        "src/cf_safeblock/single_writer_run.py",
        "src/cf_safeblock/target_dilution.py",
        "src/cf_safeblock/updates.py",
        "src/cf_safeblock/revision_eval.py",
        "target_dilution/run_revision_eval.py",
    )
    protocol_hashes = {
        "phase1_protocol_frozen.md": sha256_file(PROTOCOL_PATH),
        "phase1a_environment_lock.md": sha256_file(ENVIRONMENT_LOCK_PATH),
    }
    try:
        head = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"]
        ).decode("ascii").strip()
    except Exception:
        head = "unavailable"

    return {
        "protocol_version": "revision-fashion-test-v1",
        "protocol_sha256": protocol_hashes,
        "git_head": head,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "manifest_sha256": sha256_file(manifest_path),
        "trunk_sha256": sha256_file(trunk_path),
        "trunk_identity_sha256": sha256_file(identity_path),
        "normalization_sha256": sha256_file(normalization_path),
        "requirements_sha256": sha256_file(requirements_path),
        "runtime_source_sha256": {
            path: sha256_file(os.path.join(REPO_ROOT, path))
            for path in runtime_sources
        },
        "evaluator_sha256": sha256_file(module_path),
        "runner_sha256": sha256_file(runner_path),
        "official_archives": archive_records,
    }


def _load_final_test(data_root, base_bundle):
    manifest = read_manifest(os.path.join(data_root, "manifest.csv"))
    rows = [row for row in manifest if row["role"] == "final_test"]
    rows.sort(key=lambda row: row["official_index"])
    if len(rows) != 10000:
        raise ValueError("expected 10000 final-test manifest rows")
    indices = np.asarray([row["official_index"] for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(10000, dtype=np.int64)):
        raise ValueError("final-test manifest indices are incomplete or reordered")

    raw_dir = os.path.join(data_root, "raw")
    images = read_idx(os.path.join(raw_dir, FINAL_TEST_FILES[0]))
    labels = read_idx(os.path.join(raw_dir, FINAL_TEST_FILES[1])).astype(np.int64)
    if images.shape[0] != labels.shape[0] or images.shape[0] != len(rows):
        raise ValueError("Fashion final-test image/label/manifest lengths differ")
    manifest_labels = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    if not np.array_equal(labels[indices], manifest_labels):
        raise ValueError("final-test labels differ from the locked manifest")
    class_counts = np.bincount(labels[indices], minlength=N_CLASSES)
    if not np.array_equal(class_counts, np.full(N_CLASSES, 1000, dtype=np.int64)):
        raise ValueError("Fashion final-test class counts are not 1000 each")

    trunk_path = os.path.join(data_root, "phi", "phi_trunk.f64bin")
    with open(trunk_path, "rb") as handle:
        trunk = load_canonical_trunk(handle.read())
    pixels = images[indices].astype(np.float64) / 255.0
    features = featurize(pixels, trunk)
    bundle = dict(base_bundle)
    bundle["evaluation"] = (features, labels[indices])
    bundle["evaluation_indices"] = indices
    return bundle


def _train_head(bundle, seed, honest_rho, attacked):
    features0 = bundle["clients"][0][0]
    weight, bias = initialize_head(N_CLASSES, features0.shape[1], seed)
    for round_index in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        counts = []
        for client_id, (features, labels) in enumerate(bundle["clients"]):
            local_labels = labels
            local_rho = float(honest_rho)
            if attacked and client_id == 0 and round_index >= WARMUP_ROUNDS:
                local_labels = np.array(labels, copy=True)
                local_labels[local_labels == SOURCE] = TARGET
                local_rho = 1.0
            trained_weight, trained_bias = train_linear_head(
                features,
                local_labels,
                weight,
                bias,
                local_shuffle_seed(seed, round_index, client_id),
                rho=local_rho,
            )
            local_weights.append(trained_weight)
            local_biases.append(trained_bias)
            counts.append(int(features.shape[0]))
        weight, bias = fedavg(local_weights, local_biases, counts)
    return weight, bias


def _score(bundle, weight, bias, rho, move_fraction, seed, attacked):
    eval_x, eval_y = bundle["evaluation"]
    predictions = np.argmax(eval_x @ weight.T + bias, axis=1).astype(np.uint8)
    source_mask = eval_y == SOURCE
    target_mask = eval_y == TARGET
    n_source = int(np.sum(source_mask))
    source_predictions = predictions[source_mask]
    source_counts = np.bincount(source_predictions, minlength=N_CLASSES)
    wrong_predictions = source_predictions[source_predictions != SOURCE]
    if wrong_predictions.size:
        wrong_counts = np.bincount(wrong_predictions, minlength=N_CLASSES)
        sink_class = int(np.argmax(wrong_counts))
        sink_count = int(wrong_counts[sink_class])
    else:
        sink_class = None
        sink_count = 0

    class_metrics = []
    for class_id in range(N_CLASSES):
        mask = eval_y == class_id
        n = int(np.sum(mask))
        correct = int(np.sum(predictions[mask] == class_id))
        class_metrics.append({
            "class_id": class_id,
            "n": n,
            "correct": correct,
            "recall": (float(correct) / n) if n else None,
        })
    total = int(eval_y.size)
    correct_total = int(np.sum(predictions == eval_y))
    target_count = int(source_counts[TARGET])
    result = {
        "protocol_version": "revision-fashion-test-v1",
        "split": "final_test" if "evaluation_indices" in bundle else "repr_select",
        "seed": int(seed),
        "rho": float(rho),
        "move_fraction": float(move_fraction),
        "measured_Mt": float(target_monopoly_fraction(bundle, TARGET)),
        "condition": "attack" if attacked else "clean",
        "source": SOURCE,
        "target": TARGET,
        "n_total": total,
        "accuracy_correct": correct_total,
        "accuracy": float(correct_total) / total,
        "n_source": n_source,
        "source_correct": int(source_counts[SOURCE]),
        "source_recall": float(source_counts[SOURCE]) / n_source,
        "intended_count": target_count,
        "intended_asr": float(target_count) / n_source,
        "source_prediction_counts": [int(value) for value in source_counts],
        "sink_class": sink_class,
        "sink_count": sink_count,
        "sink_share": float(sink_count) / n_source,
        "target_class_recall": class_metrics[TARGET]["recall"],
        "per_class": class_metrics,
        "source_test_indices": np.asarray(bundle.get(
            "evaluation_indices", np.arange(eval_y.size)
        ), dtype=np.int64)[source_mask],
        "source_predictions": source_predictions,
    }
    return result, predictions


def _fraction_name(value):
    return ("%.2f" % value).replace(".", "p")


def _condition_key(rho, fraction, seed, condition):
    return "rho%s_mt%s_%s_seed%s" % (
        _fraction_name(rho), _fraction_name(fraction), condition, seed
    )


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _expected_dev_record(rho, fraction, attacked, seed):
    if attacked and rho == 0.0:
        name = "mt_%s_seed_%s.json" % (_fraction_name(fraction), seed)
        return _load_json(os.path.join(
            REPO_ROOT, "target_dilution", "phase1", "results", name
        ))
    if attacked and rho == 1.0:
        name = "mt_%s_seed_%s.json" % (_fraction_name(fraction), seed)
        return _load_json(os.path.join(
            REPO_ROOT, "target_dilution", "phase4_rho1", "fashion_dose", name
        ))
    baseline = _load_json(os.path.join(
        REPO_ROOT, "target_dilution", "clean_baseline", "verdict.json"
    ))
    for row in baseline["rows"]:
        if row["rho"] == rho and row["move_fraction"] == fraction:
            for seed_row in row["seeds"]:
                if seed_row["seed"] == seed:
                    return seed_row
    raise ValueError("development clean baseline record not found")


def _compare_smoke(actual, expected, attacked):
    expected_asr = expected["intended_asr"] if attacked else expected["predicted_as_class_1"]
    fields = {
        "source_recall": expected["source_recall"],
        "intended_asr": expected_asr,
        "sink_share": expected["sink_share"],
    }
    if "sink_class" in expected:
        fields["sink_class"] = expected["sink_class"]
    mismatches = []
    for name, value in fields.items():
        got = actual[name]
        if isinstance(value, float):
            if abs(got - value) > 1e-12:
                mismatches.append((name, got, value))
        elif got != value:
            mismatches.append((name, got, value))
    return mismatches


def run_development_smoke(data_root):
    base = load_bundle(data_root)
    checks = []
    for rho in SCALES:
        for fraction in MOVE_FRACTIONS:
            moved = move_target_fraction(base, TARGET, fraction)
            for seed in SEEDS:
                attacked = True
                weight, bias = _train_head(moved, seed, rho, attacked)
                actual, _ = _score(moved, weight, bias, rho, fraction, seed, attacked)
                expected = _expected_dev_record(rho, fraction, attacked, seed)
                mismatches = _compare_smoke(actual, expected, attacked)
                checks.append({
                    "rho": rho,
                    "move_fraction": fraction,
                    "condition": "attack",
                    "seed": seed,
                    "mismatches": mismatches,
                })
        for fraction in (0.0, 1.0):
            moved = move_target_fraction(base, TARGET, fraction)
            for seed in SEEDS:
                attacked = False
                weight, bias = _train_head(moved, seed, rho, attacked)
                actual, _ = _score(moved, weight, bias, rho, fraction, seed, attacked)
                expected = _expected_dev_record(rho, fraction, attacked, seed)
                mismatches = _compare_smoke(actual, expected, attacked)
                checks.append({
                    "rho": rho,
                    "move_fraction": fraction,
                    "condition": "clean",
                    "seed": seed,
                    "mismatches": mismatches,
                })
    failures = [check for check in checks if check["mismatches"]]
    print("development-only parity checks: %s/%s passed" % (
        len(checks) - len(failures), len(checks)
    ))
    for check in checks:
        print(check)
    if failures:
        raise AssertionError("new evaluator did not reproduce archived development anchors")
    return checks


def _write_json_new(path, payload):
    if os.path.exists(path):
        raise IOError("refusing to overwrite existing revision output: " + path)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_npz_new(path, **arrays):
    if os.path.exists(path):
        raise IOError("refusing to overwrite existing revision output: " + path)
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _summarize(records):
    by_key = {}
    for record in records:
        key = (record["rho"], record["move_fraction"], record["condition"])
        by_key.setdefault(key, []).append(record)
    summary = []
    fields = ("accuracy", "source_recall", "intended_asr", "sink_share", "target_class_recall")
    for key in sorted(by_key):
        group = by_key[key]
        values = {}
        for field in fields:
            series = [float(row[field]) for row in group]
            values[field] = {
                "mean": float(statistics.mean(series)),
                "sample_sd": float(statistics.stdev(series)) if len(series) > 1 else 0.0,
                "min": float(min(series)),
                "max": float(max(series)),
            }
        summary.append({
            "rho": key[0],
            "move_fraction": key[1],
            "measured_Mt": group[0]["measured_Mt"],
            "condition": key[2],
            "n_seeds": len(group),
            "seed_metrics": values,
        })
    return summary


def _transition(clean_record, attack_record):
    clean_indices = clean_record["source_test_indices"]
    attack_indices = attack_record["source_test_indices"]
    if not np.array_equal(clean_indices, attack_indices):
        raise ValueError("paired clean/attack source test IDs differ")
    clean = clean_record["source_predictions"].astype(np.int64)
    attack = attack_record["source_predictions"].astype(np.int64)
    table = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(table, (clean, attack), 1)
    entrants = int(np.sum((clean != TARGET) & (attack == TARGET)))
    exits = int(np.sum((clean == TARGET) & (attack != TARGET)))
    return {
        "n_source_images": int(clean.size),
        "from_prediction_rows_to_attack_prediction_columns": table.tolist(),
        "entered_target": entrants,
        "exited_target": exits,
        "net_target_count_change": entrants - exits,
        "clean_target_count": int(np.sum(clean == TARGET)),
        "attack_target_count": int(np.sum(attack == TARGET)),
    }


def run_final_test(data_root, output_dir, module_path, runner_path, resume=False):
    output_dir = os.path.abspath(output_dir)
    if os.path.commonpath([REPO_ROOT, output_dir]) != REPO_ROOT:
        raise ValueError("revision outputs must stay inside CF-SafeBlock")
    os.makedirs(output_dir, exist_ok=True)
    lock_path = os.path.join(output_dir, "input_lock.json")
    current_lock = _input_lock(data_root, module_path, runner_path)

    if os.path.isfile(lock_path):
        if not resume:
            raise IOError("output already started; use resume only for the frozen matrix")
        if _load_json(lock_path) != current_lock:
            raise ValueError("input/code lock differs; refusing to resume")
    else:
        existing = os.listdir(output_dir)
        if existing:
            raise IOError("output directory is not empty and has no input lock")
        _write_json_new(lock_path, current_lock)

    runs_dir = os.path.join(output_dir, "runs")
    predictions_dir = os.path.join(output_dir, "source_predictions")
    transitions_dir = os.path.join(output_dir, "transitions")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(transitions_dir, exist_ok=True)

    base = load_bundle(data_root)
    test_bundle = _load_final_test(data_root, base)
    records = []
    cached_source_predictions = {}
    for rho in SCALES:
        for fraction in MOVE_FRACTIONS:
            moved = move_target_fraction(test_bundle, TARGET, fraction)
            measured_mt = target_monopoly_fraction(moved, TARGET)
            for condition in ("clean", "attack"):
                attacked = condition == "attack"
                for seed in SEEDS:
                    key = _condition_key(rho, fraction, seed, condition)
                    result_path = os.path.join(runs_dir, key + ".json")
                    prediction_path = os.path.join(predictions_dir, key + ".npz")
                    result_exists = os.path.exists(result_path)
                    prediction_exists = os.path.exists(prediction_path)
                    if result_exists != prediction_exists:
                        raise IOError("partial cell output; preserve it and inspect before resuming")
                    if result_exists:
                        if not resume:
                            raise IOError("cell already exists; refusing to overwrite: " + key)
                        record = _load_json(result_path)
                        with np.load(prediction_path) as stored:
                            cached_source_predictions[key] = {
                                "indices": stored["indices"].copy(),
                                "predictions": stored["predictions"].copy(),
                            }
                        records.append(record)
                        continue

                    weight, bias = _train_head(moved, seed, rho, attacked)
                    result, predictions = _score(
                        moved, weight, bias, rho, fraction, seed, attacked
                    )
                    result["measured_Mt"] = measured_mt
                    result["attacker_rho"] = 1.0 if attacked else None
                    result["honest_rho"] = float(rho)
                    source_test_indices = result.pop("source_test_indices")
                    source_predictions = result.pop("source_predictions")
                    prediction_record = {
                        "indices": np.asarray(test_bundle["evaluation_indices"][
                            test_bundle["evaluation"][1] == SOURCE
                        ], dtype=np.int32),
                        "predictions": predictions[test_bundle["evaluation"][1] == SOURCE].astype(np.uint8),
                    }
                    if not np.array_equal(source_test_indices, prediction_record["indices"]):
                        raise ValueError("source prediction IDs differ from test bundle")
                    if not np.array_equal(source_predictions, prediction_record["predictions"]):
                        raise ValueError("source prediction arrays differ")
                    _write_json_new(result_path, result)
                    _write_npz_new(
                        prediction_path,
                        indices=prediction_record["indices"],
                        predictions=prediction_record["predictions"],
                    )
                    records.append(result)
                    cached_source_predictions[key] = prediction_record
                    print("completed fixed cell " + key, flush=True)

    # Store exact paired transitions only after all cells have been trained.
    for rho in SCALES:
        for fraction in MOVE_FRACTIONS:
            for seed in SEEDS:
                clean_key = _condition_key(rho, fraction, seed, "clean")
                attack_key = _condition_key(rho, fraction, seed, "attack")
                clean = cached_source_predictions[clean_key]
                attack = cached_source_predictions[attack_key]
                clean_record = {
                    "source_test_indices": clean["indices"],
                    "source_predictions": clean["predictions"],
                }
                attack_record = {
                    "source_test_indices": attack["indices"],
                    "source_predictions": attack["predictions"],
                }
                transition_path = os.path.join(
                    transitions_dir,
                    _condition_key(rho, fraction, seed, "paired") + ".json",
                )
                if not os.path.exists(transition_path):
                    _write_json_new(transition_path, {
                        "rho": float(rho),
                        "move_fraction": float(fraction),
                        "measured_Mt": float(target_monopoly_fraction(
                            move_target_fraction(test_bundle, TARGET, fraction), TARGET
                        )),
                        "seed": int(seed),
                        "transition": _transition(clean_record, attack_record),
                    })

    summary_path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(summary_path):
        serializable_records = []
        for record in records:
            serializable_records.append(record)
        _write_json_new(summary_path, {
            "protocol_version": "revision-fashion-test-v1",
            "n_cells": len(records),
            "summary": _summarize(records),
        })
    print("final-test matrix complete: %s cells" % len(records))
    return len(records)
