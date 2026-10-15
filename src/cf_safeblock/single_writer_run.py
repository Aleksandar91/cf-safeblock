"""Locked kill experiment. Harness checks gate the curve."""

from __future__ import annotations

import json
import os

import numpy as np

from cf_safeblock.fashion_io import read_idx, read_manifest
from cf_safeblock.fashion_split import N_CLASSES, N_CLIENTS, development_indices
from cf_safeblock.local_head import (
    flipped_labels,
    initialize_head,
    local_shuffle_seed,
    train_linear_head,
)
from cf_safeblock.representation import featurize, load_canonical_trunk
from cf_safeblock.single_writer_metrics import RHOS, SEEDS, scores, kill_verdict
from cf_safeblock.updates import assert_training_ids_exclude_final_test

N_ROUNDS = 30
WARMUP_ROUNDS = 10
OWNER_RECALL_FLOOR = 0.90
FEDAVG_RECALL_CEILING = 0.02


def load_bundle(root: str) -> dict:
    manifest = read_manifest(os.path.join(root, "manifest.csv"))
    identity = json.load(open(os.path.join(root, "phi", "phi_identity.json")))
    trunk = load_canonical_trunk(open(os.path.join(root, "phi", "phi_trunk.f64bin"), "rb").read())
    images = read_idx(os.path.join(root, "raw", "train-images-idx3-ubyte.gz"))
    labels = read_idx(os.path.join(root, "raw", "train-labels-idx1-ubyte.gz"))
    clients = []
    train_ids = []
    for client_id in range(N_CLIENTS):
        rows = [
            row for row in manifest
            if row["role"] == "client_train" and row["client_id"] == client_id
        ]
        indices = [row["official_index"] for row in rows]
        train_ids.extend(row["image_id"] for row in rows)
        clients.append(_features(images, labels, indices, trunk))
    evaluation_indices = development_indices(manifest, "repr_select")
    assert_training_ids_exclude_final_test(
        train_ids,
        [row["image_id"] for row in manifest if row["role"] == "final_test"],
    )
    return {
        "clients": clients,
        "evaluation": _features(images, labels, evaluation_indices, trunk),
        "trunk_hash": identity["parameters_sha256"],
    }


def run_kill_experiment(root: str, output_dir: str) -> dict:
    bundle = load_bundle(root)
    os.makedirs(output_dir, exist_ok=True)
    owner = owner_only_check(bundle)
    _write_json(os.path.join(output_dir, "owner_only.json"), owner)
    print("owner-only recall %.3f" % owner["source_recall"], flush=True)
    if owner["source_recall"] < OWNER_RECALL_FLOOR:
        halted = {"passed": False, "stop_reasons": ["owner_only_harness_failed"], "curve_started": False}
        _write_json(os.path.join(output_dir, "verdict.json"), halted)
        return halted

    anchor = run_condition(bundle, 1101, 1.0, "clean")
    _write_json(os.path.join(output_dir, "1101_rho1_clean.json"), anchor)
    print("rho=1 clean 1101 recall %.3f" % anchor["source_recall"], flush=True)
    if anchor["source_recall"] > FEDAVG_RECALL_CEILING:
        halted = {"passed": False, "stop_reasons": ["fedavg_harness_failed"], "curve_started": False}
        _write_json(os.path.join(output_dir, "verdict.json"), halted)
        return halted

    runs = [anchor]
    for seed in SEEDS:
        for rho in RHOS:
            for condition in ("clean", "attack"):
                if seed == 1101 and rho == 1.0 and condition == "clean":
                    continue
                print("run seed %s rho %s %s" % (seed, rho, condition), flush=True)
                record = run_condition(bundle, seed, rho, condition)
                runs.append(record)
                _write_json(
                    os.path.join(output_dir, "%s_rho%s_%s.json" % (seed, _rho_name(rho), condition)),
                    record,
                )
    verdict = kill_verdict(runs)
    verdict["curve_started"] = True
    _write_json(os.path.join(output_dir, "verdict.json"), verdict)
    return verdict


def owner_only_check(bundle: dict) -> dict:
    features, labels = bundle["clients"][0]
    weight, bias = initialize_head(N_CLASSES, features.shape[1], 1101)
    trained_weight, trained_bias = train_linear_head(features, labels, weight, bias, seed=1101)
    predicted = _predict(trained_weight, trained_bias, bundle["evaluation"][0])
    result = scores(bundle["evaluation"][1], predicted)
    result.update({"seed": 1101, "epochs": 1, "client_id": 0})
    return result


def run_condition(bundle: dict, seed: int, rho: float, condition: str) -> dict:
    attack = condition == "attack"
    features0 = bundle["clients"][0][0]
    weight, bias = initialize_head(N_CLASSES, features0.shape[1], seed)
    history = []
    for round_index in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        counts = []
        for client_id, (features, labels) in enumerate(bundle["clients"]):
            round_labels = flipped_labels(labels, client_id, round_index, attack)
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
            counts.append(features.shape[0])
        weight, bias = _average(local_weights, local_biases, counts)
        if round_index == N_ROUNDS - 1 or round_index % 5 == 0:
            current = _evaluate(weight, bias, bundle)
            history.append({"round": round_index, **current})
            print(
                "  round %s recall %.3f asr %.3f" % (
                    round_index, current["source_recall"], current["asr"]
                ),
                flush=True,
            )
    final = history[-1]
    return {
        "seed": seed,
        "rho": rho,
        "condition": condition,
        "source_recall": final["source_recall"],
        "asr": final["asr"],
        "macro_f1": final["macro_f1"],
        "rounds": history,
    }


def _evaluate(weight, bias, bundle):
    return scores(bundle["evaluation"][1], _predict(weight, bias, bundle["evaluation"][0]))


def _predict(weight, bias, features):
    return np.argmax(features.dot(weight.T) + bias, axis=1)


def _average(weights, biases, counts):
    total = float(sum(counts))
    weight = np.zeros_like(weights[0])
    bias = np.zeros_like(biases[0])
    for local_weight, local_bias, count in zip(weights, biases, counts):
        scale = count / total
        weight += scale * local_weight
        bias += scale * local_bias
    return weight, bias


def _features(images, labels, indices, trunk):
    chosen = np.asarray(indices, dtype=np.int64)
    pixels = images[chosen].astype(np.float64) / 255.0
    return featurize(pixels, trunk), labels[chosen].astype(np.int64)


def _rho_name(rho: float) -> str:
    if rho == 1.0:
        return "1"
    if rho == 0.5:
        return "0.5"
    if rho == 0.0:
        return "0"
    raise ValueError("rho is outside the locked set")


def _write_json(path, payload) -> None:
    with open(path, "w") as handle:
        json.dump(_plain(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
