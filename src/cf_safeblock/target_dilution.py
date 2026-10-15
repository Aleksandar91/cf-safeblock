"""Target-class dilution experiments (Fashion-MNIST and optional PV-19)."""

from __future__ import annotations

import json
import os
from typing import Iterable, Sequence

import numpy as np

from cf_safeblock.fashion_split import N_CLASSES
from cf_safeblock.local_head import initialize_head, local_shuffle_seed, train_linear_head
from cf_safeblock.single_writer_run import N_ROUNDS, load_bundle

WARMUP = 10
DEFAULT_SEEDS = (1101, 1207, 1309, 2101, 2207)
MT_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def move_target_fraction(bundle: dict, target: int, frac: float) -> dict:
    """Move ``frac`` of each non-owner client's class-``target`` rows to client 0."""

    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must lie in [0, 1]")
    clients = []
    moved_features = []
    moved_labels = []
    for client_id, (features, labels) in enumerate(bundle["clients"]):
        if client_id == 0:
            clients.append((features, labels))
            continue
        mask = labels == target
        indices = np.flatnonzero(mask)
        n_move = int(round(frac * indices.size))
        if n_move <= 0:
            clients.append((features, labels))
            continue
        move_idx = indices[:n_move]
        keep_idx = np.setdiff1d(np.arange(features.shape[0]), move_idx)
        moved_features.append(features[move_idx])
        moved_labels.append(labels[move_idx])
        clients.append((features[keep_idx], labels[keep_idx]))
    owner_features, owner_labels = clients[0]
    if moved_features:
        clients[0] = (
            np.concatenate([owner_features] + moved_features, axis=0),
            np.concatenate([owner_labels] + moved_labels, axis=0),
        )
    return {"clients": clients, "evaluation": bundle["evaluation"]}


def target_monopoly_fraction(bundle: dict, target: int, owner_id: int = 0) -> float:
    total = 0
    owner = 0
    for client_id, (_, labels) in enumerate(bundle["clients"]):
        count = int(np.sum(labels == target))
        total += count
        if client_id == owner_id:
            owner = count
    return float(owner / total) if total else 0.0


def fedavg(weights, biases, counts):
    total = float(sum(counts))
    weight = sum(c / total * w for c, w in zip(counts, weights))
    bias = sum(c / total * b for c, b in zip(counts, biases))
    return weight, bias


def predict(weight, bias, features):
    return np.argmax(features @ weight.T + bias, axis=1)


def score_attack(
    labels,
    predicted,
    source: int,
    target: int,
    n_classes: int = N_CLASSES,
) -> dict:
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    mask = labels == source
    if not np.any(mask):
        raise ValueError("no source-class evaluation examples")
    n = float(np.sum(mask))
    source_recall = float(np.sum(predicted[mask] == source) / n)
    intended_asr = float(np.sum(predicted[mask] == target) / n)
    other = predicted[mask]
    other = other[other != source]
    if other.size == 0:
        sink_class = int(target)
        sink_share = 1.0 - source_recall
    else:
        values, counts = np.unique(other, return_counts=True)
        sink_class = int(values[np.argmax(counts)])
        sink_share = float(np.max(counts) / n)
    return {
        "source_recall": source_recall,
        "intended_asr": intended_asr,
        "sink_class": sink_class,
        "sink_share": sink_share,
    }


def clean_runner_up(
    bundle: dict,
    source: int,
    seed: int,
    warmup: int = WARMUP,
    honest_rho: float = 0.0,
) -> dict:
    """Train clean FedAvg and return runner-up stats for class ``source``."""

    _check_rho(honest_rho)
    features0 = bundle["clients"][0][0]
    weight, bias = initialize_head(N_CLASSES, features0.shape[1], seed)
    for round_index in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        counts = []
        for client_id, (features, labels) in enumerate(bundle["clients"]):
            trained_weight, trained_bias = train_linear_head(
                features,
                labels,
                weight,
                bias,
                local_shuffle_seed(seed, round_index, client_id),
                rho=honest_rho,
            )
            local_weights.append(trained_weight)
            local_biases.append(trained_bias)
            counts.append(features.shape[0])
        weight, bias = fedavg(local_weights, local_biases, counts)
    eval_features, eval_labels = bundle["evaluation"]
    logits = eval_features @ weight.T + bias
    logits = np.array(logits, copy=True)
    logits[:, source] = -1e9
    predicted = np.argmax(logits, axis=1)
    mask = eval_labels == source
    values, counts = np.unique(predicted[mask], return_counts=True)
    runner_up = int(values[np.argmax(counts)])
    runner_up_q = float(np.max(counts) / np.sum(mask))
    return {
        "runner_up_class": runner_up,
        "runner_up_q": runner_up_q,
        "seed": int(seed),
        "honest_rho": float(honest_rho),
        "n_eval": int(np.sum(mask)),
    }


def run_flip_condition_simple(
    bundle: dict,
    *,
    source: int,
    target: int,
    seed: int,
    owner_id: int = 0,
    warmup: int = WARMUP,
    honest_rho: float = 0.0,
) -> dict:
    """Single attack trajectory with explicit source/target flip.

    ``honest_rho`` is the FedRS scale for every client that is not applying
    the flip on that round. The attacker still uses scale 1 after warmup.
    The default 0 preserves Phases 1–3.
    """

    _check_rho(honest_rho)
    features0 = bundle["clients"][0][0]
    weight, bias = initialize_head(N_CLASSES, features0.shape[1], seed)
    for round_index in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        counts = []
        for client_id, (features, labels) in enumerate(bundle["clients"]):
            round_labels = np.array(labels, copy=True)
            rho = float(honest_rho)
            if client_id == owner_id and round_index >= warmup:
                round_labels[round_labels == source] = target
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
            counts.append(features.shape[0])
        weight, bias = fedavg(local_weights, local_biases, counts)
    predicted = predict(weight, bias, bundle["evaluation"][0])
    metrics = score_attack(bundle["evaluation"][1], predicted, source, target)
    return {
        "seed": seed,
        "source": source,
        "target": target,
        "honest_rho": float(honest_rho),
        "attacker_rho": 1.0,
        "M_t": target_monopoly_fraction(bundle, target, owner_id),
        **metrics,
    }


def phase1_dose_sweep(
    root: str,
    output_dir: str,
    *,
    source: int = 0,
    target: int = 1,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    mt_values: Sequence[float] = MT_VALUES,
) -> dict:
    base = load_bundle(root)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for frac in mt_values:
        bundle = move_target_fraction(base, target, frac)
        actual_mt = target_monopoly_fraction(bundle, target)
        seed_rows = []
        for seed in seeds:
            record = run_flip_condition_simple(
                bundle, source=source, target=target, seed=seed
            )
            record["target_monopoly_frac"] = frac
            record["M_t_measured"] = actual_mt
            seed_rows.append(record)
            path = os.path.join(
                output_dir,
                "mt_%s_seed_%s.json" % (_tag(frac), seed),
            )
            _write_json(path, record)
        row = {
            "target_monopoly_frac": frac,
            "M_t_measured": actual_mt,
            "seeds": seed_rows,
            "mean_intended_asr": float(np.mean([r["intended_asr"] for r in seed_rows])),
            "mean_sink_share": float(np.mean([r["sink_share"] for r in seed_rows])),
            "mean_source_recall": float(np.mean([r["source_recall"] for r in seed_rows])),
        }
        rows.append(row)
        print(
            "M_t=%.2f ASR=%.3f sink=%.3f recall=%.3f"
            % (actual_mt, row["mean_intended_asr"], row["mean_sink_share"], row["mean_source_recall"]),
            flush=True,
        )
    verdict = _phase1_verdict(rows)
    payload = {"rows": rows, "verdict": verdict}
    _write_json(os.path.join(output_dir, "verdict.json"), payload)
    return payload


def phase2_pair_matrix(
    root: str,
    output_dir: str,
    *,
    pairs: Sequence[tuple[int, int]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict:
    base = load_bundle(root)
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for source, target in pairs:
        runner = clean_runner_up(base, source, seeds[0])
        shared_rows = []
        mono_rows = []
        for seed in seeds:
            shared_bundle = base
            mono_bundle = move_target_fraction(base, target, 1.0)
            shared = run_flip_condition_simple(
                shared_bundle, source=source, target=target, seed=seed
            )
            mono = run_flip_condition_simple(
                mono_bundle, source=source, target=target, seed=seed
            )
            shared["regime"] = "shared_target"
            mono["regime"] = "target_monopoly"
            shared_rows.append(shared)
            mono_rows.append(mono)
        match_rate = float(
            np.mean(
                [
                    1.0 if r["sink_class"] == runner["runner_up_class"] else 0.0
                    for r in shared_rows
                ]
            )
        )
        entry = {
            "source": source,
            "target": target,
            "runner_up_class": runner["runner_up_class"],
            "runner_up_q": runner["runner_up_q"],
            "shared_target": shared_rows,
            "target_monopoly": mono_rows,
            "mean_asr_shared": float(np.mean([r["intended_asr"] for r in shared_rows])),
            "mean_asr_monopoly": float(np.mean([r["intended_asr"] for r in mono_rows])),
            "sink_matches_runner_up_rate": match_rate,
        }
        results.append(entry)
        print(
            "pair %s->%s shared_asr=%.3f mono_asr=%.3f sink@RU=%.2f"
            % (
                source,
                target,
                entry["mean_asr_shared"],
                entry["mean_asr_monopoly"],
                match_rate,
            ),
            flush=True,
        )
    payload = {"pairs": results}
    _write_json(os.path.join(output_dir, "verdict.json"), payload)
    return payload


def _phase1_verdict(rows: list) -> dict:
    """Apply gate from ``target_dilution/protocol.md`` per M_t band and seed."""

    low_mt = [r for r in rows if r["M_t_measured"] < 0.999]
    high_mt = [r for r in rows if r["M_t_measured"] >= 0.999]
    n_seeds = len(rows[0]["seeds"]) if rows else 0
    low_band_pass = 0
    high_band_pass = 0
    fail_shared = 0
    for row in low_mt:
        ok = sum(
            1
            for s in row["seeds"]
            if s["intended_asr"] <= 0.10 and s["source_recall"] <= 0.10
        )
        if ok >= 4:
            low_band_pass += 1
    shared_row = next((r for r in rows if r["target_monopoly_frac"] == 0.0), None)
    if shared_row:
        fail_shared = sum(1 for s in shared_row["seeds"] if s["intended_asr"] >= 0.50)
    for row in high_mt:
        ok = sum(1 for s in row["seeds"] if s["intended_asr"] >= 0.70)
        if ok >= 4:
            high_band_pass += 1
    passed = (
        low_band_pass == len(low_mt)
        and high_band_pass == len(high_mt)
        and fail_shared < 3
        and n_seeds >= 5
    )
    return {
        "passed": passed,
        "low_mt_bands_passed": low_band_pass,
        "low_mt_bands_total": len(low_mt),
        "high_mt_bands_passed": high_band_pass,
        "high_mt_bands_total": len(high_mt),
        "fail_shared_seeds": fail_shared,
    }


def _check_rho(rho: float) -> None:
    if not np.isfinite(rho) or rho < 0.0 or rho > 1.0:
        raise ValueError("rho must lie in [0, 1]")


# Measured M_t for Fashion pair 0→1, locked to phase1/results/verdict.json.
FASHION_DOSE_MT = {
    0.0: 0.1,
    0.25: 0.3254736842105263,
    0.5: 0.5509473684210526,
    0.75: 0.7745263157894737,
    1.0: 1.0,
}
PHASE4_PAIRS = (
    (0, 1),
    (0, 6),
    (0, 3),
    (1, 0),
    (2, 4),
    (3, 5),
    (4, 2),
    (5, 3),
    (6, 0),
    (7, 9),
)
PHASE4_RHO = 1.0


def _asr_band_counts(seed_rows: list) -> dict:
    return {
        "n_seeds": len(seed_rows),
        "n_intended_asr_le_0_10": sum(1 for row in seed_rows if row["intended_asr"] <= 0.10),
        "n_intended_asr_ge_0_70": sum(1 for row in seed_rows if row["intended_asr"] >= 0.70),
    }


def phase4_fashion_dose(
    root: str,
    output_dir: str,
    *,
    source: int = 0,
    target: int = 1,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    mt_values: Sequence[float] = MT_VALUES,
) -> dict:
    """Phase 4 dose: every client uses ordinary softmax (honest rho = 1)."""

    base = load_bundle(root)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for frac in mt_values:
        if frac not in FASHION_DOSE_MT:
            raise ValueError("move fraction %s is not in the Phase 4 lock" % frac)
        bundle = move_target_fraction(base, target, frac)
        actual_mt = target_monopoly_fraction(bundle, target)
        if actual_mt != FASHION_DOSE_MT[frac]:
            raise ValueError(
                "measured M_t %s at move %s does not match the Phase 1 archive %s"
                % (actual_mt, frac, FASHION_DOSE_MT[frac])
            )
        seed_rows = []
        for seed in seeds:
            record = run_flip_condition_simple(
                bundle,
                source=source,
                target=target,
                seed=seed,
                honest_rho=PHASE4_RHO,
            )
            record["target_monopoly_frac"] = frac
            record["M_t_measured"] = actual_mt
            seed_rows.append(record)
            _write_json(
                os.path.join(output_dir, "mt_%s_seed_%s.json" % (_tag(frac), seed)),
                record,
            )
        counts = _asr_band_counts(seed_rows)
        row = {
            "target_monopoly_frac": frac,
            "M_t_measured": actual_mt,
            "seeds": seed_rows,
            "mean_intended_asr": float(np.mean([r["intended_asr"] for r in seed_rows])),
            "mean_sink_share": float(np.mean([r["sink_share"] for r in seed_rows])),
            "mean_source_recall": float(np.mean([r["source_recall"] for r in seed_rows])),
            **counts,
        }
        rows.append(row)
        print(
            "rho=1 M_t=%.4f ASR=%.3f le010=%s ge070=%s"
            % (
                actual_mt,
                row["mean_intended_asr"],
                counts["n_intended_asr_le_0_10"],
                counts["n_intended_asr_ge_0_70"],
            ),
            flush=True,
        )
    low_rows = [row for row in rows if row["target_monopoly_frac"] < 1.0]
    high_rows = [row for row in rows if row["target_monopoly_frac"] == 1.0]
    low = all(row["n_intended_asr_le_0_10"] >= 4 for row in low_rows) and len(low_rows) == 4
    high = all(row["n_intended_asr_ge_0_70"] >= 4 for row in high_rows) and len(high_rows) == 1
    payload = {
        "honest_rho": PHASE4_RHO,
        "attacker_rho": 1.0,
        "pair": "%s->%s" % (source, target),
        "rows": rows,
        "low": low,
        "high": high,
    }
    _write_json(os.path.join(output_dir, "verdict.json"), payload)
    return payload


def phase4_fashion_pairs(
    root: str,
    output_dir: str,
    *,
    pairs: Sequence[tuple[int, int]] = PHASE4_PAIRS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict:
    """Phase 4 pair panel at honest rho = 1. Runner-up is trained at rho = 1."""

    if tuple(pairs) != PHASE4_PAIRS:
        raise ValueError("Phase 4 pairs must be the locked Phase 2 list")
    base = load_bundle(root)
    os.makedirs(output_dir, exist_ok=True)
    runners = {}
    results = []
    for source, target in pairs:
        if source not in runners:
            runners[source] = clean_runner_up(
                base, source, seeds[0], honest_rho=PHASE4_RHO
            )
        runner = runners[source]
        shared_bundle = base
        mono_bundle = move_target_fraction(base, target, 1.0)
        shared_rows = []
        mono_rows = []
        for seed in seeds:
            shared = run_flip_condition_simple(
                shared_bundle,
                source=source,
                target=target,
                seed=seed,
                honest_rho=PHASE4_RHO,
            )
            mono = run_flip_condition_simple(
                mono_bundle,
                source=source,
                target=target,
                seed=seed,
                honest_rho=PHASE4_RHO,
            )
            shared["regime"] = "shared_target"
            shared["target_monopoly_frac"] = 0.0
            mono["regime"] = "target_monopoly"
            mono["target_monopoly_frac"] = 1.0
            shared_rows.append(shared)
            mono_rows.append(mono)
        match_rate = float(
            np.mean(
                [
                    1.0 if row["sink_class"] == runner["runner_up_class"] else 0.0
                    for row in shared_rows
                ]
            )
        )
        entry = {
            "source": source,
            "target": target,
            "runner_up_class": runner["runner_up_class"],
            "runner_up_q": runner["runner_up_q"],
            "runner_up_seed": runner["seed"],
            "runner_up_honest_rho": runner["honest_rho"],
            "runner_up_n_eval": runner["n_eval"],
            "shared_target": shared_rows,
            "target_monopoly": mono_rows,
            "mean_asr_shared": float(np.mean([row["intended_asr"] for row in shared_rows])),
            "mean_asr_monopoly": float(np.mean([row["intended_asr"] for row in mono_rows])),
            "sink_matches_runner_up_rate": match_rate,
        }
        results.append(entry)
        print(
            "rho=1 pair %s->%s shared_asr=%.3f mono_asr=%.3f sink@RU=%.2f"
            % (source, target, entry["mean_asr_shared"], entry["mean_asr_monopoly"], match_rate),
            flush=True,
        )
    by_source = {}
    for entry in results:
        by_source.setdefault(entry["source"], set()).add(
            (entry["runner_up_class"], entry["runner_up_q"])
        )
    if any(len(values) != 1 for values in by_source.values()):
        raise ValueError("pairs that share a source recorded different runner-ups")
    payload = {"honest_rho": PHASE4_RHO, "attacker_rho": 1.0, "pairs": results}
    _write_json(os.path.join(output_dir, "verdict.json"), payload)
    return payload


def _tag(value: float) -> str:
    return ("%0.2f" % value).replace(".", "p")


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
