"""One local epoch of a linear head. The representation is not an input."""

from __future__ import annotations

import numpy as np

LOCAL_LR = 1e-3
LOCAL_BATCH = 32
LOCAL_BETA1 = 0.9
LOCAL_BETA2 = 0.999
LOCAL_ADAM_EPS = 1e-8
INIT_STD = 0.01


def local_shuffle_seed(seed: int, round_index: int, client_id: int) -> int:
    return int((seed * 1000003 + (round_index + 1) * 9176 + client_id * 13) % (2 ** 32))


def initialize_head(n_classes: int, n_features: int, seed: int):
    rng = np.random.RandomState(seed)
    weight = rng.normal(0.0, INIT_STD, size=(n_classes, n_features)).astype(np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)
    return weight, bias


def class_scales(labels, n_classes: int, rho: float) -> np.ndarray:
    """FedRS scales: 1 for a label present in the round, otherwise ``rho``."""

    if not np.isfinite(rho) or rho < 0.0 or rho > 1.0:
        raise ValueError("rho must lie in [0, 1]")
    targets = np.array(labels, dtype=np.int64)
    scales = np.full(n_classes, float(rho), dtype=np.float64)
    present = np.unique(targets)
    if np.any(present < 0) or np.any(present >= n_classes):
        raise ValueError("labels fall outside the head")
    scales[present] = 1.0
    return scales


def train_linear_head(features, labels, weight, bias, seed: int, n_classes: int = 10, rho: float = 1.0):
    """Return a new head after one epoch. The broadcast arrays are not modified."""

    trained_weight = np.array(weight, dtype=np.float64, copy=True)
    trained_bias = np.array(bias, dtype=np.float64, copy=True)
    values = np.array(features, dtype=np.float64, copy=False)
    targets = np.array(labels, dtype=np.int64, copy=False)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("local features and labels do not match")
    scales = class_scales(targets, n_classes, rho)
    moment_w = np.zeros_like(trained_weight)
    moment_b = np.zeros_like(trained_bias)
    velocity_w = np.zeros_like(trained_weight)
    velocity_b = np.zeros_like(trained_bias)
    step = 0
    order = np.random.RandomState(seed).permutation(targets.shape[0])
    for start in range(0, order.shape[0], LOCAL_BATCH):
        batch = order[start : start + LOCAL_BATCH]
        grad_w, grad_b = _cross_entropy_gradients(
            values[batch], targets[batch], trained_weight, trained_bias, n_classes, scales
        )
        step += 1
        moment_w = LOCAL_BETA1 * moment_w + (1.0 - LOCAL_BETA1) * grad_w
        moment_b = LOCAL_BETA1 * moment_b + (1.0 - LOCAL_BETA1) * grad_b
        velocity_w = LOCAL_BETA2 * velocity_w + (1.0 - LOCAL_BETA2) * np.square(grad_w)
        velocity_b = LOCAL_BETA2 * velocity_b + (1.0 - LOCAL_BETA2) * np.square(grad_b)
        corrected_w = moment_w / (1.0 - LOCAL_BETA1 ** step)
        corrected_b = moment_b / (1.0 - LOCAL_BETA1 ** step)
        corrected_vw = velocity_w / (1.0 - LOCAL_BETA2 ** step)
        corrected_vb = velocity_b / (1.0 - LOCAL_BETA2 ** step)
        trained_weight -= LOCAL_LR * corrected_w / (np.sqrt(corrected_vw) + LOCAL_ADAM_EPS)
        trained_bias -= LOCAL_LR * corrected_b / (np.sqrt(corrected_vb) + LOCAL_ADAM_EPS)
    return trained_weight, trained_bias


def flipped_labels(labels, client_id: int, round_index: int, attack: bool):
    """Late-onset full flip of class 0 to class 1, only for client 0."""

    updated = np.array(labels, dtype=np.int64, copy=True)
    if attack and client_id == 0 and round_index >= 10:
        updated[updated == 0] = 1
    return updated


def _cross_entropy_gradients(features, labels, weight, bias, n_classes: int, scales=None):
    logits = features.dot(weight.T) + bias
    if scales is None:
        scaled = logits
        factors = np.ones(n_classes, dtype=np.float64)
    else:
        factors = np.array(scales, dtype=np.float64)
        scaled = logits * factors
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    upstream = factors * (probabilities - one_hot) / float(labels.shape[0])
    return upstream.T.dot(features), np.sum(upstream, axis=0)
