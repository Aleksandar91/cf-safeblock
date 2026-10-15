"""Frozen Fashion-MNIST trunk. The development head is not part of phi."""

from __future__ import annotations

import hashlib

import numpy as np

from cf_safeblock.fashion_split import assert_seed_available

PHI_SEED = 10009
PHI_HIDDEN = (256, 128)
PHI_MAX_EPOCHS = 30
PHI_PATIENCE = 5
PHI_MIN_DELTA = 1e-4
PHI_LR = 1e-3
PHI_BATCH = 32
PHI_BETA1 = 0.9
PHI_BETA2 = 0.999
PHI_ADAM_EPS = 1e-8
PHI_CANONICAL_PREFIX = b"cf-safeblock-phi-v1\n"
NORMALIZATION_RECORD = "\n".join((
    "pixel_scale=1/255",
    "input_layout=flat_784_row_major",
    "activation=relu",
    "batch_norm=absent",
    "running_statistics=absent",
)) + "\n"


class AdamState(object):
    def __init__(self, shapes):
        self.step = 0
        self.moment = [np.zeros(shape, dtype=np.float64) for shape in shapes]
        self.velocity = [np.zeros(shape, dtype=np.float64) for shape in shapes]


def train_locked_phi(x_train, y_train, x_select, y_select, verbose: bool = False) -> dict:
    """Train the pre-registered trunk. Metrics come only from the two arguments."""

    assert_seed_available(PHI_SEED)
    return fit_mlp(
        x_train,
        y_train,
        x_select,
        y_select,
        seed=PHI_SEED,
        hidden=PHI_HIDDEN,
        max_epochs=PHI_MAX_EPOCHS,
        patience=PHI_PATIENCE,
        min_delta=PHI_MIN_DELTA,
        learning_rate=PHI_LR,
        batch_size=PHI_BATCH,
        n_classes=10,
        verbose=verbose,
    )


def fit_mlp(
    x_train,
    y_train,
    x_select,
    y_select,
    seed,
    hidden,
    max_epochs,
    patience,
    min_delta,
    learning_rate,
    batch_size,
    n_classes,
    verbose=False,
) -> dict:
    train_x = _matrix(x_train)
    select_x = _matrix(x_select)
    train_y = _label_vector(y_train, train_x.shape[0], n_classes)
    select_y = _label_vector(y_select, select_x.shape[0], n_classes)
    if train_x.shape[1] != select_x.shape[1]:
        raise ValueError("train and select features have different widths")

    rng = np.random.RandomState(seed)
    widths = (train_x.shape[1],) + tuple(hidden) + (n_classes,)
    weights = []
    biases = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:]):
        scale = np.sqrt(2.0 / float(fan_in))
        weights.append(rng.normal(0.0, scale, size=(fan_out, fan_in)).astype(np.float64))
        biases.append(np.zeros(fan_out, dtype=np.float64))
    optimizer = AdamState([array.shape for array in weights + biases])

    best_loss = None
    best_epoch = None
    best_weights = None
    best_biases = None
    stale_epochs = 0
    history = []
    for epoch in range(max_epochs):
        order = rng.permutation(train_x.shape[0])
        train_loss = _epoch(train_x, train_y, order, weights, biases, optimizer, learning_rate, batch_size)
        select_loss, select_f1 = evaluate_mlp(select_x, select_y, weights, biases)
        history.append({
            "epoch": epoch,
            "train_ce": train_loss,
            "select_ce": select_loss,
            "select_macro_f1": select_f1,
        })
        if verbose:
            print(
                "epoch %s train_ce %.6f select_ce %.6f select_macro_f1 %.4f" % (
                    epoch, train_loss, select_loss, select_f1
                ),
                flush=True,
            )
        if best_loss is None or select_loss < best_loss - min_delta:
            best_loss = select_loss
            best_epoch = epoch
            best_weights = [array.copy() for array in weights]
            best_biases = [array.copy() for array in biases]
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    trunk = {
        "w1": best_weights[0],
        "b1": best_biases[0],
        "w2": best_weights[1],
        "b2": best_biases[1],
    }
    discarded_head = {"w3": best_weights[2], "b3": best_biases[2]}
    return {
        "trunk": trunk,
        "discarded_head": discarded_head,
        "selected_epoch": best_epoch,
        "history": history,
        "parameters_sha256": hashlib.sha256(canonical_trunk_bytes(trunk)).hexdigest(),
        "normalization_sha256": hashlib.sha256(NORMALIZATION_RECORD.encode("utf-8")).hexdigest(),
    }


def evaluate_mlp(features, labels, weights, biases):
    probabilities = _forward(features, weights, biases)[-1]
    predicted = np.argmax(probabilities, axis=1)
    loss = float(-np.mean(np.log(probabilities[np.arange(labels.shape[0]), labels] + 1e-12)))
    return loss, _macro_f1(labels, predicted, weights[-1].shape[0])


def featurize(features, trunk) -> np.ndarray:
    """Apply the frozen trunk. This function has no label input."""

    values = _matrix(features)
    hidden = np.maximum(values.dot(trunk["w1"].T) + trunk["b1"], 0.0)
    return np.maximum(hidden.dot(trunk["w2"].T) + trunk["b2"], 0.0)


def load_canonical_trunk(payload: bytes) -> dict:
    """Inverse of `canonical_trunk_bytes`."""

    if not payload.startswith(PHI_CANONICAL_PREFIX):
        raise ValueError("trunk bytes have the wrong prefix")
    offset = len(PHI_CANONICAL_PREFIX)
    trunk = {}
    for expected in ("w1", "b1", "w2", "b2"):
        end = payload.index(b"\n", offset)
        header = payload[offset:end].decode("ascii")
        offset = end + 1
        name, shape_text = header.split(" ")
        if name != expected:
            raise ValueError("trunk bytes are out of order")
        shape = tuple(int(axis) for axis in shape_text.split(",") if axis)
        count = 1
        for axis in shape:
            count *= axis
        nbytes = count * 8
        values = np.frombuffer(payload[offset : offset + nbytes], dtype="<f8").copy()
        offset += nbytes
        trunk[name] = values.reshape(shape)
    if offset != len(payload):
        raise ValueError("trunk bytes have trailing data")
    return trunk


def canonical_trunk_bytes(trunk) -> bytes:
    parts = [PHI_CANONICAL_PREFIX]
    for name in ("w1", "b1", "w2", "b2"):
        array = np.ascontiguousarray(trunk[name], dtype="<f8")
        shape = ",".join(str(axis) for axis in array.shape)
        parts.append(("%s %s\n" % (name, shape)).encode("ascii"))
        parts.append(array.tobytes())
    return b"".join(parts)


def _epoch(features, labels, order, weights, biases, optimizer, learning_rate, batch_size) -> float:
    total = 0.0
    seen = 0
    for start in range(0, order.shape[0], batch_size):
        batch = order[start : start + batch_size]
        loss, gradients = _gradients(features[batch], labels[batch], weights, biases)
        _adam(weights, biases, gradients, optimizer, learning_rate)
        total += loss * batch.shape[0]
        seen += batch.shape[0]
    return float(total / seen)


def _gradients(features, labels, weights, biases):
    activations, pre_activations = _forward_train(features, weights, biases)
    probabilities = activations[-1]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    loss = float(-np.mean(np.log(probabilities[np.arange(labels.shape[0]), labels] + 1e-12)))
    upstream = (probabilities - one_hot) / float(labels.shape[0])
    weight_grads = []
    bias_grads = []
    for layer in range(len(weights) - 1, -1, -1):
        weight_grads.append(upstream.T.dot(activations[layer]))
        bias_grads.append(np.sum(upstream, axis=0))
        if layer == 0:
            break
        upstream = upstream.dot(weights[layer])
        upstream *= pre_activations[layer] > 0.0
    weight_grads.reverse()
    bias_grads.reverse()
    return loss, weight_grads + bias_grads


def _forward(features, weights, biases):
    return _forward_train(features, weights, biases)[0]


def _forward_train(features, weights, biases):
    activations = [features]
    pre_activations = [None]
    current = features
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        current = current.dot(weight.T) + bias
        pre_activations.append(current)
        if layer < len(weights) - 1:
            current = np.maximum(current, 0.0)
        else:
            current = _softmax(current)
        activations.append(current)
    return activations, pre_activations


def _adam(weights, biases, gradients, optimizer, learning_rate) -> None:
    optimizer.step += 1
    parameters = weights + biases
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
        optimizer.moment[index] = PHI_BETA1 * optimizer.moment[index] + (1.0 - PHI_BETA1) * gradient
        optimizer.velocity[index] = PHI_BETA2 * optimizer.velocity[index] + (1.0 - PHI_BETA2) * np.square(gradient)
        moment = optimizer.moment[index] / (1.0 - PHI_BETA1 ** optimizer.step)
        velocity = optimizer.velocity[index] / (1.0 - PHI_BETA2 ** optimizer.step)
        parameter -= learning_rate * moment / (np.sqrt(velocity) + PHI_ADAM_EPS)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _macro_f1(labels, predicted, n_classes) -> float:
    scores = []
    for class_id in range(n_classes):
        true_positive = np.sum((predicted == class_id) & (labels == class_id))
        false_positive = np.sum((predicted == class_id) & (labels != class_id))
        false_negative = np.sum((predicted != class_id) & (labels == class_id))
        precision_den = true_positive + false_positive
        recall_den = true_positive + false_negative
        precision = float(true_positive) / float(precision_den) if precision_den else 0.0
        recall = float(true_positive) / float(recall_den) if recall_den else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _matrix(values) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("features must be a finite non-empty matrix")
    return array


def _label_vector(values, expected_count, n_classes) -> np.ndarray:
    labels = np.array(values, dtype=np.int64, copy=True)
    if labels.shape != (expected_count,) or np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels must address every feature row")
    return labels
