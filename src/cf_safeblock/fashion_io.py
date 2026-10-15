"""Fashion-MNIST archive IO. Representation training never opens the test images."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import struct
import urllib.request

import numpy as np

from cf_safeblock.fashion_split import (
    ARCHIVE_URL,
    FINAL_TEST_FILES,
    MANIFEST_COLUMNS,
    OFFICIAL_MD5,
    PHI_INPUT_FILES,
    assert_phi_input_names,
    assign_roles,
    development_indices,
    file_md5,
)
from cf_safeblock.representation import (
    NORMALIZATION_RECORD,
    canonical_trunk_bytes,
    train_locked_phi,
)


def read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    magic = struct.unpack_from(">I", payload, 0)[0]
    if magic == 2049:
        count = struct.unpack_from(">I", payload, 4)[0]
        values = np.frombuffer(payload, dtype=np.uint8, offset=8)
        if values.shape != (count,):
            raise ValueError("label archive length does not match its header")
        return values.copy()
    if magic == 2051:
        count, rows, cols = struct.unpack_from(">III", payload, 4)
        values = np.frombuffer(payload, dtype=np.uint8, offset=16)
        if values.size != count * rows * cols:
            raise ValueError("image archive length does not match its header")
        return values.reshape(count, rows * cols).copy()
    raise ValueError("unsupported IDX magic in %s" % os.path.basename(path))


def ensure_archives(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for name, expected in OFFICIAL_MD5.items():
        path = os.path.join(raw_dir, name)
        if (not os.path.exists(path)) or file_md5(path) != expected:
            urllib.request.urlretrieve(ARCHIVE_URL + name, path)
        actual = file_md5(path)
        if actual != expected:
            raise ValueError("%s md5 is %s" % (name, actual))


def write_manifest(path: str, rows: list) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row[column] == "" else row[column] for column in MANIFEST_COLUMNS})
    return _sha256_file(path)


def read_manifest(path: str) -> list:
    with open(path, "r", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({
                "image_id": raw["image_id"],
                "official_split": raw["official_split"],
                "official_index": int(raw["official_index"]),
                "class_id": int(raw["class_id"]),
                "role": raw["role"],
                "subrole": raw["subrole"],
                "client_id": "" if raw["client_id"] == "" else int(raw["client_id"]),
                "probe_rank": "" if raw["probe_rank"] == "" else int(raw["probe_rank"]),
            })
    return rows


def scaled_development(images: np.ndarray, labels: np.ndarray, rows: list, subrole: str):
    indices = np.array(development_indices(rows, subrole), dtype=np.int64)
    return images[indices].astype(np.float64) / 255.0, labels[indices].astype(np.int64)


def prepare_fashion_mnist(root: str) -> dict:
    """Build the manifest and freeze phi. The official test images stay sealed."""

    raw_dir = os.path.join(root, "raw")
    print("checking Fashion-MNIST archives", flush=True)
    ensure_archives(raw_dir)
    print("archives verified", flush=True)
    assert_phi_input_names(PHI_INPUT_FILES)
    if set(PHI_INPUT_FILES).intersection(FINAL_TEST_FILES):
        raise ValueError("phi inputs overlap the sealed test archive")

    train_labels = read_idx(os.path.join(raw_dir, "train-labels-idx1-ubyte.gz"))
    test_labels = read_idx(os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz"))
    rows = assign_roles(train_labels, test_labels)
    manifest_path = os.path.join(root, "manifest.csv")
    manifest_sha256 = write_manifest(manifest_path, rows)

    train_images = read_idx(os.path.join(raw_dir, "train-images-idx3-ubyte.gz"))
    x_train, y_train = scaled_development(train_images, train_labels, rows, "repr_train")
    x_select, y_select = scaled_development(train_images, train_labels, rows, "repr_select")
    print("training phi on repr_train only", flush=True)
    trained = train_locked_phi(x_train, y_train, x_select, y_select, verbose=True)

    phi_dir = os.path.join(root, "phi")
    os.makedirs(phi_dir, exist_ok=True)
    trunk_path = os.path.join(phi_dir, "phi_trunk.f64bin")
    trunk_bytes = canonical_trunk_bytes(trained["trunk"])
    with open(trunk_path, "wb") as handle:
        handle.write(trunk_bytes)
    if hashlib.sha256(trunk_bytes).hexdigest() != trained["parameters_sha256"]:
        raise RuntimeError("trunk hash changed while saving")
    normalization_path = os.path.join(phi_dir, "normalization.txt")
    with open(normalization_path, "w", newline="") as handle:
        handle.write(NORMALIZATION_RECORD)
    np.savez(
        os.path.join(phi_dir, "discarded_development_head.npz"),
        w3=trained["discarded_head"]["w3"],
        b3=trained["discarded_head"]["b3"],
    )
    history_path = os.path.join(phi_dir, "repr_select_history.csv")
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_ce", "select_ce", "select_macro_f1"),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in trained["history"]:
            writer.writerow(record)
    selected = trained["history"][trained["selected_epoch"]]
    identity = {
        "parameters_sha256": trained["parameters_sha256"],
        "normalization_sha256": trained["normalization_sha256"],
        "manifest_sha256": manifest_sha256,
        "selected_epoch": trained["selected_epoch"],
        "repr_select_ce": selected["select_ce"],
        "repr_select_macro_f1": selected["select_macro_f1"],
        "feature_dim": 128,
        "pilot_started": False,
    }
    identity_path = os.path.join(phi_dir, "phi_identity.json")
    with open(identity_path, "w") as handle:
        json.dump(identity, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return identity


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
