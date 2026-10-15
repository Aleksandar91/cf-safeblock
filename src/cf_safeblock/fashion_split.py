"""Locked Fashion-MNIST role assignment for the Phase 2 manifest."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from cf_safeblock.updates import ROLES, validate_role_ids

MANIFEST_SEED = 10007
N_CLASSES = 10
TRAIN_PER_CLASS = 6000
TEST_PER_CLASS = 1000
PROBE_POOL = 50
PRIMARY_PROBE = 20
PROBE_PREFIXES = (5, 10, 20, 50)
SURROGATE_PER_CLASS = 200
REPR_TRAIN_PER_CLASS = 800
REPR_SELECT_PER_CLASS = 200
DEVELOPMENT_PER_CLASS = REPR_TRAIN_PER_CLASS + REPR_SELECT_PER_CLASS
N_CLIENTS = 10
ATTACKER_CLIENT = 0
SOURCE_CLASS = 0

PILOT_SEEDS = (1101, 1207, 1309)
SEALED_SEEDS = (2101, 2207, 2309, 2401, 2503, 2609, 2701, 2803)
PV19_SEEDS = (101, 211, 307, 401, 503)

OFFICIAL_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
PHI_INPUT_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
)
FINAL_TEST_FILES = (
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
ARCHIVE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

MANIFEST_COLUMNS = (
    "image_id",
    "official_split",
    "official_index",
    "class_id",
    "role",
    "subrole",
    "client_id",
    "probe_rank",
)


def assign_roles(train_labels, test_labels) -> list:
    """Assign every official image once, using the locked slice order."""

    assert_seed_available(MANIFEST_SEED)
    train = _labels(train_labels, "train")
    test = _labels(test_labels, "test")
    _assert_balanced(train, TRAIN_PER_CLASS, "train")
    _assert_balanced(test, TEST_PER_CLASS, "test")

    rng = np.random.RandomState(MANIFEST_SEED)
    rows = []
    client_blocks = []
    for class_id in range(N_CLASSES):
        indices = np.where(train == class_id)[0]
        rng.shuffle(indices)
        slices = _class_slices(indices)
        rows.extend(_probe_rows(class_id, slices["probe"]))
        rows.extend(_plain_rows(class_id, slices["surrogate"], "attacker_surrogate", "surrogate"))
        rows.extend(_plain_rows(class_id, slices["repr_train"], "development", "repr_train"))
        rows.extend(_plain_rows(class_id, slices["repr_select"], "development", "repr_select"))
        client_blocks.append((class_id, slices["client_train"]))
    rows.extend(_client_rows(client_blocks))
    rows.extend(_sealed_test_rows(test))
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list) -> dict:
    """Check the five roles, the sole owner, and the nested probe prefixes."""

    if len(rows) != len({row["image_id"] for row in rows}):
        raise ValueError("image identifiers must be unique")
    role_ids = {role: [] for role in ROLES}
    for row in rows:
        if row["role"] not in ROLES:
            raise ValueError("unknown role on %s" % row["image_id"])
        if not str(row["image_id"]).startswith("fmnist-"):
            raise ValueError("manifest contains a non-Fashion-MNIST identifier")
        role_ids[row["role"]].append(row["image_id"])
    validated = validate_role_ids(role_ids)
    _assert_ownership(rows)
    _assert_probe_prefixes(rows)
    _assert_development_subroles(rows)
    return validated


def development_indices(rows: list, subrole: str) -> list:
    """Indices of one development subrole. Any other role raises."""

    if subrole not in ("repr_train", "repr_select"):
        raise ValueError("phi may use only repr_train or repr_select")
    indices = []
    for row in rows:
        if row["subrole"] != subrole:
            continue
        if row["role"] != "development" or row["official_split"] != "train":
            raise ValueError("phi input included %s" % row["image_id"])
        indices.append(int(row["official_index"]))
    expected = N_CLASSES * (REPR_TRAIN_PER_CLASS if subrole == "repr_train" else REPR_SELECT_PER_CLASS)
    if len(indices) != expected or len(indices) != len(set(indices)):
        raise ValueError("development subrole %s is incomplete" % subrole)
    return indices


def assert_phi_input_names(filenames) -> None:
    """Representation training may open only the official train archive."""

    allowed = set(PHI_INPUT_FILES)
    blocked = set(FINAL_TEST_FILES)
    for name in filenames:
        base = os.path.basename(str(name))
        if base in blocked or base not in allowed:
            raise ValueError("phi training cannot open %s" % base)


def file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _class_slices(indices: np.ndarray) -> dict:
    probe_end = PROBE_POOL
    surrogate_end = probe_end + SURROGATE_PER_CLASS
    repr_train_end = surrogate_end + REPR_TRAIN_PER_CLASS
    repr_select_end = repr_train_end + REPR_SELECT_PER_CLASS
    if repr_select_end >= len(indices):
        raise ValueError("class slice exceeds the official train class")
    return {
        "probe": indices[:probe_end],
        "surrogate": indices[probe_end:surrogate_end],
        "repr_train": indices[surrogate_end:repr_train_end],
        "repr_select": indices[repr_train_end:repr_select_end],
        "client_train": indices[repr_select_end:],
    }


def _probe_rows(class_id: int, indices: np.ndarray) -> list:
    rows = []
    for rank, official_index in enumerate(indices.tolist()):
        subrole = "probe_primary" if rank < PRIMARY_PROBE else "probe_reserve"
        rows.append(_row(official_index, "train", class_id, "server_probe", subrole, "", rank))
    return rows


def _plain_rows(class_id: int, indices: np.ndarray, role: str, subrole: str) -> list:
    return [
        _row(int(official_index), "train", class_id, role, subrole, "", "")
        for official_index in indices.tolist()
    ]


def _client_rows(client_blocks: list) -> list:
    rows = []
    for class_id, indices in client_blocks:
        values = indices.tolist()
        if class_id == SOURCE_CLASS:
            assignment = [(ATTACKER_CLIENT, values)]
        else:
            if len(values) % N_CLIENTS != 0:
                raise ValueError("client_train class %s is not divisible by %s" % (class_id, N_CLIENTS))
            width = len(values) // N_CLIENTS
            assignment = [
                (client_id, values[client_id * width : (client_id + 1) * width])
                for client_id in range(N_CLIENTS)
            ]
        for client_id, owned in assignment:
            for official_index in owned:
                rows.append(
                    _row(official_index, "train", class_id, "client_train", "client_train", client_id, "")
                )
    return rows


def _sealed_test_rows(test_labels: np.ndarray) -> list:
    rows = []
    for official_index, class_id in enumerate(test_labels.tolist()):
        rows.append(_row(official_index, "test", int(class_id), "final_test", "sealed", "", ""))
    return rows


def _row(official_index, official_split, class_id, role, subrole, client_id, probe_rank) -> dict:
    prefix = "fmnist-train" if official_split == "train" else "fmnist-test"
    return {
        "image_id": "%s-%05d" % (prefix, int(official_index)),
        "official_split": official_split,
        "official_index": int(official_index),
        "class_id": int(class_id),
        "role": role,
        "subrole": subrole,
        "client_id": client_id,
        "probe_rank": probe_rank,
    }


def _labels(values, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError("%s labels must be one-dimensional" % name)
    return labels


def _assert_balanced(labels: np.ndarray, per_class: int, name: str) -> None:
    for class_id in range(N_CLASSES):
        count = int(np.sum(labels == class_id))
        if count != per_class:
            raise ValueError("%s class %s has %s images, expected %s" % (name, class_id, count, per_class))


def _assert_ownership(rows: list) -> None:
    owned = [
        row for row in rows
        if row["role"] == "client_train" and row["class_id"] == SOURCE_CLASS
    ]
    if not owned or any(row["client_id"] != ATTACKER_CLIENT for row in owned):
        raise ValueError("class 0 training ownership is not exclusive to client 0")
    for class_id in range(1, N_CLASSES):
        counts = [0] * N_CLIENTS
        for row in rows:
            if row["role"] == "client_train" and row["class_id"] == class_id:
                counts[int(row["client_id"])] += 1
        if any(count == 0 for count in counts):
            raise ValueError("class %s missed a client" % class_id)


def _assert_probe_prefixes(rows: list) -> None:
    for class_id in range(N_CLASSES):
        ranks = sorted(
            int(row["probe_rank"])
            for row in rows
            if row["role"] == "server_probe" and row["class_id"] == class_id
        )
        if ranks != list(range(PROBE_POOL)):
            raise ValueError("probe ranks for class %s are not 0..%s" % (class_id, PROBE_POOL - 1))
        for prefix in PROBE_PREFIXES:
            selected = [
                row["image_id"] for row in rows
                if row["role"] == "server_probe"
                and row["class_id"] == class_id
                and int(row["probe_rank"]) < prefix
            ]
            if len(selected) != prefix:
                raise ValueError("nested probe prefix %s is incomplete" % prefix)


def _assert_development_subroles(rows: list) -> None:
    counts = {"repr_train": 0, "repr_select": 0}
    for row in rows:
        if row["role"] != "development":
            continue
        if row["subrole"] not in counts or row["official_split"] != "train":
            raise ValueError("invalid development row %s" % row["image_id"])
        counts[row["subrole"]] += 1
    if counts["repr_train"] != N_CLASSES * REPR_TRAIN_PER_CLASS:
        raise ValueError("repr_train count is wrong")
    if counts["repr_select"] != N_CLASSES * REPR_SELECT_PER_CLASS:
        raise ValueError("repr_select count is wrong")


def assert_seed_available(seed: int) -> None:
    reserved = set(PILOT_SEEDS).union(SEALED_SEEDS).union(PV19_SEEDS)
    if seed in reserved:
        raise ValueError("manifest seed collides with a reserved seed")
