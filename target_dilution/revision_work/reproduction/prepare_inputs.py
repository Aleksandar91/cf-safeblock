#!/usr/bin/env python
"""Validate the frozen Fashion inputs and restore only missing release assets.

Existing archives and checkpoints are never replaced. Missing official archives
are downloaded only when --download-missing is explicitly supplied. The frozen
trunk is copied from the bundled artifact only when the working copy is absent.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOCK_PATH = os.path.join(ROOT, "target_dilution", "revision_work", "fashion_test_v1", "input_lock.json")
DATA_ROOT = os.path.join(ROOT, "data", "fashion-mnist")
ASSET_PATH = os.path.join(
    ROOT, "target_dilution", "revision_work", "reproduction", "assets", "fashion_phi_trunk.bin"
)
ARCHIVE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
OFFICIAL_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def _digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check(path, algorithm, expected, description):
    if not os.path.isfile(path):
        raise IOError("missing %s: %s" % (description, path))
    actual = _digest(path, algorithm)
    if actual != expected:
        raise ValueError("%s checksum mismatch: %s" % (description, path))


def _download_missing_archive(path, name, expected_md5, expected_sha256):
    temporary = path + ".revision-download"
    if os.path.exists(temporary):
        raise IOError("temporary download already exists; inspect it before continuing: " + temporary)
    urllib.request.urlretrieve(ARCHIVE_URL + name, temporary)
    try:
        _check(temporary, "md5", expected_md5, name)
        _check(temporary, "sha256", expected_sha256, name)
        os.rename(temporary, path)
    except Exception:
        # Leave the temporary file for inspection rather than deleting evidence.
        raise


def prepare(data_root, download_missing):
    with open(LOCK_PATH, "r") as handle:
        lock = json.load(handle)

    for relative_path, expected in lock["protocol_sha256"].items():
        path = os.path.join(ROOT, "target_dilution", "revision_work", relative_path)
        _check(path, "sha256", expected, relative_path)
    for relative_path, expected in lock["runtime_source_sha256"].items():
        _check(os.path.join(ROOT, relative_path), "sha256", expected, relative_path)
    _check(
        os.path.join(ROOT, "requirements.txt"),
        "sha256",
        lock["requirements_sha256"],
        "original requirements file",
    )

    manifest = os.path.join(data_root, "manifest.csv")
    _check(manifest, "sha256", lock["manifest_sha256"], "Fashion role manifest")

    phi_dir = os.path.join(data_root, "phi")
    os.makedirs(phi_dir, exist_ok=True)
    trunk = os.path.join(phi_dir, "phi_trunk.f64bin")
    if os.path.exists(trunk):
        _check(trunk, "sha256", lock["trunk_sha256"], "working frozen trunk")
    else:
        _check(ASSET_PATH, "sha256", lock["trunk_sha256"], "bundled frozen trunk")
        shutil.copyfile(ASSET_PATH, trunk)
        _check(trunk, "sha256", lock["trunk_sha256"], "restored frozen trunk")

    _check(
        os.path.join(phi_dir, "phi_identity.json"),
        "sha256",
        lock["trunk_identity_sha256"],
        "trunk identity record",
    )
    _check(
        os.path.join(phi_dir, "normalization.txt"),
        "sha256",
        lock["normalization_sha256"],
        "normalization record",
    )

    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for name, expected_md5 in OFFICIAL_MD5.items():
        path = os.path.join(raw_dir, name)
        expected_sha256 = lock["official_archives"][name]["sha256"]
        if not os.path.isfile(path):
            if not download_missing:
                raise IOError(
                    "missing official archive: %s; rerun with --download-missing to fetch it" % path
                )
            _download_missing_archive(path, name, expected_md5, expected_sha256)
        _check(path, "md5", expected_md5, name)
        _check(path, "sha256", expected_sha256, name)

    print("Fashion inputs validated; existing files were preserved.")
    print("Data root: " + os.path.abspath(data_root))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="download only absent official archives; never replace existing files",
    )
    args = parser.parse_args()
    prepare(os.path.abspath(args.data_root), args.download_missing)


if __name__ == "__main__":
    main()
