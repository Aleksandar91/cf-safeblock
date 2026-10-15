"""Client updates and dataset-role contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROLES = (
    "client_train",
    "development",
    "server_probe",
    "attacker_surrogate",
    "final_test",
)


class RoleViolation(ValueError):
    """Raised when dataset roles overlap or are incomplete."""


@dataclass(frozen=True)
class ClientDelta:
    """Local head minus the broadcast global head, before server clipping."""

    weight: np.ndarray
    bias: np.ndarray
    example_count: int

    def __post_init__(self) -> None:
        weight = np.array(self.weight, dtype=np.float64, copy=True)
        bias = np.array(self.bias, dtype=np.float64, copy=True)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("client delta must match a linear head")
        if self.example_count <= 0:
            raise ValueError("example_count must be positive")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError("client delta must be finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)


def fedavg_alphas(clients: list) -> np.ndarray:
    counts = np.array([client.example_count for client in clients], dtype=np.float64)
    if clients == []:
        raise ValueError("at least one client is required")
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("total example count must be positive")
    return counts / total


def validate_role_ids(role_ids: dict) -> dict:
    """Require the five disjoint dataset roles used by this study."""

    missing = [role for role in ROLES if role not in role_ids]
    unknown = [role for role in role_ids if role not in ROLES]
    if missing or unknown:
        raise RoleViolation("roles must be exactly %s" % (ROLES,))
    sets = {}
    for role in ROLES:
        values = list(role_ids[role])
        if len(values) != len(set(values)):
            raise RoleViolation("duplicate ids in %s" % role)
        sets[role] = set(values)
    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1 :]:
            overlap = sets[left].intersection(sets[right])
            if overlap:
                raise RoleViolation("%s overlaps %s" % (left, right))
    return sets


def assert_training_ids_exclude_final_test(training_ids, final_test_ids) -> None:
    overlap = set(training_ids).intersection(final_test_ids)
    if overlap:
        raise RoleViolation("final test ids leaked into training inputs")
