"""M3.6 seeded memory distortion.

Builds a *distorted copy* of a case (its fault windows and disturbances),
so a prior episode run on that copy gives memory that is partly wrong
about the real case. Everything is driven by `random.Random(seed_string)`,
which is deterministic across runs and platforms.

At noise level p, independently for each item:
  fault window:  shifted by +-1 step with prob p; dropped with prob p/2
  disturbance:   moved by +-1 step with prob p;  dropped with prob p/2
and, once per case:
  a spurious fault window is added with prob p/2
  a spurious disturbance is added with prob p/2
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

from .environment import DISTURBANCES

FAULTABLE = ("charge", "cool", "restart_service")
NOISE_LEVELS = (0.0, 0.25, 0.5, 1.0)

Faults = Tuple[Tuple[str, int, int], ...]
Disturbances = Tuple[Tuple[int, str], ...]


def distort(faults: Faults, disturbances: Disturbances, level: float, seed: str) -> Tuple[Faults, Disturbances]:
    if not 0.0 <= level <= 1.0:
        raise ValueError("noise level must be in [0, 1]")
    rng = random.Random(f"red-queen-m3.6|{seed}|{level}")
    new_faults = []
    for action, first, last in faults:
        if rng.random() < level / 2:
            continue  # dropped
        if rng.random() < level:
            delta = rng.choice((-1, 1))
            first, last = max(1, first + delta), max(1, last + delta)
        new_faults.append((action, first, last))
    if rng.random() < level / 2:
        first = rng.choice((1, 2))
        new_faults.append((rng.choice(FAULTABLE), first, first + rng.choice((0, 1, 2))))

    new_dist: Dict[int, str] = {}
    for step, name in disturbances:
        if rng.random() < level / 2:
            continue
        if rng.random() < level:
            step = max(1, step + rng.choice((-1, 1)))
        new_dist.setdefault(step, name)
    if rng.random() < level / 2:
        step = rng.choice((1, 2, 3))
        new_dist.setdefault(step, rng.choice(sorted(DISTURBANCES)))
    return tuple(new_faults), tuple(sorted(new_dist.items()))
