"""M5 noisy sensors (partial observability).

Until M4.2 every agent observed the TRUE state. In M5 an agent can be made
to see the world only through a NoisySensor: battery and temperature are
reported with bounded integer noise in [-noise, +noise] (then clamped to
0..100). The service status is reported exactly.

The noise is deterministic: it depends only on (seed, step), so every run
is reproducible and repeated reads at the same step give the same value.

The environment, its event log and the GoalVerifier still use the true
state. Agents that use a sensor make their decisions from readings only;
success is still decided by the verifier from the truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

from .goal import RecoveryGoal
from .state import MAX_LEVEL, MIN_LEVEL, SystemState, clamp_level

MAX_NOISE = 25
NOISE_LEVELS: Tuple[int, ...] = (0, 5, 10, 15)


def _check_noise(noise: int) -> int:
    if isinstance(noise, bool) or not isinstance(noise, int) or not 0 <= noise <= MAX_NOISE:
        raise ValueError(f"noise must be an int in [0, {MAX_NOISE}]")
    return noise


class NoisySensor:
    """Bounded, deterministic, seeded measurement noise."""

    def __init__(self, noise: int, seed: int = 0, bias_battery: int = 0, bias_temperature: int = 0) -> None:
        self.noise = _check_noise(noise)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an int")
        for bias in (bias_battery, bias_temperature):  # M5.1: constant sensor offset (a sensor fault)
            if isinstance(bias, bool) or not isinstance(bias, int) or abs(bias) > MAX_NOISE:
                raise ValueError(f"bias must be an int in [-{MAX_NOISE}, {MAX_NOISE}]")
        self.seed = seed
        self.bias_battery = bias_battery
        self.bias_temperature = bias_temperature

    def read(self, truth: SystemState, step: int) -> SystemState:
        if self.noise == 0 and self.bias_battery == 0 and self.bias_temperature == 0:
            return truth
        if self.noise == 0:
            eb = et = 0
        else:
            rng = random.Random(f"red-queen-sensor:{self.seed}:{step}")
            eb, et = rng.randint(-self.noise, self.noise), rng.randint(-self.noise, self.noise)
        return SystemState(
            clamp_level(truth.battery + self.bias_battery + eb),
            clamp_level(truth.temperature + self.bias_temperature + et),
            truth.service_up,
        )


class SensedEnvironment:
    """What a sensor-limited agent is given instead of the environment.

    observe() returns a sensor reading; execute() passes through to the real
    environment. `events` is the real log and is used for reporting only:
    no agent decision reads it. The true state is NOT exposed here.
    """

    def __init__(self, env, sensor: NoisySensor) -> None:
        self._env = env
        self._sensor = sensor

    @property
    def step(self) -> int:
        return self._env.step

    @property
    def events(self):
        return self._env.events

    def observe(self) -> SystemState:
        return self._sensor.read(self._env.observe(), self._env.step)

    def execute(self, action_name: str) -> SystemState:
        self._env.execute(action_name)
        return self.observe()


@dataclass(frozen=True)
class Box:
    """Belief: battery and temperature intervals (inclusive) + service status."""

    b_lo: int
    b_hi: int
    t_lo: int
    t_hi: int
    service_up: bool

    def __post_init__(self) -> None:
        if not (MIN_LEVEL <= self.b_lo <= self.b_hi <= MAX_LEVEL
                and MIN_LEVEL <= self.t_lo <= self.t_hi <= MAX_LEVEL):
            raise ValueError(f"invalid box {self}")

    def worst(self) -> SystemState:
        """Worst corner for the recovery goal: least battery, most heat."""
        return SystemState(self.b_lo, self.t_hi, self.service_up)

    def contains(self, s: SystemState) -> bool:
        return (self.b_lo <= s.battery <= self.b_hi and self.t_lo <= s.temperature <= self.t_hi
                and s.service_up == self.service_up)

    def satisfies(self, goal: RecoveryGoal) -> bool:
        """True only if EVERY state in the box meets the goal."""
        return goal.is_satisfied(self.worst())

    def width(self) -> int:
        return (self.b_hi - self.b_lo) + (self.t_hi - self.t_lo)

    def to_dict(self):
        return {"battery": [self.b_lo, self.b_hi], "temperature": [self.t_lo, self.t_hi],
                "service_up": self.service_up}


def observation_box(reading: SystemState, noise: int) -> Box:
    """All true states consistent with one reading. Always contains the truth."""
    n = _check_noise(noise)
    return Box(clamp_level(reading.battery - n), clamp_level(reading.battery + n),
               clamp_level(reading.temperature - n), clamp_level(reading.temperature + n),
               reading.service_up)


class StuckSensor:
    """M5.3 test fault: a sensor that works normally until `fail_from_step`,
    then reports fixed battery/temperature values (a dead or stuck sensor).
    The service status is still read exactly."""

    def __init__(self, base: NoisySensor, fail_from_step: int, battery: int = 100, temperature: int = 0) -> None:
        if isinstance(fail_from_step, bool) or not isinstance(fail_from_step, int) or fail_from_step < 0:
            raise ValueError("fail_from_step must be an int >= 0")
        self.base = base
        self.fail_from_step = fail_from_step
        self.stuck = (clamp_level(battery), clamp_level(temperature))

    def read(self, truth: SystemState, step: int) -> SystemState:
        if step >= self.fail_from_step:
            return SystemState(self.stuck[0], self.stuck[1], truth.service_up)
        return self.base.read(truth, step)
