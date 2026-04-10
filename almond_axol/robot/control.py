"""Motor control utilities: friction model, velocity differentiator, gravity feedforward."""

from __future__ import annotations

import math
import time

# Cutoff frequency for the velocity differentiator low-pass filter (Hz)
CUTOFF_FREQ = 90.0


def compute_gravity(q: float, ga: float, gb: float) -> float:
    """Single-joint gravity model: τ = ga·cos(q) + gb·sin(q)"""
    return ga * math.cos(q) + gb * math.sin(q)


def compute_friction(
    velocity: float, Fc: float, k: float, Fv: float, Fo: float
) -> float:
    """Tanh friction model: τ = Fc * tanh(0.1 * k * v) + Fv * v + Fo"""
    return Fc * math.tanh(0.1 * k * velocity) + Fv * velocity + Fo


def compute_feedforward(
    q: float,
    velocity: float,
    ga: float,
    gb: float,
    Fc: float,
    k: float,
    Fv: float,
    Fo: float,
) -> float:
    """Full feedforward torque: gravity + friction."""
    return compute_gravity(q, ga, gb) + compute_friction(velocity, Fc, k, Fv, Fo)


class Differentiator:
    """
    First-order low-pass differentiator, matching C++ Differentiator::Differentiate.

    For each channel:
        a = 1 / (1 + Ts * CUTOFF_FREQ)
        b = a * CUTOFF_FREQ
        vel[i] = vel_prev[i] * a + b * (pos[i] - pos_prev[i])

    Args:
        n:        Number of channels to differentiate.
        fixed_dt: If provided, use this fixed time step (seconds) instead of
                  measuring wall-clock time between calls.  Pass
                  ``1.0 / loop_hz`` when the caller runs at a known fixed rate
                  to avoid velocity noise from asyncio timing jitter.
    """

    def __init__(self, n: int, fixed_dt: float | None = None) -> None:
        self._n = n
        self._fixed_dt = fixed_dt
        self._vel_prev = [0.0] * n
        self._pos_prev: list[float | None] = [None] * n
        self._last_time: float | None = None

    def differentiate(self, positions: list[float]) -> list[float]:
        now = time.perf_counter()

        if self._last_time is None or any(p is None for p in self._pos_prev):
            self._last_time = now
            self._pos_prev = list(positions)
            return [0.0] * self._n

        if self._fixed_dt is not None:
            Ts = self._fixed_dt
        else:
            Ts = now - self._last_time
        self._last_time = now

        if Ts <= 0:
            return list(self._vel_prev)

        a = 1.0 / (1.0 + Ts * CUTOFF_FREQ)
        b = a * CUTOFF_FREQ

        velocities: list[float] = []
        for i in range(self._n):
            vel = self._vel_prev[i] * a + b * (positions[i] - self._pos_prev[i])  # type: ignore[operator]
            self._vel_prev[i] = vel
            self._pos_prev[i] = positions[i]
            velocities.append(vel)

        return velocities
