"""Hold the joints a tuner is *not* sweeping, under MIT impedance.

The sweep tuners (``tune.gravity``, ``tune.friction``, ``tune.position-loop``)
move one joint and need the other six to stay put -- often in a deliberately
gravity-loaded pose. They used to park those joints with a single 0xA4
position command and leave them on the motor's own firmware position loop at
whatever ``position_kp`` it shipped with. On this elbow that is 0.06, and it
cannot hold a loaded joint: a shoulder sweep that swings the forearm toward
horizontal put ~5 Nm on an elbow "held at 0" and it flopped, twice, on the
bench.

Impedance carries its gains in every frame, is fed the gravity model, and is
unaffected by whatever stored gain a tuner happens to be searching. This class
streams it at :data:`HOLD_HZ`, tracks how far each holder drifts, and ramps a
holder's target while it keeps streaming -- so a joint is never unsupported
between "told to move" and "arrived".
"""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np

from ..constants import ARM_JOINTS, Joint
from ..robot.config import AxolConfig
from ..robot.gravity import GravityCompensator

#: Stream rate for the impedance hold. 100 Hz is the production command rate
#: the holders' kp/kd were tuned at.
HOLD_HZ = 100.0


class ImpedanceHolders:
    """Streams MIT impedance + gravity feedforward to the non-test joints.

    The point of the whole command is that the firmware position servo cannot
    hold a loaded joint, so the holders must not use it. Impedance carries its
    gains in every frame and is unaffected by the stored gains being searched.
    """

    def __init__(self, motors, exclude: Joint, is_left: bool, config: AxolConfig):
        self._motors = motors
        self._exclude = exclude
        self._is_left = is_left
        self._arm = config.left if is_left else config.right
        self._gravity = GravityCompensator(config)
        self._hold: dict[Joint, float] = {}
        self._task: asyncio.Task | None = None
        self.peak_wobble: dict[Joint, float] = {}
        self._drift_sum: dict[Joint, float] = {}
        self._drift_n: dict[Joint, int] = {}

    async def start(self) -> None:
        for j, m in self._motors.items():
            if j is self._exclude:
                continue
            self._hold[j] = await m.get_position()
        self.peak_wobble = {j: 0.0 for j in self._hold}
        self._drift_sum = {j: 0.0 for j in self._hold}
        self._drift_n = {j: 0 for j in self._hold}
        self._task = asyncio.create_task(self._loop())

    def reset_wobble(self) -> tuple[Joint | None, float, float]:
        """Zero the holder drift stats; return the worst ``(joint, peak, rms)``
        in degrees since the last reset.

        Both numbers, because they answer different questions and the peak
        alone cannot tell them apart. A holder that settles once and then sits
        there produces the same peak as one oscillating the whole pass -- and
        on the right elbow shoulder_1 read 1.41, 1.43, 1.43, 1.41, 1.43 deg
        across five gains, far too repeatable to be dynamic. rms separates
        them: a static droop has rms near its peak but no variation between
        passes, while a wobble carries rms well below peak and grows when the
        joint under it misbehaves.
        """
        worst = max(self.peak_wobble, key=lambda j: self.peak_wobble[j], default=None)
        peak = self.peak_wobble.get(worst, 0.0) if worst is not None else 0.0
        n = self._drift_n.get(worst, 0) if worst is not None else 0
        rms = math.sqrt(self._drift_sum.get(worst, 0.0) / n) if n else 0.0
        self.peak_wobble = {j: 0.0 for j in self._hold}
        self._drift_sum = {j: 0.0 for j in self._hold}
        self._drift_n = {j: 0 for j in self._hold}
        return worst, peak, rms

    async def ramp_to(self, joint: Joint, target: float, speed: float) -> None:
        """Walk one holder's target to ``target`` while it keeps streaming.

        The hold loop keeps commanding throughout, so the joint is under
        impedance the whole way — unlike a one-shot position command, which
        is what leaves an arm unsupported.
        """
        if joint not in self._hold:
            return
        start = self._hold[joint]
        dist = abs(target - start)
        if dist < 1e-4:
            return
        secs = dist / max(speed, 1e-3)
        t0 = time.monotonic()
        while True:
            frac = (time.monotonic() - t0) / secs
            if frac >= 1.0:
                break
            self._hold[joint] = start + (target - start) * frac
            await asyncio.sleep(0.01)
        self._hold[joint] = target
        await asyncio.sleep(0.3)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        dt = 1.0 / HOLD_HZ
        q = np.zeros(len(ARM_JOINTS), dtype=np.float32)
        while True:
            t0 = time.monotonic()
            for i, j in enumerate(ARM_JOINTS):
                if j in self._hold:
                    q[i] = self._hold[j]
            grav = self._gravity.gravity_arm(q, is_left=self._is_left)
            sends = []
            for i, j in enumerate(ARM_JOINTS):
                if j not in self._hold:
                    continue
                jc = getattr(self._arm, j.value)
                sends.append(
                    self._motors[j].set_impedance(
                        self._hold[j], 0.0, jc.kp, jc.kd, float(grav[i])
                    )
                )
            await asyncio.gather(*sends, return_exceptions=True)
            for j in self._hold:
                try:
                    drift = abs(self._motors[j].position - self._hold[j])
                except Exception:
                    continue
                deg = math.degrees(drift)
                self.peak_wobble[j] = max(self.peak_wobble[j], deg)
                self._drift_sum[j] = self._drift_sum.get(j, 0.0) + deg * deg
                self._drift_n[j] = self._drift_n.get(j, 0) + 1
            spent = time.monotonic() - t0
            if spent < dt:
                await asyncio.sleep(dt - spent)
