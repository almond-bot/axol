"""Jelly yaw-rate source from the ZED Box carrier board's BMI088 IMU.

Jelly's straight-line drift (uneven floor contact, omni-wheel
effective-radius mismatch) is unobservable from wheel torque feedback — see
``almond_axol.diagnostics.base.floor_sim`` — so holding a heading needs an
external yaw reference. The carrier board carries a Bosch BMI088 on its own
i2c bus, driven by StereoLabs' ``bmi_spsc`` module, which is rigidly bolted
to Jelly and completely independent of the cameras: reading it costs no
camera ownership, so the overhead ZED stays on the video relay's GPU-resident
GStreamer pipeline.

Ring protocol
─────────────
``/dev/spsc_bmi0`` mmaps (only at exactly 32768 bytes — the driver rejects
any other length) to a single-producer/single-consumer ring: a header of
``u32 head, u32 tail`` followed by 1024 records of ``u64 ts_ns`` plus three
``int16`` gyro and three ``int16`` accelerometer axes. Raw counts scale by
the ranges the driver reports in sysfs, which are the BMI088's own register
values (gyro range 1 = ±1000 deg/s, accel range 2 = ±12 g).

The consumer MUST publish its read position back to ``tail``: the producer
stalls and the driver stops the sampling timer once the ring fills, so a
read-only reader gets exactly 1024 samples and then silence. That also means
the timer cannot usefully be started at boot — it would stall long before
teleop opens the device — so :meth:`open` starts it and needs write access to
``timer_control`` (see the class docstring).

Mounting-orientation independence: rather than assuming which IMU axis is the
Jelly's vertical, the gravity direction is estimated online from the
accelerometer (low-passed, and only updated when the measured magnitude is
close to 1 g so launch/brake transients don't bend it). The yaw rate is the
angular-velocity component about that axis: ``rate = gyro · ĝ_up``, which is
CCW-positive seen from above — Jelly's +wz — for any rigid mounting. The
board sits at ~31° on this robot, and measured against a hand-rotated Jelly the
projection tracks heading with the correct sign to within a degree.
"""

from __future__ import annotations

import logging
import math
import mmap
import os
import struct
import threading
import time
from typing import Callable

_logger = logging.getLogger(__name__)

DEVICE = "/dev/spsc_bmi0"
SYSFS = "/sys/class/bmi_spsc/spsc_bmi0"

_MAP_BYTES = 32768
_HEADER = 8
_RECORD = 20
_SLOTS = 1024
_REC = struct.Struct("<Qhhhhhh")

# LSB per deg/s and full-scale g, indexed by the BMI088 range register value
# the driver echoes in sysfs.
_GYRO_LSB = {0: 16.384, 1: 32.768, 2: 65.536, 3: 131.072, 4: 262.144}
_ACCEL_FS_G = {0: 3.0, 1: 6.0, 2: 12.0, 3: 24.0}

# Accept accelerometer samples within this band around 1 g for the gravity
# estimate; outside it Jelly is accelerating/braking and the sample would
# bend the vertical.
_GRAVITY_BAND = (0.82, 1.18)  # g

# Low-pass factor per accepted sample for the gravity direction (tau ~0.5 s
# at the sensor's 200 Hz).
_GRAVITY_ALPHA = 0.01

# Ring drain period. Well under the ~5 s it takes 200 Hz to fill 1024 slots,
# so the producer never stalls, and short enough that a sample reaches the
# Jelly's 50 Hz command loop fresh.
_DRAIN_INTERVAL = 0.002


_UDEV_RULE_PATH = "/etc/udev/rules.d/99-bmi-spsc.rules"
_UDEV_RULE = """\
# ZED Box carrier-board BMI088 — Jelly yaw reference (almond_axol.robot.gyro).
# The driver ships /dev/spsc_bmi0 as root:imu already; only the sampling
# timer's sysfs control is root-only. The consumer has to start that timer
# itself (the ring stalls the producer once full, so it can't just be started
# at boot), so hand it to the same group.
SUBSYSTEM=="bmi_spsc", ACTION=="add", \\
  RUN+="/bin/sh -c 'chgrp imu /sys%p/timer_control && chmod 0660 /sys%p/timer_control'"
"""


def install() -> None:
    """Grant the ``imu`` group control of the BMI088 sampling timer.

    Idempotent and best-effort, and a no-op on hosts without the driver (so
    ``axol provision`` stays safe to run anywhere). Applies to the live device
    as well as installing the rule, since the board IMU is already enumerated
    by the time provisioning runs.
    """
    from ..utils.sudo import prime_sudo, run_root

    if not os.path.isdir("/sys/class/bmi_spsc"):
        _logger.info("no bmi_spsc driver; skipping board IMU setup")
        return
    control = f"{SYSFS}/timer_control"
    try:
        rule_ok = open(_UDEV_RULE_PATH).read() == _UDEV_RULE
    except OSError:
        rule_ok = False
    if rule_ok and os.access(control, os.W_OK):
        _logger.info("board IMU already provisioned")
        return
    if not prime_sudo():
        _logger.warning(
            "board IMU needs root to provision; Jelly heading hold will be "
            "unavailable. Run manually: sudo chgrp imu %s && sudo chmod 0660 %s",
            control,
            control,
        )
        return
    if not rule_ok:
        run_root(["tee", _UDEV_RULE_PATH], input_text=_UDEV_RULE)
        run_root(["udevadm", "control", "--reload-rules"])
    run_root(["chgrp", "imu", control])
    run_root(["chmod", "0660", control])
    _logger.info("board IMU ready (imu group controls the sampler)")


class BoardYawRateSource:
    """Streams the board BMI088's Jelly-frame yaw rate to a callback.

    The callback fires from a daemon thread at up to the sensor's 200 Hz; it
    must be cheap and thread-safe (``Jelly.feed_yaw_rate`` only latches a
    tuple).

    :meth:`open` starts the driver's sampling timer, which means writing
    ``timer_control`` in sysfs — root-only as the driver ships. Grant the
    ``imu`` group access once with a udev rule (see ``axol provision``) or the
    open raises :class:`PermissionError`.
    """

    def __init__(self, on_rate: Callable[[float], None]) -> None:
        self._on_rate = on_rate
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        self._gyro_lsb = _GYRO_LSB[1]
        self._accel_lsb = 32768.0 / _ACCEL_FS_G[2]

    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the sampler, map the ring, and begin streaming.

        Raises:
            FileNotFoundError: the bmi_spsc driver isn't bound (no board IMU).
            PermissionError: the ``imu`` group can't read the device or start
                the sampler.
        """
        self._read_ranges()
        self._start_sampler()
        fd = os.open(DEVICE, os.O_RDWR)
        try:
            mm = mmap.mmap(
                fd,
                _MAP_BYTES,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED,
            )
        except OSError:
            os.close(fd)
            raise
        self._fd, self._mm = fd, mm
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="board-gyro"
        )
        self._thread.start()
        _logger.info("board gyro: streaming BMI088 yaw rate from %s", DEVICE)

    def close(self) -> None:
        """Stop streaming and release the mapping (leaves the sampler running)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # ------------------------------------------------------------------

    def _read_ranges(self) -> None:
        """Adopt the ranges the driver is actually configured for."""
        try:
            gyro = int(open(f"{SYSFS}/gyro_range").read().strip())
            accel = int(open(f"{SYSFS}/accel_range").read().strip())
        except OSError as exc:
            raise FileNotFoundError(
                f"no board IMU at {SYSFS} — the bmi_spsc driver is not bound"
            ) from exc
        self._gyro_lsb = _GYRO_LSB.get(gyro, _GYRO_LSB[1])
        self._accel_lsb = 32768.0 / _ACCEL_FS_G.get(accel, 12.0)

    def _sampler_running(self) -> bool:
        try:
            return open(f"{SYSFS}/timer_status").read().strip() == "running"
        except OSError:
            return False

    def _start_sampler(self) -> None:
        if self._sampler_running():
            return
        try:
            with open(f"{SYSFS}/timer_control", "w") as f:
                f.write("1")
        except OSError as exc:
            raise PermissionError(
                f"cannot start the board IMU sampler ({SYSFS}/timer_control): "
                f"{exc}. Run `axol provision` to grant the imu group access, "
                f"or start it by hand with: echo 1 | sudo tee "
                f"{SYSFS}/timer_control"
            ) from exc

    def _drain_loop(self) -> None:
        """Consume the ring, publishing the read position so it never stalls."""
        mm = self._mm
        assert mm is not None
        gravity: list[float] | None = None
        _, tail = struct.unpack_from("<II", mm, 0)
        last_sample = time.monotonic()
        while not self._stop.is_set():
            head = struct.unpack_from("<I", mm, 0)[0]
            if head == tail:
                # The driver stops its timer on an i2c error ("use
                # timer_control to restart"), which otherwise looks exactly
                # like Jelly standing still. Distinguish and recover.
                if time.monotonic() - last_sample > 1.0:
                    last_sample = time.monotonic()
                    if not self._sampler_running():
                        _logger.warning("board gyro: sampler stopped — restarting")
                        try:
                            self._start_sampler()
                        except PermissionError as exc:
                            _logger.error("board gyro: %s", exc)
                            return
            else:
                last_sample = time.monotonic()
            while tail != head:
                ts, gx, gy, gz, ax, ay, az = _REC.unpack_from(
                    mm, _HEADER + tail * _RECORD
                )
                tail = (tail + 1) % _SLOTS
                acc = [v / self._accel_lsb for v in (ax, ay, az)]
                norm = math.sqrt(sum(a * a for a in acc))
                if _GRAVITY_BAND[0] < norm < _GRAVITY_BAND[1]:
                    unit = [a / norm for a in acc]
                    if gravity is None:
                        gravity = unit
                    else:
                        gravity = [
                            g + _GRAVITY_ALPHA * (u - g) for g, u in zip(gravity, unit)
                        ]
                if gravity is None:
                    continue
                gyr = [v / self._gyro_lsb for v in (gx, gy, gz)]  # deg/s
                gn = math.sqrt(sum(g * g for g in gravity))
                self._on_rate(
                    math.radians(sum(w * g for w, g in zip(gyr, gravity)) / gn)
                )
            struct.pack_into("<I", mm, 4, tail)
            self._stop.wait(_DRAIN_INTERVAL)
