"""Cart yaw-rate source from a ZED camera's built-in IMU.

The powered cart's straight-line drift (uneven floor contact, omni-wheel
effective-radius mismatch) is unobservable from wheel torque feedback — see
``almond_axol.diagnostics.base.floor_sim`` — so holding a heading needs an
external yaw reference. The overhead ZED X Mini is rigidly mounted to the
cart frame and carries an IMU, which :class:`ZedYawRateSource` polls through
the ZED SDK and reduces to a single cart-frame yaw rate for
:meth:`~almond_axol.robot.cart.Cart.feed_yaw_rate`.

Mounting-orientation independence: rather than assuming which IMU axis is
the cart's vertical, the gravity direction is estimated online from the
accelerometer (low-passed, and only updated when the measured magnitude is
close to 1 g so launch/brake transients don't bend it). The yaw rate is the
angular-velocity component about that axis: ``rate = gyro · ĝ_up``, which is
CCW-positive seen from above — the cart's +wz — for any rigid mounting.

Camera-ownership constraint: a ZED camera can only be opened by one process,
and during teleop the overhead camera is normally owned by the video relay's
GStreamer pipeline (``zedsrc``), whose IMU metadata is not reachable from
Python. Teleop therefore pins the overhead camera to the relay's *SDK*
backend when the cart IMU is enabled (camera spec ``{"imu": True}``), where
this class attaches to the already-open handle. A future extension of the
vendored zed-gstreamer patch could export IMU samples from ``zedsrc`` itself
and lift that trade.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

_logger = logging.getLogger(__name__)

# Accept accelerometer samples within this band around 1 g for the gravity
# estimate; outside it the cart is accelerating/braking and the sample would
# bend the vertical.
_GRAVITY_BAND = (8.0, 11.6)  # m/s^2

# Low-pass factor per accepted sample for the gravity direction (tau ~0.5 s
# at the default 100 Hz poll rate).
_GRAVITY_ALPHA = 0.02


class ZedYawRateSource:
    """Polls a ZED camera's IMU and pushes cart-frame yaw rate to a callback.

    Two ways to get a camera:

    * :meth:`attach` — share an already-open ``sl.Camera`` (the video relay's
      SDK backend, or teleop's in-process fallback). The handle's owner keeps
      grabbing frames; ``get_sensors_data`` is an independent, thread-safe
      SDK call.
    * :meth:`open` — open a camera exclusively for its sensors (cart driving
      without video). With no serial, the first local stereo ZED is used —
      per the hardware manual the overhead ZED X Mini is the only stereo
      camera on the robot.

    The callback fires from a daemon thread at roughly ``hz``; it must be
    cheap and thread-safe (``Cart.feed_yaw_rate`` only latches a tuple).
    """

    def __init__(self, on_rate: Callable[[float], None], hz: float = 100.0) -> None:
        self._on_rate = on_rate
        self._interval = 1.0 / hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._owned_zed: object | None = None

    def attach(self, zed: object) -> None:
        """Start polling an already-open ``sl.Camera`` (not owned; never closed)."""
        self._start(zed)

    def open(self, serial: int | None = None) -> None:
        """Open a stereo ZED by serial (or the first local one) for sensors only.

        Raises:
            ImportError: pyzed is not installed.
            ConnectionError: no stereo camera found or it failed to open.
        """
        import pyzed.sl as sl

        if serial is None:
            from .devices import list_zed_devices

            stereo = [d for d in list_zed_devices() if d["kind"] == "stereo"]
            if not stereo:
                raise ConnectionError("no stereo ZED camera found for IMU")
            serial = stereo[0]["serial"]

        zed = sl.Camera()
        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.async_grab_camera_recovery = True
        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(f"ZED {serial} failed to open for IMU: {err}")
        self._owned_zed = zed
        self._start(zed)
        _logger.info("ZED IMU source opened camera %s (sensors only)", serial)

    def close(self) -> None:
        """Stop the poll thread and close the camera if :meth:`open` owned it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._owned_zed is not None:
            try:
                self._owned_zed.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._owned_zed = None

    # ------------------------------------------------------------------

    def _start(self, zed: object) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, args=(zed,), daemon=True, name="zed-imu"
        )
        self._thread.start()

    def _poll_loop(self, zed: object) -> None:
        try:
            import pyzed.sl as sl
        except ImportError:
            _logger.warning("ZED IMU source: pyzed not installed")
            return

        sensors = sl.SensorsData()
        gravity: list[float] | None = None  # low-passed "up" in sensor frame
        failures = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            ok = (
                zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)  # type: ignore[attr-defined]
                == sl.ERROR_CODE.SUCCESS
            )
            if ok:
                failures = 0
                imu = sensors.get_imu_data()
                acc = list(imu.get_linear_acceleration())  # m/s^2, ~+1 g up at rest
                gyr = list(imu.get_angular_velocity())  # deg/s
                norm = math.sqrt(sum(a * a for a in acc))
                if _GRAVITY_BAND[0] < norm < _GRAVITY_BAND[1]:
                    unit = [a / norm for a in acc]
                    if gravity is None:
                        gravity = unit
                    else:
                        gravity = [
                            g + _GRAVITY_ALPHA * (u - g) for g, u in zip(gravity, unit)
                        ]
                if gravity is not None:
                    gn = math.sqrt(sum(g * g for g in gravity))
                    rate = math.radians(sum(w * g for w, g in zip(gyr, gravity)) / gn)
                    self._on_rate(rate)
            else:
                failures += 1
                if failures == 50:
                    _logger.warning(
                        "ZED IMU source: get_sensors_data keeps failing; "
                        "is this camera model IMU-equipped?"
                    )
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - t0)))
