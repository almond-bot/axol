"""Vive Ultimate Tracker backend via the wireless dongle (USB HID).

The Ultimate Tracker is inside-out (camera SLAM, no base stations) with
no official Linux support; the community ``pyvut`` package drives the
wireless dongle over USB HID on Linux without a headset — pairing, SLAM
host/role assignment, and pose streaming for up to five trackers sharing
one map. ``pyvut`` is an operator-installed dependency (like libsurvive)
rather than a pip requirement of this project: install it from
https://github.com/nijkah/pyvut into the same environment, along with the
``hidapi`` system libraries it needs. ``axol tracker.ultimate.install``
installs the pinned, tested revision and Linux USB permissions.

One-time provisioning caveat (upstream limitation): the SLAM **map must
be created once with VIVE Streaming Hub on Windows**; after the trackers
store it, the dongle + trackers run standalone on the Jetson.

Device keys are tracker MAC addresses (stable across sessions).

Frame conventions of the dongle's pose reports are firmware-dependent and
not officially documented, so both are configurable
(``ultimate_quat_order`` / ``ultimate_up_axis`` in
``~/.almond/tracker/config.json``) and must be verified at bring-up: hold
a tracker still and level and check the streamed pose is gravity-upright.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from ..utils.paths import almond_path
from .base import (
    TrackerPose,
    TrackerSource,
    TrackerSourceError,
    zup_to_yup_pos,
    zup_to_yup_quat,
)

_logger = logging.getLogger(__name__)

ULTIMATE_DONGLE_VID = 0x0BB4
ULTIMATE_DONGLE_PID = 0x0350
ULTIMATE_WIFI_CONFIG_FILE = almond_path("tracker", "ultimate_wifi.json")

# pyvut's numeric status values come from the tracker's pose-status enum:
# 2 = position + rotation, 3 = rotation only, 4 = frozen position.  Only 2 is
# a trustworthy 6-DoF sample.  String statuses support API-compatible wrappers.
_GOOD_STATUS_STRINGS = {"tracking", "ok"}
_MISSING = object()


def _reader_health_error(api: object) -> str | None:
    """Return a failure for the pinned pyvut reader, or ``None`` if unknown.

    The pinned ``UltimateTrackerAPI`` exposes a ``threading.Event`` at
    ``_running`` and its HID polling thread at ``_thread``. A callback-loop
    exception kills that thread without clearing the event, so pose data would
    otherwise merely become stale. Older/API-compatible pyvut variants may not
    expose these private fields; their health remains unknown rather than being
    rejected solely for using different internals.
    """
    try:
        running = getattr(api, "_running", _MISSING)
        thread = getattr(api, "_thread", _MISSING)
    except Exception:  # noqa: BLE001 - third-party properties may be dynamic
        return None
    if running is _MISSING or thread is _MISSING:
        return None
    is_set = getattr(running, "is_set", None)
    if not callable(is_set):
        return None
    try:
        reader_should_be_running = bool(is_set())
    except Exception:  # noqa: BLE001 - unknown pyvut-compatible event type
        return None
    if not reader_should_be_running:
        return None
    if thread is None:
        return "pyvut reports running but its HID reader thread is absent"
    is_alive = getattr(thread, "is_alive", None)
    if not callable(is_alive):
        return None
    try:
        alive = bool(is_alive())
    except Exception:  # noqa: BLE001 - unknown pyvut-compatible thread type
        return None
    if not alive:
        return "pyvut reports running but its HID reader thread has stopped"
    return None


def ultimate_wifi_config_error(
    path: Path = ULTIMATE_WIFI_CONFIG_FILE,
) -> str | None:
    """Return a redacted validation error for pyvut's shared-map Wi-Fi file."""
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return "file is missing"
    except OSError as exc:
        return f"file cannot be read ({exc})"
    except json.JSONDecodeError as exc:
        return f"file is not valid JSON (line {exc.lineno}, column {exc.colno})"
    if not isinstance(value, dict):
        return "top-level value must be a JSON object"

    ssid = value.get("ssid")
    password = value.get("pass")
    country = value.get("country")
    frequency = value.get("freq")
    if not isinstance(ssid, str) or not ssid:
        return "`ssid` must be a non-empty string"
    if not isinstance(password, str) or not password:
        return "`pass` must be a non-empty string"
    if not isinstance(country, str) or len(country) != 2 or not country.isalpha():
        return "`country` must be a two-letter string"
    if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency <= 0:
        return "`freq` must be a positive integer"
    return None


def ultimate_dongle_present() -> bool | None:
    """Return whether Linux USB sysfs contains the Ultimate dongle.

    ``None`` means sysfs is unavailable, so callers should avoid claiming the
    dongle is absent.  This check never opens the HID endpoint.
    """
    usb_devices = Path("/sys/bus/usb/devices")
    if not usb_devices.is_dir():
        return None
    try:
        vendor_files = tuple(usb_devices.glob("*/idVendor"))
    except OSError:
        return None
    for vendor_file in vendor_files:
        try:
            vendor = vendor_file.read_text().strip().lower()
            product = vendor_file.with_name("idProduct").read_text().strip().lower()
        except OSError:
            continue
        if vendor == f"{ULTIMATE_DONGLE_VID:04x}" and product == (
            f"{ULTIMATE_DONGLE_PID:04x}"
        ):
            return True
    return False


def _dependency_error(exc: BaseException) -> RuntimeError:
    detail = str(exc).strip()
    suffix = f" ({type(exc).__name__}: {detail})" if detail else ""
    return RuntimeError(
        "the Ultimate backend dependencies are unavailable; run "
        "`axol tracker.ultimate.install`, then retry" + suffix
    )


def _dongle_open_error(exc: BaseException) -> RuntimeError:
    detail = str(exc).strip()
    suffix = f": {detail}" if detail else ""
    if ultimate_dongle_present() is False:
        return RuntimeError(
            "VIVE Ultimate Tracker dongle 0bb4:0350 was not found; connect the "
            "HTC wireless dongle and retry"
        )
    return RuntimeError(
        "the VIVE Ultimate Tracker dongle 0bb4:0350 could not be opened"
        f"{suffix}. Check the udev permissions with "
        "`axol tracker.ultimate.check` and ensure no other process is using it"
    )


def _stop_api(api: object, *, uses_context_exit: bool) -> None:
    """Best-effort stop for both supported pyvut lifecycle variants."""
    try:
        stop = getattr(api, "stop", None)
        if not uses_context_exit and callable(stop):
            stop()
            return
        exit_context = getattr(api, "__exit__", None)
        if uses_context_exit and callable(exit_context):
            exit_context(None, None, None)
    except Exception:  # noqa: BLE001 - teardown must not mask the real failure
        _logger.exception("ultimate backend teardown failed")


class UltimateSource(TrackerSource):
    """Poses for every Ultimate Tracker paired to the connected dongle.

    Args:
        quat_order: Component order of the report quaternion (``"xyzw"``
            or ``"wxyz"``).
        up_axis: Up axis of the tracker SLAM world frame (``"z"`` converts
            through the z-up → y-up basis change, ``"y"`` passes through).
    """

    def __init__(self, quat_order: str = "wxyz", up_axis: str = "z") -> None:
        if quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"quat_order must be xyzw or wxyz, got {quat_order!r}")
        if up_axis not in ("y", "z"):
            raise ValueError(f"up_axis must be y or z, got {up_axis!r}")
        self._quat_order = quat_order
        self._up_axis = up_axis
        self._poses: dict[str, TrackerPose] = {}
        self._lock = threading.Lock()
        self._api = None
        self._api_uses_context_exit = False

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._api is not None:
            return
        wifi_info_path = None
        if ULTIMATE_WIFI_CONFIG_FILE.exists():
            config_error = ultimate_wifi_config_error(ULTIMATE_WIFI_CONFIG_FILE)
            if config_error is not None:
                raise RuntimeError(
                    f"invalid Ultimate shared-map Wi-Fi config at "
                    f"{ULTIMATE_WIFI_CONFIG_FILE}: {config_error}"
                )
            wifi_info_path = str(ULTIMATE_WIFI_CONFIG_FILE)
        else:
            _logger.warning(
                "Ultimate shared-map Wi-Fi config %s is absent; pyvut will use "
                "its packaged settings. Run `axol tracker.ultimate.check`.",
                ULTIMATE_WIFI_CONFIG_FILE,
            )
        try:
            import hid
            from pyvut import UltimateTrackerAPI
            from pyvut.tracker_core import set_tracker_core_verbose
        except (ImportError, OSError) as exc:
            raise _dependency_error(exc) from exc
        if not callable(getattr(hid, "Device", None)):
            raise RuntimeError(
                "the Ultimate backend requires the PyPI package `hid` with its "
                "`hid.Device` API; the similarly named `hidapi` package is not "
                "compatible. Run `axol tracker.ultimate.install`, then retry"
            )

        api = None
        uses_context_exit = False
        try:
            # This pinned pyvut revision has a verbose ACK message that includes
            # the shared-map Wi-Fi password.  Keep its fallback logger above
            # DEBUG and disable its direct prints before pose polling begins.
            logging.getLogger("pyvut.tracker_core").setLevel(logging.INFO)
            set_tracker_core_verbose(False)

            # The pinned pyvut revision opens HID in the constructor.  Older
            # compatible revisions may defer it until start()/__enter__().
            api_kwargs = {"mode": "DONGLE_USB"}
            if wifi_info_path is not None:
                api_kwargs["wifi_info_path"] = wifi_info_path
            api = UltimateTrackerAPI(**api_kwargs)
            # ViveTrackerGroup's constructor currently turns verbose printing
            # back on, so suppress it again immediately after construction and
            # before start() launches the ACK polling thread.
            set_tracker_core_verbose(False)
            add_callback = getattr(api, "add_pose_callback", None)
            if not callable(add_callback):
                raise RuntimeError(
                    "installed pyvut has no add_pose_callback API; run "
                    "`axol tracker.ultimate.install`"
                )

            # Register before starting whenever the API permits it so an
            # immediately available first pose is not lost.
            add_callback(self._on_pose)
            start = getattr(api, "start", None)
            stop = getattr(api, "stop", None)
            if callable(start) and callable(stop):
                start()
            else:
                enter = getattr(api, "__enter__", None)
                exit_context = getattr(api, "__exit__", None)
                if not callable(enter) or not callable(exit_context):
                    raise RuntimeError(
                        "installed pyvut has neither a start/stop lifecycle nor "
                        "a complete context-manager lifecycle; run `axol "
                        "tracker.ultimate.install`"
                    )
                enter()
                uses_context_exit = True
        except RuntimeError as exc:
            if api is not None:
                _stop_api(api, uses_context_exit=uses_context_exit)
            if "installed pyvut" in str(exc):
                raise
            raise _dongle_open_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - pyvut/HID error types vary
            if api is not None:
                _stop_api(api, uses_context_exit=uses_context_exit)
            raise _dongle_open_error(exc) from exc

        self._api = api
        self._api_uses_context_exit = uses_context_exit
        _logger.info("ultimate backend: dongle opened, waiting for tracker poses")

    def stop(self) -> None:
        if self._api is not None:
            api = self._api
            self._api = None
            _stop_api(api, uses_context_exit=self._api_uses_context_exit)
            self._api_uses_context_exit = False

    def poses(self) -> dict[str, TrackerPose]:
        api = self._api
        if api is not None:
            health_error = _reader_health_error(api)
            if health_error is not None:
                raise TrackerSourceError(
                    f"Ultimate tracker backend failed: {health_error}. Stop and "
                    "restart the Mantis operation; if it repeats, check the "
                    "dongle connection and pyvut logs"
                )
        with self._lock:
            return dict(self._poses)

    # -- Internal ---------------------------------------------------------------

    def _on_pose(self, pose: object) -> None:
        """pyvut pose callback (runs on its reader thread)."""
        try:
            key = str(pose.mac)
            pos = np.asarray(pose.position, dtype=np.float64)
            rot = np.asarray(pose.rotation, dtype=np.float64)
            status = getattr(pose, "tracking_status", None)
        except (AttributeError, OverflowError, TypeError, ValueError):
            return
        if pos.shape != (3,) or rot.shape != (4,):
            return
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
            return
        rot_norm = float(np.linalg.norm(rot))
        if not np.isfinite(rot_norm) or rot_norm <= 0.0:
            return

        if self._quat_order == "wxyz":
            rot = np.array([rot[1], rot[2], rot[3], rot[0]])
        if self._up_axis == "z":
            pos = zup_to_yup_pos(pos)
            rot = zup_to_yup_quat(rot)
        else:
            n = float(np.linalg.norm(rot))
            rot = rot / n if n > 0.0 else np.array([0.0, 0.0, 0.0, 1.0])

        if isinstance(status, str):
            tracking = status.strip().lower() in _GOOD_STATUS_STRINGS
        else:
            # bool is an int subclass, but cannot be a protocol status.
            tracking = bool(not isinstance(status, bool) and status == 2)

        sample = TrackerPose(
            pos=pos, quat=rot, t=time.perf_counter(), tracking=tracking
        )
        with self._lock:
            self._poses[key] = sample
