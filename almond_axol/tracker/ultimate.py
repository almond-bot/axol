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
import os
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..utils.paths import almond_path
from ..utils.state_files import secure_atomic_write_json, secure_open_text_read
from .base import (
    TrackerPose,
    TrackerSource,
    TrackerSourceError,
    epoch_seconds_to_perf_counter,
    zup_to_yup_pos,
    zup_to_yup_quat,
)

_logger = logging.getLogger(__name__)

ULTIMATE_DONGLE_VID = 0x0BB4
ULTIMATE_DONGLE_PID = 0x0350
ULTIMATE_WIFI_CONFIG_FILE = almond_path("tracker", "ultimate_wifi.json")

_ULTIMATE_WIFI_KEYS = frozenset({"ssid", "pass", "country", "freq"})
_ISO_ALPHA_2_COUNTRIES = frozenset(
    """
    AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI
    BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN
    CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK
    FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM
    HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN
    KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK
    ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP
    NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW
    SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF
    TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI
    VN VU WF WS YE YT ZA ZM ZW
    """.split()
)

# pyvut's numeric status values come from the tracker's pose-status enum:
# 2 = position + rotation, 3 = rotation only, 4 = frozen position.  Only 2 is
# a trustworthy 6-DoF sample.  String statuses support API-compatible wrappers.
_GOOD_STATUS_STRINGS = {"tracking", "ok"}
_MISSING = object()
_READER_STOP_TIMEOUT_S = 1.0
_PYVUT_CONSTRUCTOR_LOCK = threading.Lock()


@dataclass(frozen=True, repr=False)
class UltimateWifiConfigState:
    """One descriptor-pinned, secret-safe view of the Wi-Fi config."""

    status: Literal["valid", "missing", "invalid"]
    error: str | None
    mode: int | None = None
    # Never include credentials in the representation of readiness/debug
    # state.  The values exist only so the runtime can publish a private copy
    # for pyvut without reopening the operator-controlled source path.
    values: dict[str, object] | None = field(default=None, repr=False)


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


def ultimate_wifi_values_error(value: object) -> str | None:
    """Return a secret-safe error for pyvut's exact Wi-Fi document shape.

    Channel legality remains country-specific and must be checked by the
    operator.  The checks here reject values that cannot be valid Wi-Fi
    credentials or channel centers before they reach pyvut's HID framing.
    """
    if not isinstance(value, dict):
        return "top-level value must be a JSON object"
    keys = set(value)
    if keys != _ULTIMATE_WIFI_KEYS:
        missing = sorted(_ULTIMATE_WIFI_KEYS - keys)
        extra = sorted(keys - _ULTIMATE_WIFI_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if extra:
            details.append("unknown keys: " + ", ".join(extra))
        return "config must contain exactly ssid, pass, country, and freq" + (
            f" ({'; '.join(details)})" if details else ""
        )

    ssid = value.get("ssid")
    password = value.get("pass")
    country = value.get("country")
    frequency = value.get("freq")
    if not isinstance(ssid, str):
        return "`ssid` must be a string containing 1 to 32 UTF-8 bytes"
    try:
        ssid_bytes = ssid.encode("utf-8")
    except UnicodeEncodeError:
        return "`ssid` must contain valid UTF-8 text"
    if not 1 <= len(ssid_bytes) <= 32:
        return "`ssid` must contain 1 to 32 UTF-8 bytes"
    if any(ord(char) < 32 or ord(char) == 127 for char in ssid):
        return "`ssid` must not contain control characters"

    if not isinstance(password, str):
        return "`pass` must be an 8 to 63 character printable ASCII passphrase"
    printable_ascii = password.isascii() and all(
        32 <= ord(char) <= 126 for char in password
    )
    raw_psk = len(password) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in password
    )
    if not ((8 <= len(password) <= 63 and printable_ascii) or raw_psk):
        return (
            "`pass` must be an 8 to 63 character printable ASCII passphrase "
            "or a 64-digit hexadecimal PSK"
        )

    country_code = country.upper() if isinstance(country, str) else ""
    if country_code not in _ISO_ALPHA_2_COUNTRIES:
        return "`country` must be an ISO 3166-1 two-letter country code"

    if not isinstance(frequency, int) or isinstance(frequency, bool):
        return "`freq` must be a Wi-Fi channel center frequency in MHz"
    is_channel_center = (
        (2412 <= frequency <= 2472 and (frequency - 2412) % 5 == 0)
        or frequency == 2484
        or (5000 <= frequency <= 5895 and frequency % 5 == 0)
        or (5955 <= frequency <= 7115 and (frequency - 5955) % 5 == 0)
    )
    if not is_channel_center:
        return (
            "`freq` must be a recognized 2.4, 5, or 6 GHz Wi-Fi channel "
            "center in MHz; verify that it is permitted for `country`"
        )
    return None


def ultimate_wifi_config_state(
    path: Path = ULTIMATE_WIFI_CONFIG_FILE,
) -> UltimateWifiConfigState:
    """Read and validate the shared-map Wi-Fi file through one pinned fd.

    Every path component and the final file are opened without following
    symlinks.  File mode comes from that same descriptor, so readiness never
    validates one inode and reports permissions from another.  Returned error
    text is deliberately credential-free.
    """
    try:
        with secure_open_text_read(path) as stream:
            mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode)
            raw = stream.read()
    except FileNotFoundError:
        return UltimateWifiConfigState("missing", "file is missing")
    except OSError as exc:
        return UltimateWifiConfigState("invalid", f"file cannot be read ({exc})")
    except UnicodeError:
        return UltimateWifiConfigState("invalid", "file is not valid UTF-8")

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return UltimateWifiConfigState(
            "invalid",
            f"file is not valid JSON (line {exc.lineno}, column {exc.colno})",
            mode=mode,
        )
    error = ultimate_wifi_values_error(value)
    if error is not None:
        return UltimateWifiConfigState(
            "invalid",
            error,
            mode=mode,
            values=value if isinstance(value, dict) else None,
        )
    assert isinstance(value, dict)
    return UltimateWifiConfigState("valid", None, mode=mode, values=value)


def ultimate_wifi_config_error(
    path: Path = ULTIMATE_WIFI_CONFIG_FILE,
) -> str | None:
    """Return a redacted validation error for pyvut's shared-map Wi-Fi file."""
    state = ultimate_wifi_config_state(path)
    return None if state.status == "valid" else state.error


def _private_wifi_snapshot(
    state: UltimateWifiConfigState,
) -> tuple[tempfile.TemporaryDirectory[str], str]:
    """Publish validated values where pyvut cannot be redirected by an operator."""
    assert state.status == "valid" and state.values is not None
    # `/tmp` is explicit: a root service must not honor an operator-provided
    # TMPDIR for credential material. TemporaryDirectory creates mode 0700;
    # the file inside is additionally fixed at 0600.
    temporary = tempfile.TemporaryDirectory(prefix="axol-ultimate-wifi-", dir="/tmp")
    destination = Path(temporary.name) / "wifi_info.json"
    try:
        secure_atomic_write_json(destination, state.values, mode=0o600)
    except BaseException:
        temporary.cleanup()
        raise
    return temporary, str(destination)


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


def _pyvut_hid_devices(api: object) -> tuple[object, ...]:
    """Return the HID handles owned by the pinned pyvut tracker group."""
    try:
        group = getattr(api, "tracker_group")
        comms = getattr(group, "comms")
    except Exception:  # noqa: BLE001 - third-party properties may be dynamic
        return ()
    devices: list[object] = []
    for name in ("device_hid1", "device_hid3"):
        try:
            device = getattr(comms, name, None)
        except Exception:  # noqa: BLE001 - third-party properties may be dynamic
            continue
        if device is not None and all(device is not found for found in devices):
            devices.append(device)
    return tuple(devices)


def _close_constructor_hid_devices(
    devices: tuple[object, ...],
) -> TrackerSourceError | None:
    """Close handles captured before pyvut could publish an API object."""
    failures: list[BaseException] = []
    for device in devices:
        close = getattr(device, "close", None)
        if not callable(close):
            failures.append(RuntimeError("pyvut HID device has no close method"))
            continue
        try:
            close()
        except BaseException as exc:
            failures.append(exc)
    if not failures:
        return None
    for extra in failures[1:]:
        failures[0].add_note(
            "additional Ultimate constructor cleanup failure: "
            f"{type(extra).__name__}: {extra}"
        )
    failure = TrackerSourceError(
        "Ultimate tracker constructor cleanup failed; HID ownership is uncertain"
    )
    failure.__cause__ = failures[0]
    return failure


def _construct_pyvut_api(
    api_factory: object,
    hid_module: object,
    api_kwargs: dict[str, object],
) -> tuple[
    object | None, BaseException | None, tuple[object, ...], TrackerSourceError | None
]:
    """Construct pyvut while retaining every HID handle opened on failure.

    Pinned pyvut creates ``hid.Device`` inside a nested constructor and only
    assigns the tracker group to the API after that constructor returns. If a
    feature-report or Wi-Fi read raises in between, ordinary partial-object
    cleanup cannot reach the already-open handle. Temporarily wrapping the
    exact ``hid.Device`` callable closes that ownership gap.
    """
    if not callable(api_factory):
        return None, RuntimeError("installed pyvut API is not callable"), (), None
    original_device = getattr(hid_module, "Device")
    captured: list[object] = []
    constructor_thread = threading.get_ident()

    def tracked_device(*args: object, **kwargs: object) -> object:
        device = original_device(*args, **kwargs)
        # The module attribute is process-global during this narrow window.
        # Do not claim an unrelated HID handle opened concurrently elsewhere.
        if threading.get_ident() == constructor_thread and all(
            device is not found for found in captured
        ):
            captured.append(device)
        return device

    api: object | None = None
    error: BaseException | None = None
    with _PYVUT_CONSTRUCTOR_LOCK:
        setattr(hid_module, "Device", tracked_device)
        try:
            api = api_factory(**api_kwargs)
        except BaseException as exc:  # close native ownership on every unwind
            error = exc
        finally:
            setattr(hid_module, "Device", original_device)

    devices = tuple(captured)
    cleanup_failure = (
        _close_constructor_hid_devices(devices) if error is not None else None
    )
    return api, error, devices, cleanup_failure


def _configure_pyvut_hid_polling(api: object) -> None:
    """Make the pinned pyvut read loop observe its stop event promptly.

    Upstream calls ``Device.read()`` with no timeout, while ``stop()`` only
    clears an event and joins. The pinned ``hid`` binding defaults that read to
    blocking forever, so a quiet dongle can otherwise leave both the daemon
    reader and its open HID handle behind. Nonblocking mode is supported by
    the pinned binding and pyvut already applies a 1 ms poll interval.
    """
    devices = _pyvut_hid_devices(api)
    if not devices:
        raise RuntimeError(
            "installed pyvut does not expose its HID device; run `axol "
            "tracker.ultimate.install`"
        )
    for device in devices:
        try:
            setattr(device, "nonblocking", True)
        except Exception as exc:  # noqa: BLE001 - hid binding errors vary
            raise RuntimeError(
                "installed pyvut HID device cannot enable nonblocking reads; "
                "run `axol tracker.ultimate.install`"
            ) from exc


def _stop_api(api: object, *, uses_context_exit: bool) -> None:
    """Stop pyvut and prove its captured HID reader thread has exited."""
    # The pinned pyvut stop() drops ``api._thread`` after a bounded join even
    # when that thread is still alive. Capture it first so an apparent API
    # cleanup cannot hide a poller that still owns the dongle.
    try:
        reader = getattr(api, "_thread", None)
    except Exception:  # noqa: BLE001 - third-party properties may be dynamic
        reader = None
    hid_devices = _pyvut_hid_devices(api)

    failures: list[BaseException] = []
    try:
        if uses_context_exit:
            exit_context = getattr(api, "__exit__", None)
            if not callable(exit_context):
                raise RuntimeError("pyvut context exit is no longer callable")
            exit_context(None, None, None)
        else:
            stop = getattr(api, "stop", None)
            if not callable(stop):
                raise RuntimeError("pyvut stop is no longer callable")
            stop()
    except BaseException as exc:  # still verify/join the reader below
        failures.append(exc)

    # Unknown reader state is not proof of exit. Only an absent reader or a
    # successful is_alive() == False permits closing its HID handle.
    reader_alive = reader is not None
    if reader is not None:
        join = getattr(reader, "join", None)
        is_alive = getattr(reader, "is_alive", None)
        if callable(join) and reader is not threading.current_thread():
            try:
                join(timeout=_READER_STOP_TIMEOUT_S)
            except BaseException as exc:
                failures.append(exc)
        if callable(is_alive):
            try:
                reader_alive = bool(is_alive())
            except BaseException as exc:
                failures.append(exc)
                reader_alive = True
            else:
                if reader_alive:
                    failures.append(
                        RuntimeError(
                            "pyvut HID reader thread is still alive after stop"
                        )
                    )

    # Never close a handle underneath a reader whose exit could not be proven;
    # hidapi does not promise that close-vs-read from different threads is safe.
    # Once polling has stopped, explicitly close it: pinned pyvut itself never
    # does, and dropping its Python Device wrapper does not call hid_close().
    if not reader_alive:
        for device in hid_devices:
            close = getattr(device, "close", None)
            if not callable(close):
                failures.append(RuntimeError("pyvut HID device has no close method"))
                continue
            try:
                close()
            except BaseException as exc:
                failures.append(exc)

    if failures:
        for extra in failures[1:]:
            failures[0].add_note(
                f"additional Ultimate teardown failure: {type(extra).__name__}: {extra}"
            )
        raise TrackerSourceError(
            "Ultimate tracker teardown failed; HID ownership is uncertain"
        ) from failures[0]


def _note_start_cleanup_failure(
    error: BaseException,
    api: object,
    *,
    uses_context_exit: bool,
) -> TrackerSourceError | None:
    """Try cleanup after failed startup and return uncertain ownership state."""
    try:
        _stop_api(api, uses_context_exit=uses_context_exit)
    except BaseException as cleanup_error:
        _logger.exception("ultimate backend cleanup after failed startup failed")
        error.add_note(
            "Ultimate startup cleanup also failed; HID ownership is uncertain: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        if isinstance(cleanup_error, TrackerSourceError):
            return cleanup_error
        failure = TrackerSourceError(
            "Ultimate tracker teardown failed; HID ownership is uncertain"
        )
        failure.__cause__ = cleanup_error
        return failure
    return None


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
        self._wifi_material: tempfile.TemporaryDirectory[str] | None = None
        # A failed native teardown may have stopped or closed only part of
        # pyvut's state. Keep both the API object and the exact failure alive,
        # and never call into that uncertain state a second time.
        self._teardown_failure: TrackerSourceError | None = None
        self._orphaned_hid_devices: tuple[object, ...] = ()
        self._lifecycle_lock = threading.Lock()

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._teardown_failure is not None:
            raise TrackerSourceError(
                "Ultimate tracker cleanup is incomplete; HID ownership is uncertain"
            ) from self._teardown_failure
        if self._api is not None:
            return
        wifi_info_path = None
        wifi_state = ultimate_wifi_config_state(ULTIMATE_WIFI_CONFIG_FILE)
        if wifi_state.status == "invalid":
            raise RuntimeError(
                f"invalid Ultimate shared-map Wi-Fi config at "
                f"{ULTIMATE_WIFI_CONFIG_FILE}: {wifi_state.error}"
            )
        if wifi_state.status == "valid":
            self._wifi_material, wifi_info_path = _private_wifi_snapshot(wifi_state)
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
            self._clear_wifi_material()
            raise _dependency_error(exc) from exc
        if not callable(getattr(hid, "Device", None)):
            self._clear_wifi_material()
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
            (
                api,
                constructor_error,
                constructor_devices,
                constructor_cleanup_failure,
            ) = _construct_pyvut_api(UltimateTrackerAPI, hid, api_kwargs)
            if constructor_error is not None:
                if constructor_cleanup_failure is not None:
                    self._orphaned_hid_devices = constructor_devices
                    self._teardown_failure = constructor_cleanup_failure
                    constructor_error.add_note(
                        "Ultimate constructor cleanup failed; HID ownership is "
                        "uncertain"
                    )
                if not isinstance(constructor_error, Exception):
                    self._clear_wifi_material()
                raise constructor_error
            assert api is not None
            _configure_pyvut_hid_polling(api)
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
                # If __enter__ raises after partially acquiring the dongle,
                # its matching __exit__ is the only available cleanup API.
                uses_context_exit = True
                enter()
        except BaseException as exc:  # cleanup must also survive Ctrl+C/SystemExit
            if api is not None:
                cleanup_failure = _note_start_cleanup_failure(
                    exc, api, uses_context_exit=uses_context_exit
                )
                if cleanup_failure is not None:
                    self._api = api
                    self._api_uses_context_exit = uses_context_exit
                    self._teardown_failure = cleanup_failure
            self._clear_wifi_material()
            # A process-control exception must retain its identity after native
            # ownership has been released (or the uncertain cleanup latched).
            # Ordinary pyvut/HID failures keep the actionable runtime error used
            # by the CLI and control panel.
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, RuntimeError) and "installed pyvut" in str(exc):
                raise
            raise _dongle_open_error(exc) from exc

        self._api = api
        self._api_uses_context_exit = uses_context_exit
        _logger.info("ultimate backend: dongle opened, waiting for tracker poses")

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._teardown_failure is not None:
            raise self._teardown_failure
        if self._api is not None:
            api = self._api
            try:
                _stop_api(api, uses_context_exit=self._api_uses_context_exit)
            except TrackerSourceError as exc:
                self._teardown_failure = exc
                # The private Wi-Fi snapshot is ordinary filesystem material,
                # not native pyvut state. Removing it is safe even when HID
                # ownership cannot be proven, and avoids retaining credentials.
                try:
                    self._clear_wifi_material()
                except BaseException as cleanup_error:
                    exc.add_note(
                        "additional Ultimate Wi-Fi snapshot cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
            self._api = None
            self._api_uses_context_exit = False
            self._clear_wifi_material()
        else:
            self._clear_wifi_material()

    def _clear_wifi_material(self) -> None:
        material = self._wifi_material
        self._wifi_material = None
        if material is not None:
            material.cleanup()

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
            timestamp_ms = getattr(pose, "timestamp_ms", None)
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

        receipt_epoch = time.time()
        receipt_perf = time.perf_counter()
        try:
            timestamp_seconds = float(timestamp_ms) * 0.001
        except (OverflowError, TypeError, ValueError):
            timestamp_seconds = None
        pyvut_receipt_perf = epoch_seconds_to_perf_counter(
            timestamp_seconds,
            receipt_perf=receipt_perf,
            receipt_epoch=receipt_epoch,
        )
        sample = TrackerPose(
            pos=pos,
            quat=rot,
            t=pyvut_receipt_perf if pyvut_receipt_perf is not None else receipt_perf,
            tracking=tracking,
            # The pinned pyvut assigns timestamp_ms with time.time() while
            # parsing the HID packet; its raw tracker timestamp is not exposed
            # through TrackerPose. This is the earliest receipt, not capture.
            timestamp_is_capture=False,
        )
        with self._lock:
            self._poses[key] = sample
