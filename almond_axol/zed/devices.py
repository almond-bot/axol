"""Enumeration of locally connected ZED cameras.

Mono ZED-X One cameras live in ``sl.CameraOne``'s device list and stereo
ZED X cameras in ``sl.Camera``'s, so both are queried. The ZED X daemon only
enumerates GMSL cameras when it starts — a camera plugged in after boot stays
invisible until the daemon restarts (see :func:`..daemon.restart_zed_daemon`).

The ZED SDK keeps a *process-global* RPC connection to ``zed_x_daemon`` with a
background receive thread. That connection can wedge for the entire life of a
long-lived process — when the daemon restarts under it, or when the SDK was
first touched before the daemon was ready — after which every SDK call in that
process fails with ``InvalidState`` ("Receive thread is not running cannot
send") and ``get_device_list`` returns nothing. From Python that is
indistinguishable from "no cameras": no exception is raised, the list is just
empty, and a stereo camera silently downgrades to the mono pipeline. The
connection cannot be revived in-process; only a *fresh* process gets a fresh
client (which is why restarting the whole ``axol`` service clears it).

So :func:`list_zed_devices` runs the enumeration in a short-lived ``spawn``
subprocess, which always has a fresh RPC client and is therefore immune to a
wedged client in the host (serve / teleop) process. :func:`list_zed_devices_inproc`
is the in-process primitive that subprocess runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import multiprocessing.connection

_logger = logging.getLogger(__name__)

# A fresh enumeration subprocess imports pyzed and round-trips to the daemon;
# a few seconds is plenty. The cap also bounds a genuinely hung daemon so the
# host process never blocks forever waiting on detection.
_ENUMERATE_TIMEOUT_S = 15.0


class ZedDevice(TypedDict):
    """One detected ZED camera."""

    serial: int
    model: str
    kind: str  # "mono" (ZED-X One) | "stereo" (ZED X)


def list_zed_devices_inproc() -> list[ZedDevice]:
    """Enumerate ZED cameras using *this* process's SDK client.

    Prefer :func:`list_zed_devices`, which runs this in a fresh subprocess so a
    wedged RPC client in a long-lived host process can't make every camera look
    absent. This primitive is what that subprocess worker runs (and is safe to
    call directly from an already short-lived process).

    Raises:
        ImportError: If pyzed is not installed (run ``axol zed.install``).
    """
    import pyzed.sl as sl

    devices: dict[int, ZedDevice] = {}
    for d in sl.CameraOne.get_device_list():
        serial = int(d.serial_number)
        devices[serial] = {
            "serial": serial,
            "model": str(d.camera_model),
            "kind": "mono",
        }
    for d in sl.Camera.get_device_list():
        serial = int(d.serial_number)
        # A camera present in both lists is a stereo-capable ZED X.
        devices[serial] = {
            "serial": serial,
            "model": str(d.camera_model),
            "kind": "stereo",
        }
    out = sorted(devices.values(), key=lambda d: d["serial"])
    _logger.debug(
        "Detected %d ZED camera(s): %s",
        len(out),
        ", ".join(str(d["serial"]) for d in out) or "<none>",
    )
    return out


def _enumerate_worker(conn: "multiprocessing.connection.Connection") -> None:
    """Subprocess entry: enumerate with a fresh SDK client, send the result back.

    Sends ``("ok", devices)`` on success or ``("err", error_type, message)`` so
    the parent can re-raise the same exception class — notably ``ImportError``
    for a missing pyzed, which the UI surfaces specially.
    """
    try:
        devices = list_zed_devices_inproc()
        conn.send(("ok", devices))
    except Exception as exc:  # noqa: BLE001 - report every failure to the parent
        conn.send(("err", type(exc).__name__, str(exc)))
    finally:
        conn.close()


def list_zed_devices(timeout_s: float = _ENUMERATE_TIMEOUT_S) -> list[ZedDevice]:
    """Return every locally connected ZED camera (mono + stereo).

    Runs the enumeration in a short-lived ``spawn`` subprocess so it always uses
    a *fresh* SDK RPC client (see the module docstring): the SDK's connection to
    ``zed_x_daemon`` is process-global and can wedge for the life of a long-lived
    process, after which in-process enumeration silently returns nothing. A fresh
    subprocess sidesteps that entirely, so a single hiccup no longer pins every
    camera to "absent" (and every stereo camera to the mono pipeline) until the
    host process is restarted.

    Raises:
        ImportError: If pyzed is not installed in the subprocess.
        TimeoutError: If the subprocess doesn't answer within ``timeout_s``
            (e.g. a genuinely hung daemon).
        RuntimeError: If enumeration failed in the subprocess for any other
            reason (the original error type + message are preserved), or the
            subprocess crashed without reporting a result.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_enumerate_worker,
        args=(child_conn,),
        daemon=True,
        name="zed-enumerate",
    )
    proc.start()
    # Only the child writes; closing our copy means poll() sees EOF promptly if
    # the child dies (e.g. a native crash in pyzed) instead of blocking.
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_s):
            raise TimeoutError(
                f"ZED enumeration subprocess did not respond within "
                f"{timeout_s:.0f}s (is zed_x_daemon hung?)."
            )
        try:
            status, *payload = parent_conn.recv()
        except EOFError:
            raise RuntimeError(
                "ZED enumeration subprocess exited without a result "
                "(the SDK likely crashed while talking to zed_x_daemon)."
            ) from None
    finally:
        parent_conn.close()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)

    if status == "ok":
        return payload[0]
    error_type, message = payload
    if error_type == "ImportError":
        raise ImportError(message)
    raise RuntimeError(f"{error_type}: {message}")


def stereo_serials() -> set[int]:
    """Serials of locally connected stereo (ZED X) cameras.

    Best-effort companion to :func:`list_zed_devices` used to auto-detect
    whether a configured camera is stereo from its serial, so operators never
    have to flag it manually. Enumeration runs in a fresh subprocess, so a
    wedged in-process SDK client no longer silently downgrades a stereo camera
    to mono. Any remaining failure (daemon down, subprocess crash) returns an
    empty set — in which case callers treat cameras as mono — but it is logged
    loudly so the cause is visible rather than masquerading as "no cameras".
    """
    try:
        return {d["serial"] for d in list_zed_devices() if d["kind"] == "stereo"}
    except ImportError:
        # pyzed absent (sim / dev host): legitimately no local cameras, not a
        # fault — keep it quiet so it doesn't spam the non-hardware paths.
        _logger.debug("stereo_serials: pyzed not installed; cameras treated as mono")
        return set()
    except Exception:  # noqa: BLE001 - detection is best-effort, but surface it
        _logger.warning(
            "stereo_serials: ZED enumeration failed; treating all cameras as "
            "mono (a stereo camera will open on the wrong pipeline)",
            exc_info=True,
        )
        return set()
