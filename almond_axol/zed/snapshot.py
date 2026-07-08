"""One-shot JPEG previews of locally connected ZED cameras.

Used by the control panel's Cameras tab to show a live frame next to each
detected serial, so an operator can tell which physical camera is which
before assigning slots.

Like :mod:`.devices`, the capture runs in a short-lived ``spawn`` subprocess:
the ZED SDK's RPC client to ``zed_x_daemon`` is process-global and can wedge
for the life of a long-lived host process, and ``open()``/``grab()`` can block
in native code — a fresh subprocess with a hard timeout keeps the serve
process safe either way.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing.connection

_logger = logging.getLogger(__name__)

# Opening a GMSL camera + first grab takes a few seconds; the cap bounds a hung
# daemon / dead link so the API request never waits forever.
_SNAPSHOT_TIMEOUT_S = 20.0

# Grab attempts before giving up (the first frames after open can fail while
# the link settles).
_GRAB_ATTEMPTS = 30

# Preview width in pixels (height follows the aspect ratio).
_PREVIEW_WIDTH = 480


def snapshot_jpeg_inproc(serial: int) -> bytes:
    """Grab one frame from the camera with ``serial`` and return it as JPEG.

    Prefer :func:`snapshot_jpeg`, which runs this in a fresh subprocess. Mono
    ZED-X One cameras open via ``sl.CameraOne``; stereo ZED X via ``sl.Camera``
    (left eye), matching how the capture pipelines open them.

    Raises:
        ImportError: If pyzed is not installed.
        KeyError: If no connected camera has ``serial``.
        ConnectionError: If the camera fails to open or produces no frame.
    """
    import cv2
    import pyzed.sl as sl

    from .devices import list_zed_devices_inproc

    kinds = {d["serial"]: d["kind"] for d in list_zed_devices_inproc()}
    if serial not in kinds:
        raise KeyError(f"no connected ZED camera with serial {serial}")
    stereo = kinds[serial] == "stereo"

    if stereo:
        cam: object = sl.Camera()
        params: object = sl.InitParameters()
    else:
        cam = sl.CameraOne()
        params = sl.InitParametersOne()
    params.set_from_serial_number(serial)

    err = cam.open(params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise ConnectionError(f"failed to open camera {serial}: {err}")
    try:
        image = sl.Mat()
        for _ in range(_GRAB_ATTEMPTS):
            if cam.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            ok = (
                cam.retrieve_image(image, sl.VIEW.LEFT)
                if stereo
                else cam.retrieve_image(image)
            )
            if ok == sl.ERROR_CODE.SUCCESS:
                break
        else:
            raise ConnectionError(f"camera {serial} produced no frame")
        frame = image.get_data()  # BGRA
    finally:
        cam.close()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    height, width = frame.shape[:2]
    if width > _PREVIEW_WIDTH:
        frame = cv2.resize(
            frame, (_PREVIEW_WIDTH, round(height * _PREVIEW_WIDTH / width))
        )
    encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not encoded:
        raise ConnectionError(f"failed to encode preview for camera {serial}")
    return buf.tobytes()


def _snapshot_worker(
    conn: "multiprocessing.connection.Connection", serial: int
) -> None:
    """Subprocess entry: capture with a fresh SDK client, send the JPEG back."""
    try:
        conn.send(("ok", snapshot_jpeg_inproc(serial)))
    except Exception as exc:  # noqa: BLE001 - report every failure to the parent
        conn.send(("err", type(exc).__name__, str(exc)))
    finally:
        conn.close()


def snapshot_jpeg(serial: int, timeout_s: float = _SNAPSHOT_TIMEOUT_S) -> bytes:
    """Capture one JPEG preview frame from the camera with ``serial``.

    Runs the capture in a short-lived ``spawn`` subprocess (fresh SDK RPC
    client, bounded by ``timeout_s``) — see the module docstring.

    Raises:
        ImportError: If pyzed is not installed in the subprocess.
        TimeoutError: If the subprocess doesn't answer within ``timeout_s``.
        KeyError: If no connected camera has ``serial``.
        RuntimeError: For any other capture failure (original error preserved).
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_snapshot_worker,
        args=(child_conn, serial),
        daemon=True,
        name=f"zed-snapshot-{serial}",
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_s):
            raise TimeoutError(
                f"camera {serial} preview did not complete within {timeout_s:.0f}s"
            )
        try:
            status, *payload = parent_conn.recv()
        except EOFError:
            raise RuntimeError(
                f"camera {serial} preview subprocess exited without a result"
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
    if error_type in ("ImportError", "ModuleNotFoundError"):
        raise ImportError(message)
    if error_type == "KeyError":
        raise KeyError(message)
    raise RuntimeError(f"{error_type}: {message}")
