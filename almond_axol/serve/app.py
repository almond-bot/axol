"""FastAPI application for ``axol serve``.

Exposes a tiny JSON API the web control panel uses to list commands, launch
and stop sessions, and stream logs over a WebSocket. When a built web bundle
is available it is served too, with SPA-style fallback to ``index.html``.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..utils import adb
from ..utils.certs import ACCEPT_PAGE_HTML
from .commands import command_specs
from .manager import Session, SessionManager
from .robot_link import RobotLink
from .runner import OperationRunner
from .update import SelfUpdater

# The four core operations run in-process via the OperationRunner.
_OPERATIONS = {"teleop", "gravity-comp", "collect-data", "run-policy"}


class RunRequest(BaseModel):
    command: str
    args: dict[str, Any] = {}


class OpStartRequest(BaseModel):
    """Start one of the four in-process core operations.

    ``cameras`` (optional) carries the local ZED camera setup for teleop /
    collect-data / run-policy: ``{"serials": {"overhead": 41234567, ...},
    "resolution": "SVGA" | None}``. The runner folds it into the operation's
    config (per-camera serials, capture resolution); whether the overhead is
    stereo is auto-detected from its serial, not passed in.
    """

    op: str
    args: dict[str, Any] = {}
    cameras: dict[str, Any] | None = None


class EpisodeRequest(BaseModel):
    """run-policy episode control command: ``start`` | ``s`` | ``r`` | ``q``."""

    command: str


# Ports the launched commands expose on the serve host.
_VIEWER_PORT = 8002  # viser sim 3D viewer
_VR_PORT = 8000  # VR teleop WebSocket server


def _lan_ip() -> str:
    """Best-effort LAN IP of this machine (the one a headset/peer can reach)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _detect_cameras() -> dict[str, Any]:
    """Enumerate locally connected ZED cameras; never raises.

    Returns ``{"devices": [...], "error": str | None}`` — an empty device
    list with an error message when the ZED SDK / pyzed is unavailable.
    """
    try:
        from ..zed import list_zed_devices

        return {"devices": list_zed_devices(), "error": None}
    except ImportError:
        return {
            "devices": [],
            "error": "pyzed is not installed — run `axol zed.install` first",
        }
    except Exception as exc:  # noqa: BLE001 - SDK errors surface to the UI
        return {"devices": [], "error": f"{type(exc).__name__}: {exc}"}


def _usb_status_dict(status: adb.AdbStatus) -> dict[str, Any]:
    """Serialize the adb device + reverse-tunnel status for the UI."""
    return {
        "installed": status.installed,
        "serial": status.serial,
        "state": status.state,
        "reverseActive": status.reverse_active,
        "ready": status.ready,
    }


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="axol serve")
    # System setup (Jetson clock pinning, GStreamer install) is owned by the
    # host installer and its boot service (`axol jetson.setup` runs as an
    # ExecStartPre on axol.service; `axol provision` runs at install time). The
    # one exception is the self-updater (below), which re-runs `axol provision`
    # after a `uv tool upgrade` and self-heals a host that upgraded into this
    # build from an older main.

    manager = SessionManager()
    robot = RobotLink()
    runner = OperationRunner(robot)

    def _is_idle() -> bool:
        """Safe to restart: nothing running and no live robot link."""
        if runner.is_running():
            return False
        if any(s["status"] in ("starting", "running") for s in manager.list()):
            return False
        return not robot.status()["connected"]

    # Keeps a hosted (uv tool) install in sync with main: poked by the UI's
    # connect/poll endpoints, upgrades in the background, and restarts the
    # process (systemd relaunches it) once idle. No-ops for dev checkouts.
    updater = SelfUpdater(_is_idle)

    def _find_session(session_id: str) -> tuple[Session | None, Any]:
        """Resolve a session id to (session, owner) across runner + manager."""
        s = runner.get(session_id)
        if s is not None:
            return s, runner
        return manager.get(session_id), manager

    # Allow the Vite dev server (different origin) to call the API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/__accept")
    async def accept_cert() -> HTMLResponse:
        """Self-closing page the web UI opens to approve the self-signed cert.

        Registered before the SPA catch-all (mounted last) so it isn't shadowed.
        """
        return HTMLResponse(ACCEPT_PAGE_HTML)

    @app.get("/api/info")
    async def get_info() -> dict[str, Any]:
        """Identify the serve host so the UI can build reachable links/hints."""
        updater.poke()
        return {
            "hostname": socket.gethostname(),
            "lanIp": _lan_ip(),
            "viewerPort": _VIEWER_PORT,
            "vrPort": _VR_PORT,
            "commit": updater.commit,
        }

    # -- robot connection (detached CAN + 1 Hz motor ping) ------------------

    @app.get("/api/robot/status")
    async def robot_status() -> dict[str, Any]:
        return robot.status()

    @app.post("/api/robot/connect")
    async def robot_connect() -> dict[str, Any]:
        return await asyncio.to_thread(robot.connect)

    @app.post("/api/robot/disconnect")
    async def robot_disconnect() -> dict[str, Any]:
        return await asyncio.to_thread(robot.disconnect)

    # -- local ZED cameras ---------------------------------------------------

    @app.get("/api/cameras/detect")
    async def cameras_detect() -> dict[str, Any]:
        """List locally connected ZED cameras (serial, model, mono/stereo)."""
        return await asyncio.to_thread(_detect_cameras)

    @app.post("/api/cameras/restart-daemon")
    async def cameras_restart_daemon() -> JSONResponse:
        """Restart the ZED X daemon so cameras plugged in after boot enumerate."""
        if runner.is_running():
            return JSONResponse(
                {
                    "error": "cannot restart the ZED daemon while an operation is running"
                },
                status_code=409,
            )

        def _restart() -> dict[str, Any]:
            try:
                from ..zed import restart_zed_daemon

                restart_zed_daemon()
                return {"ok": True, "error": None}
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        result = await asyncio.to_thread(_restart)
        return JSONResponse(result, status_code=200 if result["ok"] else 500)

    # -- Quest-over-USB (adb reverse pose tunnel) ---------------------------

    @app.get("/api/usb/status")
    async def usb_status() -> dict[str, Any]:
        """adb device + reverse-tunnel status for the Quest-over-USB pose link."""
        return _usb_status_dict(await asyncio.to_thread(adb.status))

    @app.post("/api/usb/connect")
    async def usb_connect() -> dict[str, Any]:
        """Forward the headset's localhost:VR_PORT to this host via `adb reverse`.

        The first adb command against a freshly plugged-in headset also triggers
        the USB-debugging authorization popup on the device.
        """
        return _usb_status_dict(await asyncio.to_thread(adb.connect))

    # -- in-process operations (teleop / gravity / collect / policy) --------

    @app.get("/api/op/status")
    async def op_status() -> dict[str, Any]:
        # The UI polls this while the panel is open, so a long-lived tab still
        # picks up updates (the poke itself is debounced).
        updater.poke()
        session = runner.current()
        return {
            "running": runner.is_running(),
            "session": session.to_dict() if session else None,
        }

    @app.post("/api/op/start")
    async def op_start(req: OpStartRequest) -> JSONResponse:
        if req.op not in _OPERATIONS:
            return JSONResponse(
                {"error": f"unknown operation: {req.op}"}, status_code=400
            )
        try:
            session = runner.start(
                req.op, req.args, cameras=req.cameras, loop=asyncio.get_running_loop()
            )
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(session.to_dict())

    @app.post("/api/op/stop")
    async def op_stop() -> JSONResponse:
        session = await asyncio.to_thread(runner.stop)
        if session is None:
            return JSONResponse({"error": "no operation running"}, status_code=404)
        return JSONResponse(session.to_dict())

    @app.post("/api/op/episode")
    async def op_episode(req: EpisodeRequest) -> JSONResponse:
        ok = runner.episode_command(req.command)
        if not ok:
            return JSONResponse(
                {"error": "no run-policy episode control active"}, status_code=409
            )
        return JSONResponse({"ok": True})

    @app.get("/api/commands")
    async def get_commands() -> list[dict[str, Any]]:
        return command_specs()

    @app.get("/api/sessions")
    async def get_sessions() -> list[dict[str, Any]]:
        sessions = manager.list()
        current = runner.current()
        if current is not None:
            sessions.append(current.to_dict())
        return sessions

    @app.post("/api/run")
    async def run(req: RunRequest) -> JSONResponse:
        try:
            session = await manager.start(req.command, req.args)
        except KeyError:
            return JSONResponse(
                {"error": f"unknown command: {req.command}"}, status_code=400
            )
        return JSONResponse(session.to_dict())

    @app.post("/api/sessions/{session_id}/stop")
    async def stop(session_id: str) -> JSONResponse:
        # In-process operation sessions are stopped through the runner.
        if runner.get(session_id) is not None:
            session = await asyncio.to_thread(runner.stop)
            return JSONResponse(session.to_dict() if session else {"ok": True})
        ok = await manager.stop(session_id)
        if not ok:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        session = manager.get(session_id)
        return JSONResponse(session.to_dict() if session else {"ok": True})

    @app.get("/api/sessions/{session_id}/log")
    async def get_log(session_id: str, offset: int = 0) -> JSONResponse:
        """Offset-based log poll (HTTP alternative to the WebSocket below)."""
        session, _owner = _find_session(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        lines, next_offset = session.read_log(offset)
        return JSONResponse(
            {
                "lines": lines,
                "nextOffset": next_offset,
                "status": session.status,
                "exitCode": session.exit_code,
            }
        )

    @app.websocket("/api/sessions/{session_id}/logs")
    async def logs(ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        session, owner = _find_session(session_id)
        if session is None:
            await ws.send_json({"type": "error", "message": "unknown session"})
            await ws.close()
            return

        queue = owner.subscribe(session)
        try:
            # Replay the buffered backlog first.
            for line in list(session.log):
                await ws.send_json({"type": "log", "line": line})
            await ws.send_json({"type": "status", "session": session.to_dict()})

            while True:
                line = await queue.get()
                if line is None:
                    await ws.send_json({"type": "status", "session": session.to_dict()})
                    break
                await ws.send_json({"type": "log", "line": line})
        except WebSocketDisconnect:
            pass
        finally:
            owner.unsubscribe(session, queue)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await runner.shutdown()
        await manager.shutdown()
        await asyncio.to_thread(robot.shutdown)

    if static_dir is not None:
        _mount_spa(app, static_dir)

    return app


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the built web bundle with client-side-routing fallback.

    Vite emits content-hashed files under ``assets/`` (safe to cache forever);
    everything else — crucially ``index.html`` — is served ``no-cache`` so a
    rebuild is picked up immediately instead of the browser serving a stale
    ``index.html`` that points at deleted asset hashes.
    """
    index = static_dir / "index.html"
    immutable = {"Cache-Control": "public, max-age=31536000, immutable"}
    no_cache = {"Cache-Control": "no-cache"}

    @app.get("/{full_path:path}", response_model=None)
    async def spa(full_path: str) -> FileResponse | JSONResponse:
        if full_path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        candidate = static_dir / full_path
        if full_path and candidate.is_file():
            headers = immutable if full_path.startswith("assets/") else no_cache
            return FileResponse(candidate, headers=headers)
        if index.is_file():
            return FileResponse(index, headers=no_cache)
        return JSONResponse({"error": "web bundle not built"}, status_code=404)
