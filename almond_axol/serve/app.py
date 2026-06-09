"""FastAPI application for ``axol serve``.

Exposes a tiny JSON API the web control panel uses to list commands, launch
and stop sessions, and stream logs over a WebSocket. When a built web bundle
is available it is served too, with SPA-style fallback to ``index.html``.
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..certs import ACCEPT_PAGE_HTML
from ..sudo import SUDO_PASSWORD_ENV
from .commands import command_specs
from .manager import Session, SessionManager
from .netdetect import best_eth_iface, iface_owning, list_eth_ifaces
from .orchestrator import PtpLink, StreamLink, box_ssl_context
from .robot_link import RobotLink
from .runner import OperationRunner

# Orchestrated commands launch the full ZED bring-up (clock sync + streaming)
# instead of just the bare command when a ZED spec is supplied.
_ZED_COMMANDS = {"collect-data", "run-policy"}

# The four core operations run in-process via the OperationRunner.
_OPERATIONS = {"teleop", "gravity-comp", "collect-data", "run-policy"}


class RunRequest(BaseModel):
    command: str
    args: dict[str, Any] = {}
    # When present with ``enabled`` true (and the command supports it), run the
    # multi-machine ZED orchestration (see :mod:`.orchestrator`).
    zed: dict[str, Any] | None = None


class OpStartRequest(BaseModel):
    """Start one of the four in-process core operations."""

    op: str
    args: dict[str, Any] = {}
    # ZED spec (box url + camera/topology/iface settings) for collect-data /
    # run-policy; the box link comes from ``/api/zed/connect``.
    zed: dict[str, Any] | None = None


class EpisodeRequest(BaseModel):
    """run-policy episode control command: ``start`` | ``s`` | ``r`` | ``q``."""

    command: str


class RobotConnectRequest(BaseModel):
    """Optional sudo password used only if CAN bring-up needs root access."""

    password: str | None = None


class ZedConnectRequest(BaseModel):
    """Lightweight ZED box link: store url and verify reachability.

    ``password`` (optional) is forwarded to the PTP daemons on both machines
    when passwordless sudo isn't available; used once and never stored.

    ``cameras`` (optional) maps camera slot (``overhead`` / ``left_arm`` /
    ``right_arm``) to its ZED-X One serial. When any are given, streaming for
    those cameras starts once the clocks lock.
    """

    url: str
    password: str | None = None
    cameras: dict[str, str] | None = None


class SyncClocksRequest(BaseModel):
    """Remote ``zed.sync-clocks`` launch (host → ZED box, orchestrator only).

    The interface can be given explicitly (``iface``) or resolved on the box
    from the streaming IP it owns (``ip``); if neither pins a NIC the box
    falls back to its best wired interface.
    """

    role: str
    iface: str | None = None
    ip: str | None = None
    transport: str | None = None
    timestamping: str | None = None
    # Forwarded sudo password for the box's PTP daemons (never stored).
    password: str | None = None


class StreamRequest(BaseModel):
    """Remote ``zed.stream`` launch (host → ZED box, orchestrator only)."""

    overhead: str | None = None
    left_arm: str | None = None
    right_arm: str | None = None
    resolution: str | None = None
    fps: int | None = None
    bitrate: int | None = None


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


def _normalize_box_url(url: str) -> str:
    """Add scheme + default control port (8001) to a bare ZED box address.

    A bare IP defaults to ``https`` because ``axol serve`` is TLS by default
    (same as the workstation link); pass an explicit ``http://`` if the box was
    started with ``--no-tls``.
    """
    from urllib.parse import urlsplit

    base = url.strip().rstrip("/")
    if "://" not in base:
        base = f"https://{base}"
    parts = urlsplit(base)
    if parts.port is None and parts.hostname:
        base = f"{parts.scheme}://{parts.hostname}:8001"
    return base


def _fetch_box_info(url: str) -> tuple[bool, dict[str, Any]]:
    """Fetch the ZED box's ``/api/info``; ``(ok, data_or_error)``."""
    base = _normalize_box_url(url)
    try:
        with urllib.request.urlopen(
            f"{base}/api/info", timeout=5.0, context=box_ssl_context()
        ) as resp:
            return True, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return False, {"error": f"box returned HTTP {exc.code}"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, {"error": f"cannot reach ZED box: {exc}"}


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="axol serve")
    manager = SessionManager()
    robot = RobotLink()
    # PTP clock sync between this host and the box. Started when the box is
    # connected so the clocks are already disciplined before a task; reused by
    # the collect-data / run-policy orchestrator.
    ptp = PtpLink(manager)
    # ZED camera streams from the box; started on connect (after the clocks
    # lock) when serials are configured, and reused by the orchestrator.
    stream = StreamLink(manager, ptp)
    runner = OperationRunner(robot, ptp, stream)
    # Detached ZED box link (light reachability check; PTP clock sync + camera
    # streaming both start on connect). Mutated by /api/zed/connect|disconnect.
    zed_state: dict[str, Any] = {
        "connected": False,
        "boxUrl": None,
        "info": None,
        "error": None,
        "ptp": ptp.status(),
        "stream": stream.status(),
    }
    # Background task that re-pings the box while connected; the connect-time
    # reachability check is a one-shot, so without this the link would show
    # "connected" forever after the box is powered off.
    zed_monitor: dict[str, asyncio.Task[None] | None] = {"task": None}

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
        """Identify the serve host so the UI can build reachable links/hints.

        ``ethIface`` / ``ethIfaces`` let the control panel default (and offer a
        dropdown for) the wired ZED-link interface on this machine; the box's
        own values are fetched the same way through ``/api/zed/box-info``.
        """
        return {
            "hostname": socket.gethostname(),
            "lanIp": _lan_ip(),
            "viewerPort": _VIEWER_PORT,
            "vrPort": _VR_PORT,
            "ethIface": best_eth_iface(),
            "ethIfaces": list_eth_ifaces(),
        }

    @app.get("/api/zed/box-info")
    async def zed_box_info(url: str) -> JSONResponse:
        """Proxy the ZED box's ``/api/info`` (reachability + iface candidates).

        Proxied through the host so the browser avoids cross-origin / mixed
        content calls to the box and a single page reports both machines.
        """
        ok, data = await asyncio.to_thread(_fetch_box_info, url)
        return JSONResponse(data, status_code=200 if ok else 502)

    # -- robot connection (detached CAN + 1 Hz motor ping) ------------------

    @app.get("/api/robot/status")
    async def robot_status() -> dict[str, Any]:
        return robot.status()

    @app.post("/api/robot/connect")
    async def robot_connect(
        req: RobotConnectRequest | None = None,
    ) -> dict[str, Any]:
        password = req.password if req else None
        return await asyncio.to_thread(robot.connect, password)

    @app.post("/api/robot/disconnect")
    async def robot_disconnect() -> dict[str, Any]:
        return await asyncio.to_thread(robot.disconnect)

    # -- ZED box link (detached, lightweight) -------------------------------

    def _stop_zed_monitor() -> None:
        task = zed_monitor["task"]
        if task is not None:
            task.cancel()
            zed_monitor["task"] = None

    async def _zed_monitor(url: str) -> None:
        """Poll the box while connected; tear the link down if it disappears.

        The PTP master keeps running locally and the slave tail silently retries
        a dead box, so neither notices a power-off on its own — this heartbeat is
        what flips the UI back to disconnected (and unlocks the clock).
        """
        fails = 0
        while True:
            await asyncio.sleep(2.5)
            ok, _data = await asyncio.to_thread(_fetch_box_info, url)
            if ok:
                fails = 0
                continue
            fails += 1
            if fails < 2:  # tolerate a transient blip before tearing down
                continue
            zed_state["connected"] = False
            zed_state["error"] = "ZED box is no longer reachable"
            await stream.stop()
            await ptp.stop()
            zed_state["ptp"] = ptp.status()
            zed_state["stream"] = stream.status()
            zed_monitor["task"] = None
            return

    @app.get("/api/zed/status")
    async def zed_status() -> dict[str, Any]:
        zed_state["ptp"] = ptp.status()
        zed_state["stream"] = stream.status()
        return zed_state

    @app.post("/api/zed/connect")
    async def zed_connect(req: ZedConnectRequest) -> JSONResponse:
        ok, data = await asyncio.to_thread(_fetch_box_info, req.url)
        zed_state["boxUrl"] = _normalize_box_url(req.url)
        _stop_zed_monitor()
        if ok:
            zed_state["connected"] = True
            zed_state["info"] = data
            zed_state["error"] = None
            # Start clock sync immediately so the link is already locked by the
            # time a collect-data / run-policy task begins.
            zed_state["ptp"] = await ptp.start(req.url, req.password)
            # If camera serials were given, start streaming them too (it waits
            # for the clocks to lock first); a task then reuses the live feeds.
            zed_state["stream"] = await stream.start(req.url, req.cameras or {})
            # Heartbeat the box so the link drops if it goes away.
            zed_monitor["task"] = asyncio.create_task(_zed_monitor(req.url))
            return JSONResponse(zed_state)
        zed_state["connected"] = False
        zed_state["info"] = None
        zed_state["error"] = data.get("error", "unreachable")
        await stream.stop()
        await ptp.stop()
        zed_state["ptp"] = ptp.status()
        zed_state["stream"] = stream.status()
        return JSONResponse(zed_state, status_code=502)

    @app.post("/api/zed/disconnect")
    async def zed_disconnect() -> dict[str, Any]:
        _stop_zed_monitor()
        await stream.stop()
        await ptp.stop()
        zed_state.update(
            {
                "connected": False,
                "boxUrl": None,
                "info": None,
                "error": None,
                "ptp": ptp.status(),
                "stream": stream.status(),
            }
        )
        return zed_state

    # -- in-process operations (teleop / gravity / collect / policy) --------

    @app.get("/api/op/status")
    async def op_status() -> dict[str, Any]:
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
                req.op, req.args, zed=req.zed, loop=asyncio.get_running_loop()
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
        orchestrate = (
            req.zed is not None
            and bool(req.zed.get("enabled"))
            and req.command in _ZED_COMMANDS
        )
        try:
            if orchestrate:
                assert req.zed is not None
                session = await manager.start_orchestrated(
                    req.command, req.args, req.zed
                )
            else:
                session = await manager.start(req.command, req.args)
        except KeyError:
            return JSONResponse(
                {"error": f"unknown command: {req.command}"}, status_code=400
            )
        return JSONResponse(session.to_dict())

    @app.post("/api/zed/sync-clocks")
    async def zed_sync_clocks(req: SyncClocksRequest) -> JSONResponse:
        """Launch ``zed.sync-clocks`` (driven remotely by a host orchestrator).

        The PTP interface is resolved here on the box: an explicit ``iface``
        wins, else the NIC that owns the streaming ``ip``, else the best wired
        interface. Returns ``{"error": ...}`` if none can be determined.
        """
        iface = req.iface
        if not iface and req.ip:
            iface = iface_owning(req.ip)
        if not iface:
            iface = best_eth_iface()
        if not iface:
            return JSONResponse(
                {
                    "error": (
                        "could not determine the ZED box network interface "
                        f"(no NIC owns {req.ip!r} and no wired NIC was found)"
                    )
                }
            )
        argv = ["zed.sync-clocks", "--role", req.role, "--iface", iface]
        if req.transport:
            argv += ["--transport", req.transport]
        if req.timestamping:
            argv += ["--timestamping", req.timestamping]
        env_extra = {SUDO_PASSWORD_ENV: req.password} if req.password else None
        session = await manager.start_raw("zed.sync-clocks", argv, env_extra=env_extra)
        return JSONResponse(session.to_dict())

    @app.post("/api/zed/stream")
    async def zed_stream(req: StreamRequest) -> JSONResponse:
        """Launch ``zed.stream`` (driven remotely by a host orchestrator)."""
        argv = ["zed.stream"]
        for flag, value in (
            ("--overhead", req.overhead),
            ("--left-arm", req.left_arm),
            ("--right-arm", req.right_arm),
        ):
            if value:
                argv += [flag, str(value)]
        if req.resolution:
            argv += ["--resolution", req.resolution]
        if req.fps is not None:
            argv += ["--fps", str(req.fps)]
        if req.bitrate is not None:
            argv += ["--bitrate", str(req.bitrate)]
        session = await manager.start_raw("zed.stream", argv)
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
        """Offset-based log poll (used by a remote host orchestrator).

        The WebSocket below is for live browser streaming; this HTTP variant is
        what one ``axol serve`` uses to tail another's sessions.
        """
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
        _stop_zed_monitor()
        await runner.shutdown()
        await stream.stop()
        await ptp.stop()
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
