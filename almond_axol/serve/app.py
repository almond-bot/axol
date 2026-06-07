"""FastAPI application for ``axol serve``.

Exposes a tiny JSON API the web control panel uses to list commands, launch
and stop sessions, and stream logs over a WebSocket. When a built web bundle
is available it is served too, with SPA-style fallback to ``index.html``.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .commands import command_specs
from .manager import SessionManager
from .netdetect import best_eth_iface, list_eth_ifaces

# Orchestrated commands launch the full ZED bring-up (clock sync + streaming)
# instead of just the bare command when a ZED spec is supplied.
_ZED_COMMANDS = {"collect-data", "run-policy"}


class RunRequest(BaseModel):
    command: str
    args: dict[str, Any] = {}
    # When present with ``enabled`` true (and the command supports it), run the
    # multi-machine ZED orchestration (see :mod:`.orchestrator`).
    zed: dict[str, Any] | None = None


class SyncClocksRequest(BaseModel):
    """Remote ``zed.sync-clocks`` launch (host → ZED box, orchestrator only)."""

    role: str
    iface: str
    transport: str | None = None
    timestamping: str | None = None


class StreamRequest(BaseModel):
    """Remote ``zed.stream`` launch (host → ZED box, orchestrator only)."""

    overhead: str | None = None
    left_arm: str | None = None
    right_arm: str | None = None
    resolution: str | None = None
    fps: int | None = None
    bitrate: int | None = None
    setup_ip: str | None = None


# Ports the launched commands expose on the serve host.
_VIEWER_PORT = 8080  # viser sim 3D viewer
_VR_PORT = 8000  # VR teleop WebSocket server


def _lan_ip() -> str:
    """Best-effort LAN IP of this machine (the one a headset/peer can reach)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="axol serve")
    manager = SessionManager()

    # Allow the Vite dev server (different origin) to call the API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        from urllib.parse import urlsplit

        base = url.strip().rstrip("/")
        if "://" not in base:
            base = f"http://{base}"
        parts = urlsplit(base)
        if parts.port is None and parts.hostname:
            base = f"{parts.scheme}://{parts.hostname}:8090"
        try:
            with urllib.request.urlopen(f"{base}/api/info", timeout=5.0) as resp:
                return JSONResponse(json.loads(resp.read().decode()))
        except urllib.error.HTTPError as exc:
            return JSONResponse(
                {"error": f"box returned HTTP {exc.code}"}, status_code=502
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return JSONResponse(
                {"error": f"cannot reach ZED box: {exc}"}, status_code=502
            )

    @app.get("/api/commands")
    async def get_commands() -> list[dict[str, Any]]:
        return command_specs()

    @app.get("/api/sessions")
    async def get_sessions() -> list[dict[str, Any]]:
        return manager.list()

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
        """Launch ``zed.sync-clocks`` (driven remotely by a host orchestrator)."""
        argv = ["zed.sync-clocks", "--role", req.role, "--iface", req.iface]
        if req.transport:
            argv += ["--transport", req.transport]
        if req.timestamping:
            argv += ["--timestamping", req.timestamping]
        session = await manager.start_raw("zed.sync-clocks", argv)
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
        if req.setup_ip:
            argv += ["--setup-ip", req.setup_ip]
        session = await manager.start_raw("zed.stream", argv)
        return JSONResponse(session.to_dict())

    @app.post("/api/sessions/{session_id}/stop")
    async def stop(session_id: str) -> JSONResponse:
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
        session = manager.get(session_id)
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
        session = manager.get(session_id)
        if session is None:
            await ws.send_json({"type": "error", "message": "unknown session"})
            await ws.close()
            return

        queue = manager.subscribe(session)
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
            manager.unsubscribe(session, queue)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await manager.shutdown()

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
