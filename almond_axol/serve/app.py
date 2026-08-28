"""FastAPI application for ``axol serve``.

Exposes a tiny JSON API the web control panel uses to list commands, launch
and stop sessions, and stream logs over a WebSocket. When a built web bundle
is available it is served too, with SPA-style fallback to ``index.html``.
"""

from __future__ import annotations

import asyncio
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from ..constants import URDF_PATH
from ..utils import adb, ports
from ..utils.certs import ACCEPT_PAGE_HTML
from ..utils.sudo import prime_sudo
from .commands import COMMANDS, command_specs, operation_ids
from .manager import Session, SessionManager
from .robot_link import RobotLink, scoped_motor_faults
from .runner import OperationRunner
from .settings import SettingsStore, advanced_schema, settings_schema
from .telemetry import DiagnosticsRunStore, TelemetryHub
from .update import SelfUpdater


class RunRequest(BaseModel):
    command: str
    args: dict[str, Any] = {}


class DiagnosticsRunRequest(BaseModel):
    """Launch a catalog command as a *diagnostics run*: the session is wrapped
    in a persisted run record with the telemetry observed while it ran."""

    command: str
    args: dict[str, Any] = {}


class OpStartRequest(BaseModel):
    """Start one of the four in-process core operations.

    ``cameras`` (optional) carries the local ZED camera setup for teleop /
    collect-data / run-policy, e.g.::

        {
          "serials": {"overhead": 41234567, "left_arm": ..., "right_arm": ...},
          "stream_resolution": "HD1200",   # capture res → headset; "off" disables
          "record_resolution": "SVGA",     # dataset downscale; "off" disables
          "stream": {"overhead": "both", "left_arm": true},   # per-slot headset
          "record": {"overhead": "left", "left_arm": false}   # per-slot dataset
        }

    The ``stream`` / ``record`` maps decide per camera whether it takes part in
    each branch: ``false`` opts a camera out, ``true`` opts a mono camera in, and
    an eye name (``"both"`` / ``"left"`` / ``"right"``) opts a stereo camera in
    with that eye selection. The runner folds all of this into the operation's
    config (serials, capture/record resolution, per-camera stream/record enable,
    per-eye selection). Whether a slot is stereo is auto-detected from its
    serial, not passed in. The legacy ``"resolution"`` key is still accepted as
    the streaming resolution.
    """

    op: str
    args: dict[str, Any] = {}
    cameras: dict[str, Any] | None = None


class RobotConnectRequest(BaseModel):
    """Optional CAN interface selection for a robot-link connect.

    ``channelsSet`` distinguishes "connect with the stored/default interfaces"
    (an empty body) from an explicit selection. A ``None`` channel disables
    that arm, so a single non-Axol-hub adapter can drive one arm only. The
    selection is persisted to the shared settings
    (``robot.left_channel`` / ``robot.right_channel``), so every operation and
    later connects use it too.
    """

    leftChannel: str | None = None
    rightChannel: str | None = None
    channelsSet: bool = False


class SettingsUpdateRequest(BaseModel):
    """Partial update of the shared operator settings (serve/settings.py).

    ``values`` and ``advanced`` merge per key (``null`` resets a key to its
    default); ``cameras`` replaces the stored camera spec wholesale. Omitted
    sections are left untouched.
    """

    values: dict[str, Any] | None = None
    cameras: dict[str, Any] | None = None
    # Distinguish "clear the cameras" (null) from "don't touch them" (omitted).
    camerasSet: bool = False
    advanced: dict[str, Any] | None = None


class EpisodeRequest(BaseModel):
    """A control command for the running op, as named in its own snapshot.

    ``run-policy`` takes ``start`` | ``s`` | ``r`` | ``q``; ``collect-data``
    takes ``start`` | ``s`` | ``r`` | ``continue``; ``waypoints`` takes
    ``record`` | ``undo`` | ``clear`` | ``grip-left`` | ``grip-right`` |
    ``play`` | ``stop`` | ``quit``. The panel sends back whatever the op
    published in ``controls``.
    """

    command: str


class ProximityRequest(BaseModel):
    """Disable (default) or restore the headset's proximity sensor over adb.

    Disabling keeps the Quest awake with nobody wearing it, so headless
    sessions driven from the panel don't die when the headset is set down.
    """

    disabled: bool = True


class SessionInputRequest(BaseModel):
    """A line written to a session's stdin (answers an interactive prompt).

    Empty ``line`` (the default) sends a bare newline — i.e. "press Enter".
    """

    line: str = ""


# Ports the launched commands expose on the serve host.
_VIEWER_PORT = 8002  # viser sim 3D viewer
_VR_PORT = ports.VR_PORT  # VR teleop WebSocket server (shared with the adb tunnel)


def _lan_ip() -> str:
    """Best-effort LAN IP of this machine (the one a headset/peer can reach)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


# ARPHRD_CAN in /sys/class/net/<iface>/type — identifies CAN interfaces.
_ARPHRD_CAN = "280"


def _list_can_interfaces() -> list[dict[str, Any]]:
    """Every SocketCAN network interface on this host (name + up state).

    Lets the UI offer real choices when the Axol hub adapter isn't present
    (its named interfaces missing) and the operator must pick the interface(s)
    of whatever CAN adapter is attached instead.
    """
    interfaces: list[dict[str, Any]] = []
    for iface in sorted(Path("/sys/class/net").glob("*")):
        try:
            if iface.joinpath("type").read_text().strip() != _ARPHRD_CAN:
                continue
            flags = int(iface.joinpath("flags").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        interfaces.append({"name": iface.name, "up": bool(flags & 0x1)})
    return interfaces


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
    # after a release upgrade and self-heals a host that upgraded into this
    # build from an older release.

    manager = SessionManager()
    hub = TelemetryHub()
    settings = SettingsStore()
    # The link opens the interfaces configured in the shared settings (the
    # Axol hub's persistent names unless the operator picked others).
    left_channel, right_channel = settings.can_channels()
    robot = RobotLink(
        left_channel, right_channel, hub=hub, has_gripper=settings.has_gripper
    )
    runner = OperationRunner(robot, settings=settings)
    runs = DiagnosticsRunStore(hub)
    # ZED devices are exclusive. Hold this across preview capture and operation
    # startup so both paths make their idle check while owning one reservation.
    camera_reservation = asyncio.Lock()

    def _is_idle() -> bool:
        """Safe to restart: no operation running.

        A connected robot is fine -- restarting drops the CAN link, which simply
        reconnects after the relaunch; only an in-flight operation must not be
        interrupted.
        """
        if runner.is_running():
            return False
        return not any(s["status"] in ("starting", "running") for s in manager.list())

    # Surfaces "update available" (a newer release tag, found via read-only
    # `git ls-remote --tags`) to the control panel via /api/update/status and
    # applies an on-demand tag-pinned reinstall via /api/update/start,
    # restarting the process (systemd relaunches it) once idle. Nothing
    # upgrades automatically. No-ops for dev checkouts.
    updater = SelfUpdater(_is_idle)

    def _find_session(session_id: str) -> tuple[Session | None, Any]:
        """Resolve a session id to (session, owner) across runner + manager."""
        s = runner.get(session_id)
        if s is not None:
            return s, runner
        return manager.get(session_id), manager

    async def _motor_fault_response(
        scope_args: dict[str, Any] | None = None,
    ) -> JSONResponse | None:
        """Return the shared motor-fault rejection, or ``None`` when clear.

        ``scope_args`` (a diagnostics launch's request args) narrows the check
        to the motors that run will actually touch — an arm/joint-scoped run
        (guided zeroing of a joint subset, a one-arm ROM test, a single-motor
        tool) must not be blocked by faults on motors it never drives, e.g. a
        bench arm with only some motors on the bus.
        """
        faults = await asyncio.to_thread(robot.motor_faults)
        if scope_args:
            faults = scoped_motor_faults(faults, scope_args)
        if not faults:
            return None
        detail = ", ".join(
            f"{f['arm']} {f['joint'].lower()} ({f['problem']})" for f in faults
        )
        return JSONResponse(
            {"error": f"motor fault — fix before starting: {detail}"},
            status_code=409,
        )

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
        # Self-heal a host that upgraded into this build from an older release
        # (the old code never ran `axol provision`); idempotent, once per process.
        updater.ensure_provisioned()
        return {
            "hostname": socket.gethostname(),
            "lanIp": _lan_ip(),
            "viewerPort": _VIEWER_PORT,
            "vrPort": _VR_PORT,
            "version": updater.version,
            # Backend git commit, compared against the commit baked into the
            # web bundle at build time to warn about a UI/backend mismatch.
            # Against a release (tag-pinned) install a hosted UI compares
            # versions instead — its commit tracks main and legitimately
            # differs between releases.
            "commit": updater.commit,
            "releaseInstall": updater.release_install,
        }

    @app.get("/api/update/status")
    async def update_status(refresh: bool = False) -> dict[str, Any]:
        """Installed vs. latest release version so the UI can offer an update.

        ``refresh=1`` forces a synchronous remote check (used on connect / page
        load) so the result is current; the steady-state poll omits it and gets
        the cheap debounced/cached value.
        """
        return await updater.status(force=refresh)

    @app.post("/api/update/start")
    async def update_start() -> JSONResponse:
        """Apply a user-initiated upgrade; the server restarts onto new code."""
        started, reason = updater.start()
        if not started:
            return JSONResponse({"error": reason}, status_code=409)
        return JSONResponse({"started": True})

    # -- host power ----------------------------------------------------------

    async def _host_power(flag: str, verb: str) -> JSONResponse:
        """Run ``shutdown <flag> now`` on the serve host.

        Refused while an operation or session is running — cutting power mid-
        run would drop the arms. The hosted install runs as root; a dev serve
        escalates via ``sudo -n`` so a headless context fails fast instead of
        blocking on a password prompt.
        """
        if not _is_idle():
            return JSONResponse(
                {"error": "an operation or session is running — stop it first"},
                status_code=409,
            )

        def _run() -> tuple[bool, str]:
            cmd = ["shutdown", flag, "now"]
            if os.geteuid() != 0:
                if not prime_sudo():
                    return False, "root required (no passwordless sudo)"
                cmd = ["sudo", "-n", *cmd]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            return proc.returncode == 0, (proc.stderr or proc.stdout).strip()

        ok, detail = await asyncio.to_thread(_run)
        if not ok:
            return JSONResponse(
                {"error": f"{verb} failed: {detail or 'unknown error'}"},
                status_code=500,
            )
        return JSONResponse({"ok": True})

    @app.post("/api/host/shutdown")
    async def host_shutdown() -> JSONResponse:
        """Power off the serve host (``shutdown -h now``)."""
        return await _host_power("-h", "shutdown")

    @app.post("/api/host/restart")
    async def host_restart() -> JSONResponse:
        """Reboot the serve host (``shutdown -r now``)."""
        return await _host_power("-r", "restart")

    # -- robot connection (detached CAN + 1 Hz motor ping) ------------------

    @app.get("/api/robot/status")
    async def robot_status() -> dict[str, Any]:
        return robot.status()

    @app.post("/api/robot/connect", response_model=None)
    async def robot_connect(
        req: RobotConnectRequest | None = None,
    ) -> dict[str, Any] | JSONResponse:
        """Connect the robot link, optionally onto explicit CAN interfaces.

        An explicit selection (``channelsSet``) is persisted to the shared
        settings first; either way the link is (re)pointed at the settings'
        channels, so an interface change takes effect on the next connect.
        """
        if req is not None and req.channelsSet:
            if not req.leftChannel and not req.rightChannel:
                return JSONResponse(
                    {"error": "select a CAN interface for at least one arm"},
                    status_code=400,
                )
            if robot.status()["state"] == "busy":
                return JSONResponse(
                    {"error": "cannot change CAN interfaces while a task owns the bus"},
                    status_code=409,
                )
            settings.update(
                values={
                    "robot.left_channel": req.leftChannel or "null",
                    "robot.right_channel": req.rightChannel or "null",
                }
            )
        channels = settings.can_channels()
        if channels != robot.channels():
            if robot.status()["state"] == "busy":
                return JSONResponse(
                    {"error": "cannot change CAN interfaces while a task owns the bus"},
                    status_code=409,
                )
            await asyncio.to_thread(robot.disconnect)
            robot.set_channels(*channels)
        return await asyncio.to_thread(robot.connect)

    @app.post("/api/robot/disconnect")
    async def robot_disconnect() -> dict[str, Any]:
        return await asyncio.to_thread(robot.disconnect)

    @app.get("/api/can/interfaces")
    async def can_interfaces() -> dict[str, Any]:
        """SocketCAN interfaces on this host, for the CAN adapter picker."""
        return {"interfaces": await asyncio.to_thread(_list_can_interfaces)}

    @app.get("/api/robot/motors/{arm}/{joint}")
    async def robot_motor_details(arm: str, joint: str) -> JSONResponse:
        """One-motor full readout (the ``motor.info`` set) over the idle link."""
        try:
            details = await asyncio.to_thread(robot.motor_details, arm, joint)
        except KeyError:
            return JSONResponse({"error": "unknown motor"}, status_code=404)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(details)

    # -- motor telemetry (diagnostics dashboard) -----------------------------

    @app.get("/api/telemetry")
    async def telemetry_snapshot() -> dict[str, Any]:
        """Link state + latest fast frame + latest slow sweep for every motor."""
        return hub.snapshot()

    @app.get("/api/telemetry/history")
    async def telemetry_history(
        seconds: float = 120.0, max_frames: int = 2000
    ) -> dict[str, Any]:
        """Buffered telemetry frames for chart backfill on page load.

        ``slow`` carries the 1 Hz sweep history (temperature/voltage) so the
        temperature chart backfills too.
        """
        return {
            "frames": hub.history(seconds, max_frames),
            "slow": hub.slow_history(seconds),
            "timing": hub.timing_history(seconds, max_frames),
        }

    @app.websocket("/api/telemetry/ws")
    async def telemetry_ws(ws: WebSocket) -> None:
        """Live motor + control timing stream (see :mod:`.telemetry`)."""
        await ws.accept()
        queue = hub.subscribe()
        try:
            await ws.send_json({"type": "hello", **hub.snapshot()})
            while True:
                await ws.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(queue)

    # -- diagnostics runs (script launches with telemetry capture) -----------

    async def _watch_diagnostics_run(
        meta: dict[str, Any] | None, session: Session, uses_can_bus: bool
    ) -> None:
        """Wait for the session to end, return the bus, persist the run (if any)."""
        queue = manager.subscribe(session)
        try:
            while session.status in ("starting", "running", "stopping"):
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue  # re-check status: end-of-stream may have raced us
                if line is None:
                    break
        finally:
            manager.unsubscribe(session, queue)
            if uses_can_bus:
                await asyncio.to_thread(robot.reacquire)
        if meta is not None:
            await asyncio.to_thread(
                runs.finalize,
                meta,
                session.status,
                session.exit_code,
                list(session.log),
            )

    @app.post("/api/diagnostics/run")
    async def diagnostics_run(req: DiagnosticsRunRequest) -> JSONResponse:
        # Diagnostics commands open the CAN bus (or reconfigure its interfaces)
        # themselves, so the launch does the same single-owner dance as the
        # in-process operations: refuse while something else owns the bus, and
        # hand the idle link's buses over for the duration of the run.
        command = COMMANDS.get(req.command)
        if command is None:
            return JSONResponse(
                {"error": f"unknown command: {req.command}"}, status_code=400
            )
        if runner.is_running():
            return JSONResponse(
                {"error": "an operation is running — stop it first"}, status_code=409
            )
        if any(s["status"] in ("starting", "running") for s in manager.list()):
            return JSONResponse(
                {"error": "another session is running — stop it first"},
                status_code=409,
            )
        if command.drives_motors:
            fault_response = await _motor_fault_response(scope_args=req.args)
            if fault_response is not None:
                return fault_response

        # A camera-only diagnostic (ZED cable check) doesn't touch the CAN bus,
        # so leave the idle motor telemetry streaming while it runs.
        uses_can_bus = command.uses_can_bus
        if uses_can_bus:
            await asyncio.to_thread(robot.release)
        try:
            # A writable stdin lets the UI answer the diagnostic's hands-on
            # prompts (the "Continue" button) via /input below.
            session = await manager.start(req.command, req.args, stdin_pipe=True)
        except Exception:
            if uses_can_bus:
                await asyncio.to_thread(robot.reacquire)
            raise
        # Only the Diagnostics tests are recorded in the run history; the
        # ad-hoc launches (CAN bring-up, motor calibration tools) still get
        # the bus handover + prompt plumbing but leave no record behind.
        record = command.category == "Diagnostics"
        meta = runs.begin(session.id, req.command, req.args) if record else None
        if session.status == "error":
            if uses_can_bus:
                await asyncio.to_thread(robot.reacquire)
            if meta is not None:
                await asyncio.to_thread(
                    runs.finalize,
                    meta,
                    session.status,
                    session.exit_code,
                    list(session.log),
                )
        else:
            asyncio.create_task(_watch_diagnostics_run(meta, session, uses_can_bus))
        return JSONResponse({"run": meta, "session": session.to_dict()})

    @app.get("/api/diagnostics/runs")
    async def diagnostics_runs() -> dict[str, Any]:
        return {"runs": await asyncio.to_thread(runs.list)}

    @app.delete("/api/diagnostics/runs")
    async def diagnostics_runs_clear() -> dict[str, Any]:
        """Delete the whole run history (the dashboard's Clear button)."""
        return {"removed": await asyncio.to_thread(runs.clear)}

    @app.get("/api/diagnostics/runs/{run_id}")
    async def diagnostics_run_data(run_id: str) -> JSONResponse:
        data = await asyncio.to_thread(runs.load, run_id)
        if data is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        return JSONResponse(data)

    # -- tuning runs (sine/step probes, motion replays, offline suites) -------
    #
    # These are the artifacts the tuning library persists under
    # ~/.almond/diagnostics/tuning/ (see almond_axol.tuning.runs). Distinct
    # from the diagnostics run *history* above: a tuning run is a scored
    # experiment with full time series, made for charting and A/B comparison.

    @app.get("/api/tuning/gains")
    async def tuning_gains() -> dict[str, Any]:
        """Effective per-joint control gains for both arms.

        Shipping defaults from ``config.py`` with this robot's calibration
        file overlaid — exactly what a tuning run uses when a gain field is
        left empty. The workbench shows these as the slider baselines.
        ``kd_host_hz`` is resolved to the shared default where a joint
        doesn't set its own band centre.
        """
        import math

        from ..constants import ARM_JOINTS
        from ..robot.config import AxolConfig
        from ..robot.control import DAMP_BP_Q, DAMP_BP_W0

        def _load() -> dict[str, Any]:
            cfg = AxolConfig()
            out: dict[str, Any] = {}
            for side in ("left", "right"):
                arm_cfg = getattr(cfg, side)
                joints: dict[str, Any] = {}
                for j in ARM_JOINTS:
                    jc = getattr(arm_cfg, j.value)
                    joints[j.value] = {
                        "kp": jc.kp,
                        "kd": jc.kd,
                        "kd_host": jc.kd_host,
                        "kd_host_hz": (
                            jc.kd_host_hz
                            if jc.kd_host_hz is not None
                            else round(DAMP_BP_W0 / (2 * math.pi), 1)
                        ),
                        "kd_host_q": (
                            jc.kd_host_q if jc.kd_host_q is not None else DAMP_BP_Q
                        ),
                        "j_eff": jc.j_eff,
                    }
                out[side] = joints
            return out

        return {"gains": await asyncio.to_thread(_load)}

    @app.get("/api/tuning/runs")
    async def tuning_runs() -> dict[str, Any]:
        from ..tuning import list_runs

        return {"runs": await asyncio.to_thread(list_runs)}

    @app.delete("/api/tuning/runs")
    async def tuning_runs_clear() -> dict[str, Any]:
        from ..tuning import clear_runs

        return {"removed": await asyncio.to_thread(clear_runs)}

    @app.get("/api/tuning/runs/{run_id}")
    async def tuning_run_data(run_id: str, max_points: int = 4000) -> JSONResponse:
        """One run's metadata plus its time series, decimated for charting.

        ``max_points`` caps each series' length (stride decimation — plenty
        for on-screen charts; the full-resolution NPZ stays on disk for
        offline analysis). NaN samples become null in the JSON.
        """
        from ..tuning import load_run

        loaded = await asyncio.to_thread(load_run, run_id)
        if loaded is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        meta, series = loaded

        def _decimate() -> dict[str, list[float | None]]:
            def to_json(col: np.ndarray, stride: int) -> list[float | None]:
                return [float(v) if math.isfinite(v) else None for v in col[::stride]]

            out: dict[str, list[float | None]] = {}
            for key, arr in series.items():
                a = np.asarray(arr, dtype=float)
                if a.ndim == 0 or len(a) == 0:
                    continue
                stride = max(1, len(a) // max(max_points, 2))
                if a.ndim == 1:
                    out[key] = to_json(a, stride)
                else:
                    # Multi-column series (e.g. a motion run's N×14 joint
                    # matrix) become one flat key per column, "<key>/<i>";
                    # column names live in meta.params (e.g. "columns").
                    cols = a.reshape(len(a), -1)
                    for i in range(cols.shape[1]):
                        out[f"{key}/{i}"] = to_json(cols[:, i], stride)
            return out

        return JSONResponse(
            {"meta": meta, "series": await asyncio.to_thread(_decimate)}
        )

    @app.delete("/api/tuning/runs/{run_id}")
    async def tuning_run_delete(run_id: str) -> JSONResponse:
        from ..tuning import delete_run

        if not await asyncio.to_thread(delete_run, run_id):
            return JSONResponse({"error": "unknown run"}, status_code=404)
        return JSONResponse({"deleted": run_id})

    @app.get("/api/tuning/motions")
    async def tuning_motions() -> dict[str, Any]:
        """The committed reference motions available for tune.motion replays."""
        from ..tuning.motion import list_motions

        def _list() -> list[dict[str, Any]]:
            return [
                {
                    "name": m.name,
                    "rate": m.rate,
                    "samples": len(m.q),
                    "durationS": len(m.q) / m.rate if m.rate else 0.0,
                    "meta": m.meta,
                }
                for m in list_motions()
            ]

        return {"motions": await asyncio.to_thread(_list)}

    @app.get("/api/tuning/recordings")
    async def tuning_recordings() -> dict[str, Any]:
        """The flight recordings motion.build can consume, newest first.

        From either recorder: a teleop session (``--teleop.record``) or a
        hand-guided gravity-comp one (``--record``). Feeds the workbench's
        Build-motion recording picker.
        """
        from ..teleop.recorder import list_recordings

        def _list() -> list[dict[str, Any]]:
            return [
                {
                    "name": r["name"],
                    "kind": r["kind"],
                    "modifiedAt": r["modified_at"],
                    "durationS": r["duration_s"],
                }
                for r in list_recordings()
            ]

        return {"recordings": await asyncio.to_thread(_list)}

    # -- local ZED cameras ---------------------------------------------------

    @app.get("/api/cameras/detect")
    async def cameras_detect() -> dict[str, Any]:
        """List locally connected ZED cameras (serial, model, mono/stereo)."""
        return await asyncio.to_thread(_detect_cameras)

    @app.get("/api/cameras/preview/{serial}", response_model=None)
    async def camera_preview(serial: int) -> Response | JSONResponse:
        """One live JPEG frame from a connected ZED, so operators can tell
        which physical camera a serial belongs to. Cameras are exclusive:
        refused while an operation may be using them."""
        async with camera_reservation:
            if runner.is_running():
                return JSONResponse(
                    {"error": "cannot preview cameras while an operation is running"},
                    status_code=409,
                )

            def _capture() -> bytes:
                from ..zed.snapshot import snapshot_jpeg

                return snapshot_jpeg(serial)

            try:
                data = await asyncio.to_thread(_capture)
            except ImportError:
                return JSONResponse(
                    {"error": "pyzed is not installed — run `axol zed.install` first"},
                    status_code=503,
                )
            except KeyError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except Exception as exc:  # noqa: BLE001 - surface capture errors to the UI
                return JSONResponse(
                    {"error": f"{type(exc).__name__}: {exc}"}, status_code=502
                )
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

    @app.post("/api/cameras/restart-daemon")
    async def cameras_restart_daemon() -> JSONResponse:
        """Restart the ZED X daemon so cameras plugged in after boot enumerate."""
        async with camera_reservation:
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

    # -- shared operator settings (see serve/settings.py) --------------------

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        """Stored shared settings + the schemas describing every category."""
        return {
            **settings.snapshot(),
            "schema": settings_schema(),
            "advancedSchema": advanced_schema(),
        }

    @app.put("/api/settings")
    async def put_settings(req: SettingsUpdateRequest) -> JSONResponse:
        try:
            snapshot = settings.update(
                values=req.values,
                cameras=req.cameras
                if (req.camerasSet or req.cameras is not None)
                else ...,
                advanced=req.advanced,
            )
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(snapshot)

    # -- datasets on disk (the replay / collect-data dataset picker) ----------

    @app.get("/api/datasets")
    async def get_datasets() -> dict[str, Any]:
        """LeRobot datasets on this host, newest first.

        Scans the shared ``recording.root`` setting when set (the directory
        collect-data writes to), otherwise the LeRobot cache dir — the same
        place replay-dataset resolves a bare repo id against.
        """
        from pathlib import Path

        from ..recording.datasets import list_datasets

        stored_root = settings.snapshot()["values"].get("recording.root")
        base = Path(str(stored_root)).expanduser() if stored_root else None
        found = await asyncio.to_thread(list_datasets, base)
        return {
            "datasets": [
                {
                    "repoId": d.repo_id,
                    "root": d.root,
                    "episodes": d.episodes,
                    "fps": d.fps,
                }
                for d in found
            ]
        }

    # -- robot model (URDF + meshes for the pose editor) ---------------------

    @app.get("/api/urdf/{asset_path:path}", response_model=None)
    async def urdf_asset(asset_path: str) -> FileResponse | JSONResponse:
        """Serve the robot URDF and its STL meshes to the web pose editor."""
        base = URDF_PATH.parent.resolve()
        target = (base / asset_path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        media = "model/stl" if target.suffix == ".stl" else "application/xml"
        return FileResponse(target, media_type=media)

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

    @app.post("/api/usb/proximity")
    async def usb_proximity(req: ProximityRequest) -> JSONResponse:
        """Disable/restore the headset's proximity sensor (`adb shell am broadcast`).

        Disabled, the headset stays awake with nobody wearing it — headless
        sessions driven from the panel keep their pose stream and camera relay.
        The override holds until restored or the headset reboots. Needs an
        attached, authorized headset (same requirement as the pose tunnel).
        """
        ok, error = await asyncio.to_thread(adb.set_proximity_disabled, req.disabled)
        if not ok:
            return JSONResponse(
                {"error": error or "adb broadcast failed"}, status_code=502
            )
        return JSONResponse({"ok": True})

    # -- in-process operations (teleop / gravity / collect / policy) --------

    @app.get("/api/op/status")
    async def op_status() -> dict[str, Any]:
        session = runner.current()
        return {
            "running": runner.is_running(),
            "session": session.to_dict() if session else None,
            "policy": runner.policy_state(),
        }

    @app.post("/api/op/start")
    async def op_start(req: OpStartRequest) -> JSONResponse:
        if req.op not in operation_ids():
            return JSONResponse(
                {"error": f"unknown operation: {req.op}"}, status_code=400
            )
        async with camera_reservation:
            # A faulted motor (over-temp, stall, encoder error, unreachable, …)
            # must block every hardware operation — driving through a fault risks
            # the arm. A sim run never touches the motors, and a robot-free run
            # (teleop's cart_only) never touches the *arms*, so both stay allowed.
            cmd = COMMANDS[req.op]
            is_sim = cmd.sim_flag is not None and bool(req.args.get(cmd.sim_flag))
            robot_free = is_sim or any(
                bool(req.args.get(flag)) for flag in cmd.robot_free_flags
            )
            if not robot_free:
                fault_response = await _motor_fault_response()
                if fault_response is not None:
                    return fault_response
            try:
                session = runner.start(
                    req.op,
                    req.args,
                    cameras=req.cameras,
                    loop=asyncio.get_running_loop(),
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
            return JSONResponse({"error": "no episode control active"}, status_code=409)
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

    @app.post("/api/sessions/{session_id}/input")
    async def session_input(session_id: str, req: SessionInputRequest) -> JSONResponse:
        """Answer a session's interactive prompt (the diagnostics Continue button)."""
        session, _owner = _find_session(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        ok = await session.send_input(req.line)
        if not ok:
            return JSONResponse(
                {"error": "session is not accepting input"}, status_code=409
            )
        return JSONResponse({"ok": True})

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
