"""Multi-machine ZED bring-up for ``collect-data`` / ``run-policy``.

``collect-data`` and ``run-policy`` need the ZED cameras streaming from the ZED
box (Jetson) with both machines' clocks PTP-synced first. Done by hand that's
four terminals across two machines in a strict order; this orchestrator drives
the whole sequence from a single web "Start", reusing each machine's own
``axol serve`` as the remote control surface.

Order (all torn down in reverse on stop / when the main command exits):

1. ``zed.sync-clocks --role master`` on this host          (local subprocess)
2. ``zed.sync-clocks --role slave``  on the ZED box        (box ``axol serve``)
3. wait until the slave's PTP offset locks under threshold
4. ``zed.stream`` on the ZED box                           (box ``axol serve``)
5. wait until the camera stream ports accept connections
6. ``collect-data`` / ``run-policy`` on this host          (local subprocess)

The box is reached over HTTP (no SSH): the host POSTs to the box's
``/api/zed/*`` endpoints and tails the resulting sessions via ``/api/sessions/
{id}/log``. Both machines must run the same ``axol`` version.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from .commands import build_argv
from .manager import Session, pump_into, spawn_proc

# Map each camera slot to the TCP port ``zed.stream`` serves it on (the
# ``collect-data`` defaults: overhead 30000, left_arm 30002, right_arm 30004).
_CAMERA_PORTS = {"overhead": 30000, "left_arm": 30002, "right_arm": 30004}

# PTP is considered "locked" once the slave's master offset stays within this
# many nanoseconds for a few consecutive samples.
_PTP_LOCK_NS = 100_000
_PTP_LOCK_SAMPLES = 3
_PTP_TIMEOUT_S = 150.0

# How long to wait for each streamed camera port to start accepting connections.
_STREAM_TIMEOUT_S = 90.0

_OFFSET_RE = re.compile(r"master offset\s+(-?\d+)")


class OrchestrationError(Exception):
    """A step failed; the run is aborted and everything started is torn down."""


def _normalize_url(url: str) -> str:
    """Normalize a box address to ``http://host:8090`` form (defaults applied)."""
    from urllib.parse import urlsplit

    url = url.strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = f"http://{url}"
    parts = urlsplit(url)
    if parts.port is None and parts.hostname:
        url = f"{parts.scheme}://{parts.hostname}:8090"
    return url


def _post_json(
    url: str, payload: dict[str, Any], timeout: float = 10.0
) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class _Remote:
    """A session running on the ZED box, tailed into the local log stream."""

    def __init__(self, base: str, label: str, session_id: str) -> None:
        self.base = base
        self.label = label
        self.id = session_id
        self.offset = 0


class ZedOrchestrator:
    def __init__(
        self,
        session: Session,
        command_id: str,
        args: dict[str, Any],
        zed: dict[str, Any],
    ) -> None:
        self.session = session
        self.command_id = command_id
        self.args = dict(args)
        self.zed = zed
        self.box: str = _normalize_url(str(zed.get("boxUrl", "")))
        self.topology: str = zed.get("topology", "direct")

        self._master_proc: asyncio.subprocess.Process | None = None
        self._main_proc: asyncio.subprocess.Process | None = None
        self._remotes: list[_Remote] = []
        self._tasks: list[asyncio.Task[Any]] = []
        self._ptp_samples = 0
        self._ptp_locked = asyncio.Event()
        self._stopping = False
        self._lock = asyncio.Lock()

    # -- public API ---------------------------------------------------------

    async def run(self) -> None:
        try:
            await self._run_steps()
        except OrchestrationError as exc:
            self._fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            self._fail(f"unexpected error: {exc!r}")

    async def stop(self) -> None:
        async with self._lock:
            if self._stopping:
                return
            self._stopping = True
        self.session.emit("[serve] stopping ZED orchestration…")
        await self._teardown()
        if self.session.status not in ("exited", "error"):
            self.session.status = "exited"
        self.session.close_stream()

    # -- step sequence ------------------------------------------------------

    async def _run_steps(self) -> None:
        if not self.box:
            raise OrchestrationError("no ZED box address configured")
        host_iface = str(self.zed.get("hostIface", "")).strip()
        box_iface = str(self.zed.get("boxIface", "")).strip()
        cameras = self._cameras()
        if not cameras:
            raise OrchestrationError("at least one ZED camera serial is required")

        self.session.status = "starting"

        # 1. Master clock sync on this host.
        self.session.emit("[serve] step 1/6 — PTP master clock sync (host)")
        if not host_iface:
            raise OrchestrationError("host ethernet interface is required")
        self._master_proc = await spawn_proc(
            ["zed.sync-clocks", "--role", "master", "--iface", host_iface]
        )
        self._spawn_task(
            pump_into(self._master_proc, self.session, prefix="master-sync")
        )

        # 2. Slave clock sync on the ZED box.
        self.session.emit("[serve] step 2/6 — PTP slave clock sync (ZED box)")
        if not box_iface:
            raise OrchestrationError("ZED box ethernet interface is required")
        slave = await self._box_run(
            "/api/zed/sync-clocks",
            {"role": "slave", "iface": box_iface},
            label="slave-sync",
        )
        self._tail_remote(slave, on_line=self._watch_ptp)

        # 3. Wait for PTP lock.
        self.session.emit("[serve] step 3/6 — waiting for clocks to lock…")
        await self._await(self._ptp_locked, _PTP_TIMEOUT_S, "clocks did not lock")
        self.session.emit("[serve] clocks locked")

        # 4. Start camera streaming on the ZED box.
        self.session.emit("[serve] step 4/6 — starting ZED camera streams (box)")
        stream_args: dict[str, Any] = dict(cameras)
        for opt in ("resolution", "fps", "bitrate"):
            if self.zed.get(opt) not in (None, ""):
                stream_args[opt] = self.zed[opt]
        if self.topology == "direct":
            stream_args["setup_ip"] = box_iface
        stream = await self._box_run("/api/zed/stream", stream_args, label="stream")
        self._tail_remote(stream)

        # 5. Wait for the stream ports to accept connections.
        self.session.emit("[serve] step 5/6 — waiting for camera streams…")
        await self._await_streams(cameras)
        self.session.emit("[serve] camera streams up")

        # 6. Run the main command locally.
        if self._stopping:
            return
        self.session.emit(f"[serve] step 6/6 — starting {self.command_id}")
        argv = build_argv(self.command_id, self._main_args())
        self._main_proc = await spawn_proc([self.command_id, *argv])
        self.session.proc = self._main_proc
        self.session.status = "running"
        self.session.emit(f"[serve] $ axol {self.command_id} {' '.join(argv)}".rstrip())
        rc = await pump_into(self._main_proc, self.session)
        self.session.exit_code = rc
        self.session.emit(f"[serve] {self.command_id} exited with code {rc}")
        await self.stop()

    # -- argv / camera helpers ---------------------------------------------

    def _cameras(self) -> dict[str, str]:
        """Provided camera serials keyed by ``zed.stream`` flag name."""
        raw = self.zed.get("cameras") or {}
        out: dict[str, str] = {}
        for slot in _CAMERA_PORTS:
            val = str(raw.get(slot, "")).strip()
            if val:
                out[slot] = val
        return out

    def _main_args(self) -> dict[str, Any]:
        """Form args augmented with the topology-derived network flags."""
        args = dict(self.args)
        if self.topology == "direct":
            # Receiver assigns 192.168.10.2 on its link NIC; cameras are at the
            # default zed_host 192.168.10.1, so only the iface needs forcing.
            args.setdefault("zed_iface", self.zed.get("hostIface", ""))
        else:
            zed_host = str(self.zed.get("zedHost", "")).strip()
            if zed_host:
                args.setdefault("robot_config.zed_host", zed_host)
        return args

    def _stream_host(self) -> str:
        if self.topology == "direct":
            return "192.168.10.1"
        return str(self.zed.get("zedHost", "")).strip() or self._box_host()

    def _box_host(self) -> str:
        # Strip scheme/port from the box URL for raw TCP probes.
        netloc = self.box.split("://", 1)[-1]
        return netloc.split(":", 1)[0].split("/", 1)[0]

    # -- box (remote) plumbing ---------------------------------------------

    async def _box_run(
        self, path: str, payload: dict[str, Any], *, label: str
    ) -> _Remote:
        url = f"{self.box}{path}"
        try:
            result = await asyncio.to_thread(_post_json, url, payload)
        except urllib.error.HTTPError as exc:
            raise OrchestrationError(
                f"{label}: ZED box returned HTTP {exc.code} for {path}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OrchestrationError(
                f"{label}: cannot reach ZED box at {self.box} ({exc})"
            ) from exc
        if result.get("error"):
            raise OrchestrationError(f"{label}: {result['error']}")
        session_id = result.get("id")
        if not session_id:
            raise OrchestrationError(f"{label}: ZED box did not return a session id")
        remote = _Remote(self.box, label, session_id)
        self._remotes.append(remote)
        return remote

    def _tail_remote(self, remote: _Remote, on_line: Any = None) -> None:
        self._spawn_task(self._tail_loop(remote, on_line))

    async def _tail_loop(self, remote: _Remote, on_line: Any) -> None:
        """Poll a box session's log and mirror it into the local stream."""
        url = f"{remote.base}/api/sessions/{remote.id}/log"
        while not self._stopping:
            try:
                data = await asyncio.to_thread(
                    _get_json, f"{url}?{urlencode({'offset': remote.offset})}"
                )
            except (urllib.error.URLError, OSError):
                await asyncio.sleep(1.0)
                continue
            for line in data.get("lines", []):
                self.session.emit(f"[{remote.label}] {line}")
                if on_line is not None:
                    on_line(line)
            remote.offset = data.get("nextOffset", remote.offset)
            if data.get("status") in ("exited", "error") and not data.get("lines"):
                # Remote step ended; if it wasn't us tearing down, that's fatal.
                if not self._stopping and remote.label != "stream":
                    self._ptp_locked.set()  # unblock any waiter so it can fail
                return
            await asyncio.sleep(0.5)

    async def _box_stop(self, remote: _Remote) -> None:
        try:
            await asyncio.to_thread(
                _post_json, f"{remote.base}/api/sessions/{remote.id}/stop", {}
            )
        except (urllib.error.URLError, OSError) as exc:
            self.session.emit(f"[serve] failed to stop {remote.label} on box: {exc}")

    # -- readiness gates ----------------------------------------------------

    def _watch_ptp(self, line: str) -> None:
        m = _OFFSET_RE.search(line)
        if m is None:
            return
        try:
            offset = abs(int(m.group(1)))
        except ValueError:
            return
        if offset <= _PTP_LOCK_NS:
            self._ptp_samples += 1
            if self._ptp_samples >= _PTP_LOCK_SAMPLES:
                self._ptp_locked.set()
        else:
            self._ptp_samples = 0

    async def _await(self, event: asyncio.Event, timeout: float, msg: str) -> None:
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise OrchestrationError(f"{msg} (timed out after {timeout:.0f}s)") from exc
        if self._stopping:
            raise OrchestrationError("stopped before ready")

    async def _await_streams(self, cameras: dict[str, str]) -> None:
        host = self._stream_host()
        ports = [_CAMERA_PORTS[slot] for slot in cameras]
        deadline = asyncio.get_event_loop().time() + _STREAM_TIMEOUT_S
        pending = set(ports)
        while pending:
            if self._stopping:
                raise OrchestrationError("stopped before streams were ready")
            if asyncio.get_event_loop().time() > deadline:
                missing = ", ".join(f"{host}:{p}" for p in sorted(pending))
                raise OrchestrationError(f"camera streams not reachable: {missing}")
            for port in list(pending):
                if await self._port_open(host, port):
                    pending.discard(port)
                    self.session.emit(f"[serve] stream port {host}:{port} ready")
            if pending:
                await asyncio.sleep(1.0)

    async def _port_open(self, host: str, port: int) -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2.0
            )
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True

    # -- teardown -----------------------------------------------------------

    def _spawn_task(self, coro: Any) -> None:
        self._tasks.append(asyncio.create_task(coro))

    async def _teardown(self) -> None:
        # Reverse order: main → streams → slave-sync → master-sync.
        if self._main_proc is not None:
            await _kill_local(self._main_proc, self.session)
        for remote in reversed(self._remotes):
            await self._box_stop(remote)
        if self._master_proc is not None:
            await _kill_local(self._master_proc, self.session)
        for task in self._tasks:
            task.cancel()

    def _fail(self, message: str) -> None:
        self.session.error = message
        self.session.status = "error"
        self.session.emit(f"[serve] error: {message}")
        asyncio.create_task(self.stop())


async def _kill_local(proc: asyncio.subprocess.Process, session: Session) -> None:
    """SIGINT → SIGTERM → SIGKILL a local subprocess group, like the manager."""
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, grace in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace)
            return
        except asyncio.TimeoutError:
            continue
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        await proc.wait()
    except Exception:  # pragma: no cover - defensive
        pass
