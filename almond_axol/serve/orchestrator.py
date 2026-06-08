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
import functools
import json
import os
import re
import signal
import ssl
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from ..sudo import (
    SUDO_BAD_PASSWORD_CODE as _SUDO_BAD_PASSWORD_CODE,
)
from ..sudo import (
    SUDO_BAD_PASSWORD_MARKER as _SUDO_BAD_PASSWORD_MARKER,
)
from ..sudo import (
    SUDO_PASSWORD_ENV as _SUDO_PASSWORD_ENV,
)
from ..sudo import (
    SUDO_REQUIRED_CODE as _SUDO_REQUIRED_CODE,
)
from ..sudo import (
    SUDO_REQUIRED_MARKER as _SUDO_REQUIRED_MARKER,
)
from .commands import build_argv
from .manager import Session, pump_into, spawn_proc
from .netdetect import best_eth_iface, iface_for_route

if TYPE_CHECKING:
    from .manager import SessionManager

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


@functools.cache
def box_ssl_context() -> ssl.SSLContext:
    """Unverified TLS context for a box's self-signed ``axol serve``.

    ``axol serve`` uses a self-signed cert by default, so over HTTPS we skip
    verification — the same trust model as the browser "accept the cert once"
    flow. Used for every host→box request (info, sync-clocks, stream, logs).
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _normalize_url(url: str) -> str:
    """Normalize a box address to ``https://host:8090`` form (defaults applied).

    A bare IP defaults to ``https`` since ``axol serve`` is TLS by default; an
    explicit ``http://`` is preserved for boxes started with ``--no-tls``.
    """
    from urllib.parse import urlsplit

    url = url.strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = f"https://{url}"
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
    with urllib.request.urlopen(
        req, timeout=timeout, context=box_ssl_context()
    ) as resp:
        return json.loads(resp.read().decode())


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        url, timeout=timeout, context=box_ssl_context()
    ) as resp:
        return json.loads(resp.read().decode())


class _Remote:
    """A session running on the ZED box, tailed into the local log stream."""

    def __init__(self, base: str, label: str, session_id: str) -> None:
        self.base = base
        self.label = label
        self.id = session_id
        self.offset = 0


class PtpLink:
    """Long-lived PTP clock-sync pair started when the ZED box is connected.

    Runs ``zed.sync-clocks --role master`` on this host and ``--role slave`` on
    the box so both system clocks stay disciplined while the rig is idle. A
    collect-data / run-policy task then reuses the locked link instead of
    bringing PTP up itself, so recording can start immediately. Lives on the
    server event loop; torn down on disconnect.

    The PTP interface is derived from the box address (this host uses whichever
    NIC routes to it; the box self-detects the NIC that owns it), so the
    operator only provides the box URL.
    """

    def __init__(self, manager: "SessionManager") -> None:
        self._manager = manager
        self.box: str = ""
        self.session: Session | None = None
        self._master_proc: asyncio.subprocess.Process | None = None
        self._slave: _Remote | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._locked = asyncio.Event()
        self._ptp_samples = 0
        self._offset_ns: int | None = None
        self._stopping = False
        self._needs_sudo = False
        self._bad_password = False
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        s = self.session
        return s is not None and s.status == "running" and not self._stopping

    @property
    def locked(self) -> bool:
        return self._locked.is_set()

    def status(self) -> dict[str, Any]:
        s = self.session
        error = s.error if (s is not None and s.status == "error") else None
        return {
            "running": self.running,
            "locked": self.locked,
            "offsetNs": self._offset_ns,
            "sessionId": s.id if s else None,
            # The PTP daemons need root; when passwordless sudo isn't available
            # (or the supplied password was wrong) the UI prompts for one.
            "needsSudo": self._needs_sudo or self._bad_password,
            "badPassword": self._bad_password,
            # Surface a failed link (e.g. the ptp4l/phc2sys daemons exiting) so
            # the UI shows "sync error" with a reason instead of "idle".
            "error": error,
        }

    async def start(
        self, box_url: str, sudo_password: str | None = None
    ) -> dict[str, Any]:
        """Bring up the master (local) + slave (box) PTP daemons. Idempotent.

        ``sudo_password`` (when supplied) is forwarded to both the local daemons
        and the box's so PTP can escalate without a tty; it's used once per
        daemon launch and never stored.
        """
        await self.stop()
        async with self._lock:
            self._stopping = False
            self._locked.clear()
            self._ptp_samples = 0
            self._offset_ns = None
            self._needs_sudo = False
            self._bad_password = False
            self.box = _normalize_url(box_url)
            if not self.box:
                return {
                    "running": False,
                    "locked": False,
                    "error": "no ZED box address",
                }

            session = self._manager.new_session("zed.sync-clocks")
            session.teardown = self.stop
            session.status = "running"
            self.session = session

            box_host = self._box_host()
            host_iface = iface_for_route(box_host) or best_eth_iface()
            if not host_iface:
                msg = (
                    "could not determine the host network interface for the ZED "
                    f"link to {box_host!r}"
                )
                self._fail(msg)
                return {"running": False, "locked": False, "error": msg}

            session.emit(
                f"[serve] PTP clock sync starting (host interface {host_iface})"
            )
            env_extra = {_SUDO_PASSWORD_ENV: sudo_password} if sudo_password else None
            try:
                self._master_proc = await spawn_proc(
                    ["zed.sync-clocks", "--role", "master", "--iface", host_iface],
                    env_extra,
                )
            except OSError as exc:
                self._fail(f"failed to start host PTP master: {exc}")
                return {"running": False, "locked": False, "error": str(exc)}
            self._spawn(self._run_master(self._master_proc))

            try:
                self._slave = await self._box_slave(box_host, sudo_password)
            except OrchestrationError as exc:
                self._fail(str(exc))
                return {"running": False, "locked": False, "error": str(exc)}
            self._spawn(self._tail(self._slave))
            return self.status()

    async def stop(self) -> None:
        async with self._lock:
            session = self.session
            if session is None:
                return
            self._stopping = True
            if self._master_proc is not None:
                await _kill_local(self._master_proc, session)
                self._master_proc = None
            if self._slave is not None:
                await self._box_stop(self._slave)
                self._slave = None
            for task in self._tasks:
                task.cancel()
            self._tasks = []
            if session.status == "running":
                session.status = "exited"
            session.emit("[serve] PTP clock sync stopped")
            session.close_stream()
            self.session = None
            self._locked.clear()
            self._needs_sudo = False
            self._bad_password = False

    # -- internals ----------------------------------------------------------

    def _box_host(self) -> str:
        netloc = self.box.split("://", 1)[-1]
        return netloc.split(":", 1)[0].split("/", 1)[0]

    def _spawn(self, coro: Any) -> None:
        self._tasks.append(asyncio.create_task(coro))

    def _fail(self, message: str) -> None:
        if self.session is not None:
            self.session.error = message
            self.session.status = "error"
            self.session.emit(f"[serve] error: {message}")

    async def _run_master(self, proc: asyncio.subprocess.Process) -> None:
        """Pump the host master daemon and flag the link if it exits early."""
        rc = await pump_into(
            proc, self.session, prefix="master-sync", on_line=self._watch_sudo
        )
        if self._stopping:
            return
        if self._bad_password or rc == _SUDO_BAD_PASSWORD_CODE:
            self._bad_password = True
            self._fail("incorrect sudo password")
        elif self._needs_sudo or rc == _SUDO_REQUIRED_CODE:
            self._needs_sudo = True
            self._fail("PTP needs a sudo password")
        else:
            self._fail(
                f"host PTP daemon (ptp4l/phc2sys) exited with code {rc}; clocks "
                "are not synced — see the clock-sync log"
            )

    async def _box_slave(self, box_host: str, sudo_password: str | None) -> _Remote:
        url = f"{self.box}/api/zed/sync-clocks"
        payload: dict[str, Any] = {"role": "slave", "ip": box_host}
        if sudo_password:
            payload["password"] = sudo_password
        try:
            result = await asyncio.to_thread(_post_json, url, payload)
        except urllib.error.HTTPError as exc:
            raise OrchestrationError(
                f"slave-sync: ZED box returned HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OrchestrationError(
                f"slave-sync: cannot reach ZED box at {self.box} ({exc})"
            ) from exc
        if result.get("error"):
            raise OrchestrationError(f"slave-sync: {result['error']}")
        session_id = result.get("id")
        if not session_id:
            raise OrchestrationError("slave-sync: ZED box did not return a session id")
        return _Remote(self.box, "slave-sync", session_id)

    async def _box_stop(self, remote: _Remote) -> None:
        try:
            await asyncio.to_thread(
                _post_json, f"{remote.base}/api/sessions/{remote.id}/stop", {}
            )
        except (urllib.error.URLError, OSError):
            pass

    async def _tail(self, remote: _Remote) -> None:
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
                if self.session is not None:
                    self.session.emit(f"[slave-sync] {line}")
                self._watch(line)
                self._watch_sudo(line)
            remote.offset = data.get("nextOffset", remote.offset)
            if data.get("status") in ("exited", "error") and not data.get("lines"):
                if not self._stopping:
                    if self._bad_password:
                        self._fail("incorrect sudo password")
                    elif self._needs_sudo:
                        self._fail("PTP needs a sudo password")
                    else:
                        self._fail(
                            "PTP daemon on the ZED box exited; clocks are not "
                            "synced — see the clock-sync log"
                        )
                return
            await asyncio.sleep(0.5)

    def _watch_sudo(self, line: str) -> None:
        if _SUDO_BAD_PASSWORD_MARKER in line:
            self._bad_password = True
        elif _SUDO_REQUIRED_MARKER in line:
            self._needs_sudo = True

    def _watch(self, line: str) -> None:
        m = _OFFSET_RE.search(line)
        if m is None:
            return
        try:
            offset = int(m.group(1))
        except ValueError:
            return
        self._offset_ns = offset
        if abs(offset) <= _PTP_LOCK_NS:
            self._ptp_samples += 1
            if self._ptp_samples >= _PTP_LOCK_SAMPLES:
                self._locked.set()
        else:
            self._ptp_samples = 0


class ZedOrchestrator:
    def __init__(
        self,
        session: Session,
        command_id: str,
        args: dict[str, Any],
        zed: dict[str, Any],
        run_main: Any = None,
        ptp: "PtpLink | None" = None,
    ) -> None:
        self.session = session
        self.command_id = command_id
        self.args = dict(args)
        self.zed = zed
        # When set, step 6 awaits this coroutine factory (returning an exit
        # code) instead of spawning the command as a subprocess — used by the
        # in-process operation runner so the ZED pipeline wraps an in-process
        # collect-data / run-policy task.
        self._run_main = run_main
        # Long-lived PTP link established when the box was connected. When it's
        # already up for this box, steps 1–3 are skipped and we just wait on its
        # lock; the link is left running afterwards (owned by the connection).
        self._ptp = ptp
        self.box: str = _normalize_url(str(zed.get("boxUrl", "")))

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
        cameras = self._cameras()
        if not cameras:
            raise OrchestrationError("at least one ZED camera serial is required")

        self.session.status = "starting"

        # Steps 1–3: PTP clock sync. Prefer the link the operator already
        # brought up when connecting the box; only start our own if there isn't
        # a matching one running.
        if self._ptp is not None and self._ptp.running and self._ptp.box == self.box:
            await self._reuse_ptp()
        else:
            await self._start_ptp()

        # 4. Start camera streaming on the ZED box.
        self.session.emit("[serve] step 4/6 — starting ZED camera streams (box)")
        stream_args: dict[str, Any] = dict(cameras)
        for opt in ("resolution", "fps", "bitrate"):
            if self.zed.get(opt) not in (None, ""):
                stream_args[opt] = self.zed[opt]
        stream = await self._box_run("/api/zed/stream", stream_args, label="stream")
        self._tail_remote(stream)

        # 5. Wait for the stream ports to accept connections.
        self.session.emit("[serve] step 5/6 — waiting for camera streams…")
        await self._await_streams(cameras)
        self.session.emit("[serve] camera streams up")

        # 6. Run the main command — in-process (runner) or as a subprocess.
        if self._stopping:
            return
        self.session.emit(f"[serve] step 6/6 — starting {self.command_id}")
        if self._run_main is not None:
            self.session.status = "running"
            rc = await self._run_main(self._main_args())
            self.session.exit_code = rc
            self.session.emit(f"[serve] {self.command_id} finished with code {rc}")
            await self.stop()
            return
        argv = build_argv(self.command_id, self._main_args())
        self._main_proc = await spawn_proc([self.command_id, *argv])
        self.session.proc = self._main_proc
        self.session.status = "running"
        self.session.emit(f"[serve] $ axol {self.command_id} {' '.join(argv)}".rstrip())
        rc = await pump_into(self._main_proc, self.session)
        self.session.exit_code = rc
        self.session.emit(f"[serve] {self.command_id} exited with code {rc}")
        await self.stop()

    async def _start_ptp(self) -> None:
        """Bring PTP up ourselves (no usable connection-time link)."""
        # PTP binds to a NIC, so each side needs an interface name. Derive it
        # from the ZED box address rather than asking the operator: this host
        # uses whichever interface routes to it; the ZED box self-detects the
        # NIC that owns it (see /api/zed/sync-clocks).
        box_host = self._box_host()
        host_iface = iface_for_route(box_host) or best_eth_iface()

        # 1. Master clock sync on this host.
        self.session.emit("[serve] step 1/6 — PTP master clock sync (host)")
        if not host_iface:
            raise OrchestrationError(
                f"could not determine the host network interface for the ZED "
                f"link (no route to {box_host!r} and no wired NIC found)"
            )
        self.session.emit(f"[serve] host ZED-link interface: {host_iface}")
        self._master_proc = await spawn_proc(
            ["zed.sync-clocks", "--role", "master", "--iface", host_iface]
        )
        self._spawn_task(
            pump_into(self._master_proc, self.session, prefix="master-sync")
        )

        # 2. Slave clock sync on the ZED box. We pass the box IP so it can
        # resolve the NIC that owns it; the box falls back to its best wired
        # interface if the lookup fails.
        self.session.emit("[serve] step 2/6 — PTP slave clock sync (ZED box)")
        slave = await self._box_run(
            "/api/zed/sync-clocks",
            {"role": "slave", "ip": box_host},
            label="slave-sync",
        )
        self._tail_remote(slave, on_line=self._watch_ptp)

        # 3. Wait for PTP lock.
        self.session.emit("[serve] step 3/6 — waiting for clocks to lock…")
        await self._await(self._ptp_locked, _PTP_TIMEOUT_S, "clocks did not lock")
        self.session.emit("[serve] clocks locked")

    async def _reuse_ptp(self) -> None:
        """Wait on the PTP link the operator started when connecting the box.

        The link lives on the server event loop while this orchestrator runs on
        the runner's own loop, so we can't await its lock event across loops —
        poll the plain ``locked`` flag instead. The link is left running on
        completion (it's torn down on disconnect, not at task end).
        """
        assert self._ptp is not None
        self.session.emit(
            "[serve] steps 1–3/6 — reusing the PTP clock sync from the ZED box "
            "connection"
        )
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _PTP_TIMEOUT_S
        while not self._ptp.locked:
            if self._stopping:
                raise OrchestrationError("stopped before clocks locked")
            if not self._ptp.running:
                raise OrchestrationError(
                    "the ZED PTP link dropped before clocks locked"
                )
            if loop.time() > deadline:
                raise OrchestrationError("clocks did not lock")
            await asyncio.sleep(0.5)
        self.session.emit("[serve] clocks locked")

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
        """Form args augmented with the ZED stream host.

        ``robot_config.zed_host`` is required by the main command. The cameras
        stream from the same ZED box address the operator connected to, so we
        reuse it. Networking between this host and the box is the operator's
        responsibility.
        """
        args = dict(self.args)
        args.setdefault("robot_config.zed_host", self._box_host())
        return args

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
        host = self._box_host()
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
