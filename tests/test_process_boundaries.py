from __future__ import annotations

import builtins
import io
import logging
import multiprocessing
from types import SimpleNamespace

import pytest

from almond_axol.utils import affinity, jetson_diag, proc_diag
from almond_axol.zed import devices, snapshot


class _Connection:
    def __init__(self, message=None, *, ready: bool = True, eof: bool = False) -> None:
        self.message = message
        self.ready = ready
        self.eof = eof
        self.closed = False
        self.sent: list[object] = []

    def poll(self, timeout: float) -> bool:
        return self.ready

    def recv(self):
        if self.eof:
            raise EOFError
        return self.message

    def send(self, value) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, *, alive: bool = False) -> None:
        self.alive = alive
        self.started = False
        self.terminated = False

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float) -> None:
        return None

    def is_alive(self) -> bool:
        return self.alive and not self.terminated

    def terminate(self) -> None:
        self.terminated = True


class _Context:
    def __init__(self, parent: _Connection, *, alive: bool = False) -> None:
        self.parent = parent
        self.child = _Connection()
        self.proc = _Process(alive=alive)

    def Pipe(self):
        return self.parent, self.child

    def Process(self, **kwargs):
        self.process_kwargs = kwargs
        return self.proc


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (("ok", [{"serial": 2, "model": "ZED", "kind": "stereo"}]), 2),
        (("err", "ImportError", "pyzed missing"), ImportError),
        (("err", "SDKError", "daemon down"), RuntimeError),
    ],
)
def test_zed_enumeration_process_messages(monkeypatch, message, expected) -> None:
    context = _Context(_Connection(message), alive=True)
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    if isinstance(expected, int):
        assert devices.list_zed_devices()[0]["serial"] == expected
    else:
        with pytest.raises(expected):
            devices.list_zed_devices()
    assert context.proc.started
    assert context.proc.terminated
    assert context.parent.closed and context.child.closed


def test_zed_enumeration_timeout_and_crash(monkeypatch) -> None:
    context = _Context(_Connection(ready=False))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    with pytest.raises(TimeoutError, match="did not respond"):
        devices.list_zed_devices(timeout_s=0.1)

    context = _Context(_Connection(ready=True, eof=True))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    with pytest.raises(RuntimeError, match="without a result"):
        devices.list_zed_devices()


def test_zed_worker_and_stereo_serials(monkeypatch) -> None:
    conn = _Connection()
    monkeypatch.setattr(
        devices,
        "list_zed_devices_inproc",
        lambda: [{"serial": 9, "model": "X", "kind": "stereo"}],
    )
    devices._enumerate_worker(conn)  # type: ignore[arg-type]
    assert conn.sent[0][0] == "ok"
    assert conn.closed

    monkeypatch.setattr(
        devices,
        "list_zed_devices",
        lambda: [
            {"serial": 1, "model": "One", "kind": "mono"},
            {"serial": 2, "model": "X", "kind": "stereo"},
        ],
    )
    assert devices.stereo_serials() == {2}
    monkeypatch.setattr(
        devices,
        "list_zed_devices",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert devices.stereo_serials() == set()
    monkeypatch.setattr(
        devices,
        "list_zed_devices",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert devices.stereo_serials() == set()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (("ok", b"jpeg"), b"jpeg"),
        (("err", "ImportError", "cv2 missing"), ImportError),
        (("err", "KeyError", "unknown camera"), KeyError),
        (("err", "ConnectionError", "open failed"), RuntimeError),
    ],
)
def test_snapshot_process_messages(monkeypatch, message, expected) -> None:
    context = _Context(_Connection(message))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    if isinstance(expected, bytes):
        assert snapshot.snapshot_jpeg(123) == expected
    else:
        with pytest.raises(expected):
            snapshot.snapshot_jpeg(123)


def test_snapshot_timeout_crash_and_worker(monkeypatch) -> None:
    context = _Context(_Connection(ready=False), alive=True)
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    with pytest.raises(TimeoutError, match="preview did not complete"):
        snapshot.snapshot_jpeg(4, timeout_s=0.1)
    assert context.proc.terminated

    context = _Context(_Connection(eof=True))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    with pytest.raises(RuntimeError, match="without a result"):
        snapshot.snapshot_jpeg(4)

    conn = _Connection()
    monkeypatch.setattr(snapshot, "snapshot_jpeg_inproc", lambda serial: b"image")
    snapshot._snapshot_worker(conn, 4)  # type: ignore[arg-type]
    assert conn.sent == [("ok", b"image")]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (2, None),
        (
            4,
            {
                "realtime": {0, 1},
                "ik": {0, 1},
                "relay": {2, 3},
                "background": {2, 3},
            },
        ),
        (
            6,
            {
                "realtime": {0, 1},
                "ik": {0, 1},
                "relay": {2, 3},
                "background": {4, 5},
            },
        ),
        (
            8,
            {
                "realtime": {0, 1},
                "ik": {2},
                "relay": {3, 4},
                "background": {5, 6, 7},
            },
        ),
    ],
)
def test_affinity_core_groups(monkeypatch, count, expected) -> None:
    monkeypatch.setattr(affinity.os, "cpu_count", lambda: count)
    assert affinity.core_groups() == expected


def test_affinity_pinners_and_relay_isolation(monkeypatch) -> None:
    monkeypatch.setattr(affinity.os, "cpu_count", lambda: 8)
    calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", lambda pid, cores: calls.append((pid, cores))
    )
    assert affinity.pin_realtime()
    assert affinity.pin_ik()
    assert affinity.pin_ik_startup()
    assert affinity.pin_relay()
    assert affinity.pin_background()
    assert [cores for _, cores in calls] == [
        {0, 1},
        {2},
        {0, 1, 2},
        {3, 4},
        {5, 6, 7},
    ]

    monkeypatch.setattr(
        affinity.os,
        "sched_setaffinity",
        lambda pid, cores: calls.append((pid, cores)),
    )
    monkeypatch.setattr(
        affinity.os,
        "listdir",
        lambda path: ["10", "20", "not-a-thread"],
    )
    import threading

    monkeypatch.setattr(
        threading,
        "enumerate",
        lambda: [SimpleNamespace(native_id=10), SimpleNamespace(native_id=None)],
    )
    assert affinity.isolate_relay_cpu()
    assert calls[-2:] == [(10, {3}), (20, {4})]

    def denied(pid, cores):
        raise OSError("denied")

    monkeypatch.setattr(affinity.os, "sched_setaffinity", denied)
    assert not affinity.pin_realtime()
    assert not affinity.isolate_relay_cpu()


def test_proc_readers(monkeypatch) -> None:
    files = {
        "/proc/stat": "cpu 1 2 3 4\ncpu0 10 2 3 20 5\ncpu1 4 1 1 4\nintr 2\n",
        "/proc/12/stat": "12 (worker name) S " + " ".join(["0"] * 10 + ["7", "8"]),
        "/proc/12/statm": "100 3",
        "/proc/12/task/12/children": "13 14",
        "/proc/meminfo": "MemAvailable: 1024 kB\nSwapFree: 10 kB\nSwapTotal: 30 kB\n",
    }

    def fake_open(path, *args, **kwargs):
        if str(path) not in files:
            raise OSError("missing")
        return io.StringIO(files[str(path)])

    monkeypatch.setattr(builtins, "open", fake_open)
    assert proc_diag.read_percpu() == {"cpu0": (15, 40), "cpu1": (6, 10)}
    assert proc_diag.read_proc_cpu(12) == (15, "worker name")
    assert proc_diag.read_proc_rss(12) == 3 * proc_diag.PAGE_SIZE
    assert proc_diag.read_children(12) == [13, 14]
    assert proc_diag.read_meminfo() == (1024**2, 10 * 1024, 30 * 1024)
    assert proc_diag._gib(2 * 1024**3) == "2.0G"
    assert proc_diag.read_proc_cpu(99) is None
    assert proc_diag.read_proc_rss(99) == 0
    assert proc_diag.read_children(99) == []


def test_system_diag_scans_labeled_processes(monkeypatch) -> None:
    diag = proc_diag.SystemDiag({10: "main"}, logging.getLogger("test"))
    monkeypatch.setattr(proc_diag, "read_proc_cpu", lambda pid: (pid, f"p{pid}"))
    monkeypatch.setattr(proc_diag, "read_children", lambda pid: [11])
    assert diag._scan_labeled() == {10: (10, "main"), 11: (11, "main-p11")}
    assert diag._label(10, "python") == "main"
    assert diag._label(20, "other") == "other"
    diag.stop()
    assert diag._stop.is_set()


def test_tegrastats_parser_and_unavailable_run(monkeypatch, caplog) -> None:
    logger = logging.getLogger("tegra-test")
    caplog.set_level(logging.DEBUG, logger="tegra-test")
    diag = jetson_diag.TegraStatsDiag(logger)
    diag._log_line(
        "RAM 2048/8192MB SWAP 10/100MB CPU [10%@1000,20%@1500] "
        "EMC_FREQ 30% GR3D_FREQ 40% NVENC 600 cpu@50C gpu@61.5C throttling"
    )
    assert "gr3d=40%" in caplog.text
    assert "cpufreq=1000-1500MHz" in caplog.text
    assert "tmax=61.5C" in caplog.text
    assert "throttle=yes" in caplog.text

    monkeypatch.setattr(jetson_diag.jetson, "_is_jetson", lambda: False)
    monkeypatch.setattr(jetson_diag.shutil, "which", lambda name: None)
    diag.run()
    assert "diag disabled" in caplog.text
