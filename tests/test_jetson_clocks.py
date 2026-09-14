from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from almond_axol.utils import jetson


class _Writer:
    def __init__(self, results: list[tuple[bool, str]] | None = None) -> None:
        self.results = list(results or [])
        self.writes: list[tuple[Path, str]] = []

    def write(self, path: Path, value: str) -> tuple[bool, str]:
        self.writes.append((path, value))
        return self.results.pop(0) if self.results else (True, "")


def _proc(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_root_escalator_prefers_direct_operations(tmp_path: Path) -> None:
    escalator = jetson._RootEscalator(interactive=False)
    target = tmp_path / "setting"
    assert escalator.write(target, "value") == (True, "")
    assert target.read_text() == "value"
    assert escalator.run(["true"]) == (True, "")


def test_root_escalator_primes_once_and_falls_back_to_sudo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primes: list[bool] = []
    monkeypatch.setattr(jetson, "prime_sudo", lambda: primes.append(True) or True)

    class Unwritable:
        def write_text(self, value: str) -> None:
            raise PermissionError("direct denied")

        def __str__(self) -> str:
            return "/sys/setting"

    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _proc(0)

    monkeypatch.setattr(jetson.subprocess, "run", run)
    escalator = jetson._RootEscalator(interactive=True)
    assert escalator.write(Unwritable(), "42") == (True, "")  # type: ignore[arg-type]
    assert escalator.write(Unwritable(), "43") == (True, "")  # type: ignore[arg-type]
    assert primes == [True]
    assert calls[0] == ["sudo", "-n", "tee", "/sys/setting"]


def test_root_escalator_reports_best_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = [
        _proc(2, stdout="direct output"),
        _proc(1, stderr="sudo output"),
    ]
    monkeypatch.setattr(
        jetson.subprocess, "run", lambda *args, **kwargs: outcomes.pop(0)
    )
    escalator = jetson._RootEscalator(interactive=False)
    assert escalator.run(["command"], input_text="n\n") == (False, "sudo output")

    def missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0][0] != "sudo":
            raise FileNotFoundError("missing executable")
        return _proc(1)

    monkeypatch.setattr(jetson.subprocess, "run", missing)
    assert escalator.run(["missing"]) == (False, "missing executable")


def test_power_mode_query_handles_output_and_missing_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        jetson.subprocess,
        "run",
        lambda *args, **kwargs: _proc(0, stdout="NV Power Mode: MAXN\n0\n"),
    )
    assert jetson._query_power_mode("nvpmodel") == "0"
    monkeypatch.setattr(
        jetson.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert jetson._query_power_mode("nvpmodel") is None


def test_max_power_mode_is_gated_and_skips_when_already_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    escalator = SimpleNamespace(
        run=lambda *args, **kwargs: pytest.fail("unexpected run")
    )
    monkeypatch.setattr(jetson, "_is_jetson", lambda: False)
    jetson._set_max_power_mode(escalator)

    monkeypatch.setattr(jetson, "_is_jetson", lambda: True)
    monkeypatch.setattr(jetson.shutil, "which", lambda name: None)
    jetson._set_max_power_mode(escalator)

    monkeypatch.setattr(jetson.shutil, "which", lambda name: "/usr/bin/nvpmodel")
    monkeypatch.setattr(jetson, "_query_power_mode", lambda binary: jetson._MAXN_MODE)
    jetson._set_max_power_mode(escalator)


def test_max_power_mode_switches_or_persists_for_next_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jetson, "_is_jetson", lambda: True)
    monkeypatch.setattr(jetson.shutil, "which", lambda name: "/usr/bin/nvpmodel")
    modes = iter(["2", "0"])
    monkeypatch.setattr(jetson, "_query_power_mode", lambda binary: next(modes))
    calls: list[tuple[list[str], str | None]] = []
    escalator = SimpleNamespace(
        run=lambda argv, input_text=None: calls.append((argv, input_text))
        or (True, ""),
        write=lambda *args: pytest.fail("unexpected write"),
    )
    jetson._set_max_power_mode(escalator)
    assert calls == [(["/usr/bin/nvpmodel", "-m", "0"], "n\n")]

    monkeypatch.setattr(jetson, "_query_power_mode", lambda binary: "2")
    writer = _Writer()
    escalator = SimpleNamespace(
        run=lambda *args, **kwargs: (True, "reboot required"), write=writer.write
    )
    jetson._set_max_power_mode(escalator)
    assert writer.writes == [(jetson._NVPMODEL_STATUS, "pmode:0000")]


def test_engine_and_cpu_pinning_cover_changed_equal_and_unreadable_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "engine.nvenc"
    engine.mkdir()
    (engine / "max_freq").write_text("100\n")
    (engine / "min_freq").write_text("20\n")
    already = tmp_path / "engine.vic"
    already.mkdir()
    (already / "max_freq").write_text("80\n")
    (already / "min_freq").write_text("80\n")
    unreadable = tmp_path / "missing.vic"

    cpu0 = tmp_path / "cpu0"
    cpu1 = tmp_path / "cpu1"
    cpu2 = tmp_path / "cpu2"
    for cpu, governor in ((cpu0, "schedutil"), (cpu1, "performance")):
        (cpu / "cpufreq").mkdir(parents=True)
        (cpu / "cpufreq" / "scaling_governor").write_text(governor)

    original_glob = Path.glob

    def glob(path: Path, pattern: str):
        if str(path) == "/sys/class/devfreq":
            if pattern == "*.nvenc":
                return iter([engine])
            return iter([already, unreadable])
        if str(path) == "/sys/devices/system/cpu":
            return iter([cpu2, cpu1, cpu0])
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", glob)
    writer = _Writer(results=[(True, ""), (False, "read only")])
    jetson._pin_engines(writer)
    assert writer.writes == [(engine / "min_freq", "100")]

    monkeypatch.setattr(jetson, "_is_jetson", lambda: True)
    jetson._pin_cpu(writer)
    assert writer.writes[-1] == (cpu0 / "cpufreq" / "scaling_governor", "performance")

    monkeypatch.setattr(jetson, "_is_jetson", lambda: False)
    before = list(writer.writes)
    jetson._pin_cpu(writer)
    assert writer.writes == before


def test_public_clock_helpers_share_escalator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(
        jetson, "_pin_engines", lambda esc: seen.append(("engine", esc))
    )
    monkeypatch.setattr(jetson, "_pin_cpu", lambda esc: seen.append(("cpu", esc)))
    monkeypatch.setattr(
        jetson, "_set_max_power_mode", lambda esc: seen.append(("mode", esc))
    )

    jetson.pin_engine_clocks(interactive=True)
    jetson.pin_realtime_clocks(interactive=True)
    assert [name for name, _ in seen] == ["engine", "mode", "engine", "cpu"]
    assert seen[1][1] is seen[2][1] is seen[3][1]
