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


@pytest.mark.parametrize(
    ("nodes", "pinned"),
    [
        # Orin (Orin NX 16GB, AGX Orin; nvgpu): one node per engine, each
        # pinned min_freq = max_freq.
        (
            ("15480000.nvenc", "15340000.vic", "17000000.gpu"),
            {
                ("15480000.nvenc", "min_freq", "900"),
                ("15340000.vic", "min_freq", "900"),
                ("17000000.gpu", "min_freq", "900"),
            },
        ),
        # Thor T5000 (OpenRM): NVENC is clocked by the GPU's multimedia (NVD)
        # domain and CUDA by GPC; only the VIC keeps its own node. The old
        # globs pinned the VIC alone and left encode + CUDA at the floor. The
        # GPU domains get the performance governor and never a min_freq
        # access (see jetson._GOVERNOR_CLOCK_GLOBS).
        (
            ("8188050000.vic", "gpu-gpc-0", "gpu-nvd-0"),
            {
                ("8188050000.vic", "min_freq", "900"),
                ("gpu-gpc-0", "governor", "performance"),
                ("gpu-nvd-0", "governor", "performance"),
            },
        ),
    ],
    ids=["orin", "thor"],
)
def test_engine_pins_cover_every_supported_jetson(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nodes: tuple[str, ...],
    pinned: set[tuple[str, str, str]],
) -> None:
    devfreq = tmp_path / "devfreq"
    for name in (*nodes, "unrelated-cpu-bw"):
        node = devfreq / name
        node.mkdir(parents=True)
        (node / "max_freq").write_text("900\n")
        (node / "min_freq").write_text("300\n")
        (node / "governor").write_text("nvhost_podgov\n")
        (node / "available_governors").write_text(
            "nvhost_podgov performance userspace\n"
        )

    original_glob = Path.glob

    def glob(path: Path, pattern: str):
        if str(path) == "/sys/class/devfreq":
            return original_glob(devfreq, pattern)
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", glob)
    writer = _Writer()
    jetson._pin_engines(writer)
    assert {(p.parent.name, p.name, v) for p, v in writer.writes} == pinned


def test_thor_gpu_domains_never_touch_their_frequency_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reading min_freq / max_freq takes the devfreq lock and cur_freq calls
    # into the GPU driver; a JetPack 7.2 report has min_freq reads stuck in D
    # state behind a deadlocked devfreq_wq worker. Already-pinned domains are
    # recognised from `governor` alone, and one without the performance
    # governor is reported rather than min_freq-pinned.
    devfreq = tmp_path / "devfreq"
    for name, governor, available in (
        ("gpu-gpc-0", "performance", "performance nvhost_podgov"),
        ("gpu-nvd-0", "nvhost_podgov", "nvhost_podgov userspace"),
    ):
        node = devfreq / name
        node.mkdir(parents=True)
        (node / "governor").write_text(governor + "\n")
        (node / "available_governors").write_text(available + "\n")
    original_glob = Path.glob
    original_read = Path.read_text

    def glob(path: Path, pattern: str):
        if str(path) == "/sys/class/devfreq":
            return original_glob(devfreq, pattern)
        return original_glob(path, pattern)

    def read_text(path: Path, *args, **kwargs):
        assert path.name not in {"min_freq", "max_freq", "cur_freq"}, path
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "read_text", read_text)
    writer = _Writer()
    jetson._pin_engines(writer)
    assert writer.writes == []


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        # AGX Orin 64GB: MAXN plus the fixed-TDP modes.
        (
            "< POWER_MODEL ID=0 NAME=MAXN >\n< POWER_MODEL ID=1 NAME=MODE_15W >\n"
            "< POWER_MODEL ID=2 NAME=MODE_30W >\n< POWER_MODEL ID=3 NAME=MODE_50W >\n",
            ("0", "MAXN"),
        ),
        # Thor T5000 (L4T R38): MAXN is mode 0, the 120 W default is mode 1.
        (
            "< POWER_MODEL ID=0 NAME=MAXN >\n< POWER_MODEL ID=1 NAME=120W >\n"
            "< POWER_MODEL ID=2 NAME=90W >\n< POWER_MODEL ID=3 NAME=70W >\n",
            ("0", "MAXN"),
        ),
    ],
    ids=["agx-orin", "thor-t5000"],
)
def test_max_power_mode_on_agx_orin_and_thor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: str,
    expected: tuple[str, str],
) -> None:
    conf = tmp_path / "nvpmodel.conf"
    conf.write_text(config)
    monkeypatch.setattr(jetson, "_NVPMODEL_CONFIG", conf)
    assert jetson._preferred_max_power_mode() == expected


def test_host_model_strips_the_device_tree_terminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.write_bytes(b"NVIDIA Jetson AGX Thor Developer Kit\0")
    monkeypatch.setattr(jetson, "_DEVICE_TREE_MODEL", model)
    assert jetson.host_model() == "NVIDIA Jetson AGX Thor Developer Kit"
    monkeypatch.setattr(jetson, "_DEVICE_TREE_MODEL", tmp_path / "absent")
    assert jetson.host_model() is None


def test_pin_check_reads_only_the_governor_on_thor_gpu_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from almond_axol.utils import jetson_diag

    devfreq = tmp_path / "devfreq"
    for name, governor in (("gpu-gpc-0", "performance"), ("gpu-nvd-0", "podgov")):
        (devfreq / name).mkdir(parents=True)
        (devfreq / name / "governor").write_text(governor + "\n")
    original_glob = Path.glob
    original_read = Path.read_text

    def glob(path: Path, pattern: str):
        if str(path) == "/sys/class/devfreq":
            return original_glob(devfreq, pattern)
        if str(path) == "/sys/devices/system/cpu":
            return iter([])
        return original_glob(path, pattern)

    def read_text(path: Path, *args, **kwargs):
        assert path.name not in {"min_freq", "max_freq", "cur_freq"}, path
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "read_text", read_text)
    logger = logging.getLogger("test.tegra")
    warnings: list[str] = []
    monkeypatch.setattr(logger, "warning", lambda msg, *a: warnings.append(msg % a))
    jetson_diag.TegraStatsDiag(logger)._check_pins()
    assert len(warnings) == 1
    assert "gpu-nvd-0" in warnings[0]
