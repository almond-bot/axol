from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from almond_axol.utils import packages


@pytest.fixture
def systemd(monkeypatch):
    monkeypatch.setattr(packages, "_systemd_booted", lambda: True)
    monkeypatch.setattr(
        packages.shutil,
        "which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None,
    )


def test_root_apt_runs_in_its_own_scope_waiting_for_the_lock(
    monkeypatch, systemd
) -> None:
    monkeypatch.setattr(packages.os, "geteuid", lambda: 0)
    assert packages.package_manager_command(["apt-get", "install", "-y", "cmake"]) == [
        "/usr/bin/systemd-run",
        "--scope",
        "--quiet",
        "--collect",
        "--description=Axol package install",
        "env",
        "DEBIAN_FRONTEND=noninteractive",
        "apt-get",
        "-o",
        f"DPkg::Lock::Timeout={packages.DPKG_LOCK_WAIT_S}",
        "install",
        "-y",
        "cmake",
    ]


def test_operator_dpkg_escalates_without_apt_options(monkeypatch) -> None:
    monkeypatch.setattr(packages.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(packages, "_systemd_booted", lambda: False)
    assert packages.package_manager_command(["dpkg", "-i", "/tmp/driver.deb"]) == [
        "sudo",
        "env",
        "DEBIAN_FRONTEND=noninteractive",
        "dpkg",
        "-i",
        "/tmp/driver.deb",
    ]


@pytest.mark.parametrize(("euid", "own_session"), [(0, True), (1000, False)])
def test_run_never_times_out_and_detaches_only_as_root(
    monkeypatch, euid: int, own_session: bool
) -> None:
    monkeypatch.setattr(packages.os, "geteuid", lambda: euid)
    monkeypatch.setattr(packages, "_systemd_booted", lambda: False)
    done = subprocess.CompletedProcess([], 0, "", "")
    with patch.object(packages.subprocess, "run", return_value=done) as run:
        assert packages.run_package_manager(["apt-get", "update"]) is done
    kwargs = run.call_args.kwargs
    assert "timeout" not in kwargs
    assert kwargs["start_new_session"] is own_session
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_failure_names_the_package_dpkg_could_not_process(monkeypatch) -> None:
    monkeypatch.setattr(packages.os, "geteuid", lambda: 0)
    monkeypatch.setattr(packages, "_systemd_booted", lambda: False)
    failed = subprocess.CompletedProcess(
        [],
        100,
        "Setting up nvidia-l4t-kernel (36.4.4) ...\n",
        "dpkg: error processing package nvidia-l4t-kernel (--configure):\n"
        "Errors were encountered while processing:\n"
        " nvidia-l4t-kernel\n"
        " nvidia-l4t-kernel-headers\n"
        "E: Sub-process /usr/bin/dpkg returned an error code (1)\n",
    )
    with (
        patch.object(packages.subprocess, "run", return_value=failed),
        pytest.raises(RuntimeError) as raised,
    ):
        packages.run_package_manager(["apt-get", "install", "-y", "cmake"], check=True)
    message = str(raised.value)
    assert "nvidia-l4t-kernel, nvidia-l4t-kernel-headers" in message
    assert "sudo dpkg --configure -a" in message


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "E: dpkg was interrupted, you must manually run "
            "'sudo dpkg --configure -a' to correct the problem.",
            "an earlier package install was interrupted",
        ),
        (
            "E: Sub-process /usr/bin/dpkg returned an error code (1)",
            "dpkg failed while configuring packages",
        ),
        ("E: Unable to locate package nope\n", "E: Unable to locate package nope"),
        ("", "exit code 100"),
    ],
)
def test_failure_detail_falls_back_to_something_actionable(
    output: str, expected: str
) -> None:
    detail = packages.failure_detail(subprocess.CompletedProcess([], 100, "", output))
    assert detail.startswith(expected)
