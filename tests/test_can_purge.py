"""Clearing the CAN TX queue an e-stop poisons.

Motor power dies mid-command, nothing ACKs, and the kernel parks up to
``txqueuelen`` position commands on the interface. They outlive the session
and replay when the motors come back, so they must be purged — by the
realtime core when it declares the stall, and by the next bring-up if that
purge never happened. Covered here: the ``sudoers.d`` grant that lets a
manual (non-root) session do the flap at all, and the bring-up side's
detection and refusal.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.can import setup as can_setup
from almond_axol.constants import CAN_BRINGUP_SCRIPT, CAN_LEFT, CAN_RIGHT
from almond_axol.utils import can_purge


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


class PurgeGrantTest(unittest.TestCase):
    """The ``sudoers.d`` drop-in ``axol provision`` installs."""

    def test_grants_the_exact_commands_the_core_runs(self) -> None:
        # safety.rs runs the provisioned bring-up script, falling back to a
        # per-interface flap. sudoers matches resolved absolute paths, so a
        # rule that named anything else would simply never match.
        commands = can_purge.purge_commands()
        self.assertTrue(
            any(command.endswith(f"bash {CAN_BRINGUP_SCRIPT}") for command in commands),
            commands,
        )
        for iface in (CAN_LEFT, CAN_RIGHT):
            for direction in ("down", "up"):
                self.assertTrue(
                    any(
                        command.endswith(f"ip link set {iface} {direction}")
                        for command in commands
                    ),
                    commands,
                )
        for command in commands:
            self.assertTrue(command.startswith("/"), command)

    def test_grant_is_scoped_to_the_operator_and_needs_no_password(self) -> None:
        text = can_purge.sudoers_text("shawn")
        self.assertIn("shawn ALL=(root) NOPASSWD: AXOL_CAN_PURGE\n", text)
        # Nothing open-ended: no shell, no wildcard, no blanket ALL command.
        self.assertNotIn("NOPASSWD: ALL", text)
        self.assertNotIn("*", text)

    def test_generated_rule_parses(self) -> None:
        # A malformed file in sudoers.d breaks sudo host-wide, so the bytes
        # are validated before they are ever published.
        with tempfile.TemporaryDirectory() as scratch:
            staged = Path(scratch) / "50-axol-can-purge"
            staged.write_text(can_purge.sudoers_text("shawn"))
            if not Path("/usr/sbin/visudo").exists():
                self.skipTest("visudo is not installed")
            self.assertTrue(can_purge._is_valid_sudoers(staged))

    def test_refuses_to_install_what_visudo_rejects(self) -> None:
        with (
            patch.object(can_purge, "operator_user", return_value="shawn"),
            patch.object(can_purge, "_is_valid_sudoers", return_value=False),
            patch.object(can_purge, "run_root", side_effect=AssertionError),
            self.assertRaises(RuntimeError),
        ):
            can_purge.install()

    def test_installs_root_owned_and_read_only(self) -> None:
        runs: list[list[str]] = []

        def fake_run_root(argv, *, input_text=None, check=False):
            runs.append(argv)
            # Nothing installed yet: `cmp` reports a difference.
            return _completed(1 if argv[0] == "cmp" else 0)

        with (
            patch.object(can_purge, "operator_user", return_value="shawn"),
            patch.object(can_purge, "_is_valid_sudoers", return_value=True),
            patch.object(can_purge, "prime_sudo", return_value=True),
            patch.object(can_purge.os, "geteuid", return_value=1000),
            patch.object(can_purge, "run_root", fake_run_root),
        ):
            can_purge.install()

        install_call = next(
            argv for argv in runs if argv[0] == "install" and argv[-1].endswith("purge")
        )
        self.assertIn("0440", install_call)
        self.assertEqual(install_call[-1], str(can_purge.SUDOERS_PATH))
        self.assertIn("root", install_call)

    def test_rerun_with_the_rule_in_place_reinstalls_nothing(self) -> None:
        def fake_run_root(argv, *, input_text=None, check=False):
            if argv[0] == "cmp":
                return _completed(0)
            raise AssertionError(f"unexpected privileged call: {argv}")

        with (
            patch.object(can_purge, "operator_user", return_value="shawn"),
            patch.object(can_purge, "_is_valid_sudoers", return_value=True),
            patch.object(can_purge, "prime_sudo", return_value=True),
            patch.object(can_purge.os, "geteuid", return_value=1000),
            patch.object(can_purge, "run_root", fake_run_root),
        ):
            can_purge.install()

    def test_without_sudo_it_warns_instead_of_failing_provisioning(self) -> None:
        # Every other provisioning grant degrades to a warning; this one must
        # too, or a host that cannot escalate fails the whole command.
        with (
            patch.object(can_purge, "operator_user", return_value="shawn"),
            patch.object(can_purge, "_is_valid_sudoers", return_value=True),
            patch.object(can_purge, "prime_sudo", return_value=False),
            patch.object(can_purge.os, "geteuid", return_value=1000),
            patch.object(can_purge, "run_root", side_effect=AssertionError),
            self.assertLogs(can_purge._logger, "WARNING") as logged,
        ):
            can_purge.install()
        self.assertIn("e-stop", "\n".join(logged.output))


class BacklogDetectionTest(unittest.TestCase):
    """Reading the queue depth `tc` reports for an interface."""

    def test_reads_the_queued_frame_count(self) -> None:
        shown = (
            "qdisc pfifo_fast 0: root refcnt 2 bands 3 priomap 1 2 2 2\n"
            " Sent 72607376 bytes 4537961 pkt (dropped 0, overlimits 0 requeues 3)\n"
            " backlog 4096b 512p requeues 3\n"
        )
        with patch.object(
            can_setup.subprocess, "run", return_value=_completed(0, shown)
        ):
            self.assertEqual(can_setup.tx_backlog(CAN_LEFT), 512)

    def test_idle_interface_reports_an_empty_queue(self) -> None:
        shown = " backlog 0b 0p requeues 3\n"
        with patch.object(
            can_setup.subprocess, "run", return_value=_completed(0, shown)
        ):
            self.assertEqual(can_setup.tx_backlog(CAN_LEFT), 0)

    def test_unknown_when_tc_is_missing_or_fails(self) -> None:
        with patch.object(can_setup.subprocess, "run", side_effect=OSError):
            self.assertIsNone(can_setup.tx_backlog(CAN_LEFT))
        with patch.object(can_setup.subprocess, "run", return_value=_completed(1, "")):
            self.assertIsNone(can_setup.tx_backlog(CAN_LEFT))


class PurgeStaleTxTest(unittest.TestCase):
    """What a bring-up does about frames the last session left queued."""

    def test_clean_queues_flap_nothing(self) -> None:
        with (
            patch.object(can_setup, "_iface_present", return_value=True),
            patch.object(can_setup, "tx_backlog", return_value=0),
            patch.object(can_setup, "bring_up_interfaces", side_effect=AssertionError),
        ):
            self.assertEqual(can_setup.purge_stale_tx([CAN_LEFT, CAN_RIGHT]), [])

    def test_unreadable_backlog_is_not_treated_as_poisoned(self) -> None:
        # A host without `tc`, or a bench interface the qdisc query fails on,
        # must not start flapping buses on every connect.
        with (
            patch.object(can_setup, "_iface_present", return_value=True),
            patch.object(can_setup, "tx_backlog", return_value=None),
            patch.object(can_setup, "bring_up_interfaces", side_effect=AssertionError),
        ):
            self.assertEqual(can_setup.purge_stale_tx([CAN_LEFT]), [])

    def test_absent_interfaces_are_skipped(self) -> None:
        with (
            patch.object(can_setup, "_iface_present", return_value=False),
            patch.object(can_setup, "tx_backlog", side_effect=AssertionError),
            patch.object(can_setup, "bring_up_interfaces", side_effect=AssertionError),
        ):
            self.assertEqual(can_setup.purge_stale_tx(["can-bench"]), [])

    def test_queued_frames_flap_the_whole_group_together(self) -> None:
        # One poisoned channel still cycles both: the arm interfaces are two
        # halves of one dual-channel adapter, and flapping them one at a time
        # wedges its RX path (see the generated bring-up script).
        flapped: list[tuple[list[str], bool]] = []
        backlogs = iter([64, 0, 0, 0])

        with (
            patch.object(can_setup, "_iface_present", return_value=True),
            patch.object(
                can_setup, "tx_backlog", side_effect=lambda _ch: next(backlogs)
            ),
            patch.object(
                can_setup,
                "bring_up_interfaces",
                side_effect=lambda chans, *, force_cycle=False: flapped.append(
                    (chans, force_cycle)
                ),
            ),
        ):
            purged = can_setup.purge_stale_tx([CAN_LEFT, CAN_RIGHT])

        self.assertEqual(purged, [CAN_LEFT])
        self.assertEqual(flapped, [([CAN_LEFT, CAN_RIGHT], True)])

    def test_a_queue_that_survives_the_flap_raises(self) -> None:
        # Enabling motors into a queue that still holds stale commands is the
        # jerk this whole path exists to prevent, so it must fail loudly.
        with (
            patch.object(can_setup, "_iface_present", return_value=True),
            patch.object(can_setup, "tx_backlog", return_value=32),
            patch.object(can_setup, "bring_up_interfaces", return_value=None),
            self.assertRaises(RuntimeError) as raised,
        ):
            can_setup.purge_stale_tx([CAN_LEFT])
        self.assertIn("refusing to enable motors", str(raised.exception))


class ConnectPurgesTest(unittest.IsolatedAsyncioTestCase):
    """``AxolHardware.connect()`` clears the queue before any bus opens."""

    def _axol(self):
        from almond_axol.robot.axol import AxolHardware

        opened: list[str] = []

        class _Bus:
            def __init__(self, channel: str) -> None:
                self.channel = channel

            async def start(self) -> None:
                opened.append(self.channel)

            def _add_listener(self, _listener) -> None:
                pass

        with (
            patch("almond_axol.robot.axol.CanBus", side_effect=_Bus),
            patch(
                "almond_axol.motor.motor.make_driver",
                side_effect=lambda *_a, **_k: SimpleNamespace(
                    kp_max=500.0,
                    kd_max=5.0,
                    set_feedback_callback=lambda _cb: None,
                ),
            ),
        ):
            axol = AxolHardware(left_channel=CAN_LEFT, right_channel=CAN_RIGHT)
        return axol, opened

    async def test_purge_runs_before_the_buses_open(self) -> None:
        axol, opened = self._axol()
        seen: list[list[str]] = []

        def fake_purge(channels: list[str]) -> list[str]:
            self.assertEqual(opened, [], "a bus opened before the purge")
            seen.append(channels)
            return []

        with patch.object(can_setup, "purge_stale_tx", fake_purge):
            await axol.connect()

        self.assertEqual(seen, [[CAN_LEFT, CAN_RIGHT]])
        self.assertEqual(sorted(opened), sorted([CAN_LEFT, CAN_RIGHT]))

    async def test_an_unpurgeable_queue_refuses_the_connection(self) -> None:
        from almond_axol.motor import MotorError

        axol, opened = self._axol()
        with (
            patch.object(
                can_setup,
                "purge_stale_tx",
                side_effect=RuntimeError("still hold queued frames"),
            ),
            self.assertRaises(MotorError) as raised,
        ):
            await axol.connect()
        self.assertIn("still hold queued frames", str(raised.exception))
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
