"""``axol teach-poses`` — guided key-pose recording during a VR teleop session.

Deploying taught-pose applications (pick stations, fixtured scenes) needs a
fast way to record a set of named joint poses. Hand-guiding a limp arm works
but is imprecise and slow; this mode instead runs a NORMAL teleop session and
walks the operator through a label list with in-headset banner prompts:

  RIGHT thumbstick CLICK  record the arm's current measured pose
  LEFT  thumbstick CLICK  step back and re-record the previous label

Each recorded pose is saved immediately (atomic ``WaypointSet.save``), so an
interrupted session loses nothing. Prompts are shown in the headset via the
VR server's banner channel (``VRServer.set_banner``) and mirrored to the
terminal. The thumbstick clicks otherwise drive the powered cart's lift, so
this command refuses ``--cart.enabled``.

Example:
  axol teach-poses --labels home pick_hover place_1 place_2 --out poses.json
  axol teach-poses --labels home bin_a bin_b -- --teleop.position_multiplier 1.2
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import threading
import time

import numpy as np

_logger = logging.getLogger(__name__)

_DEBOUNCE_S = 0.6
_SETTLE_S = 0.15


class _Clicks:
    """Rising-edge latch for the two stick clicks; fed from the VR frame
    callback (must stay allocation-light), drained on the guide thread."""

    def __init__(self) -> None:
        self.events: "queue.Queue[str]" = queue.Queue()
        self._prev = (False, False)
        self._last = 0.0

    def feed(self, frame) -> None:
        now = time.monotonic()
        r = bool(getattr(frame, "r_stick_click", False))
        l = bool(getattr(frame, "l_stick_click", False))
        pr, pl = self._prev
        if r and not pr and now - self._last >= _DEBOUNCE_S:
            self._last = now
            self.events.put_nowait("record")
        if l and not pl and now - self._last >= _DEBOUNCE_S:
            self._last = now
            self.events.put_nowait("redo")
        self._prev = (r, l)


def _guide(robot, server, clicks: _Clicks, labels: list[str], out: str,
           done: threading.Event) -> None:
    from ..waypoints import Waypoint, WaypointSet

    try:
        try:
            ws = WaypointSet.load(out)
        except Exception:  # noqa: BLE001 - new file
            ws = WaypointSet(waypoints=[])
        by_label = {w.label: w for w in ws.waypoints}
        recorded: list[str] = []
        i = 0
        while i < len(labels):
            name = labels[i]
            state = "on file" if name in by_label else "new"
            server.set_banner(
                f"TEACH {i + 1}/{len(labels)}: {name}  ({state})  "
                "[R-click record / L-click redo]"
            )
            print(f"\n>>> pose {i + 1}/{len(labels)}: {name} — drive there, "
                  "hold still, RIGHT stick click to record", flush=True)
            ev = None
            while ev is None:
                try:
                    ev = clicks.events.get(timeout=0.5)
                except queue.Empty:
                    if done.is_set():
                        return
            if ev == "redo":
                if recorded:
                    i = labels.index(recorded.pop())
                    print(f">>> redoing {labels[i]}", flush=True)
                continue
            time.sleep(_SETTLE_S)
            left = np.asarray(robot.left.positions, dtype=np.float32).copy()
            right = np.asarray(robot.right.positions, dtype=np.float32).copy()
            wp = Waypoint(left=left, right=right, label=name)
            if name in by_label:
                ws.waypoints[ws.waypoints.index(by_label[name])] = wp
            else:
                ws.waypoints.append(wp)
            by_label[name] = wp
            ws.save(out)
            recorded.append(name)
            server.set_banner(f"RECORDED {name}  ({i + 1}/{len(labels)})")
            print(f">>> recorded {name} -> {out}", flush=True)
            time.sleep(0.8)
            i += 1
        server.set_banner("ALL POSES RECORDED — park the arm, then Ctrl-C")
        print(f"\n>>> all {len(labels)} poses recorded to {out}; park the arm "
              "and end the session with Ctrl-C", flush=True)
    finally:
        done.set()


async def _run(cfg, labels: list[str], out: str) -> None:
    from ..robot import Axol
    from ..teleop import VRTeleop

    robot = Axol(config=cfg.axol, left_channel=cfg.left_channel,
                 right_channel=cfg.right_channel)
    teleop = VRTeleop(robot, config=cfg.teleop, kinematics_config=cfg.kinematics,
                      vr_server_config=cfg.vr_server, cart=None)

    clicks = _Clicks()
    vendor_cb = teleop._on_vr_frame

    def _chained(frame) -> None:
        vendor_cb(frame)
        clicks.feed(frame)

    teleop._vr_server.set_on_frame(_chained)

    done = threading.Event()
    async with teleop:
        # Headset video, same wiring as `axol teleop`.
        from .teleop import (
            _connect_zed_cameras,
            _register_zed_video,
            _start_video_relay,
            _stereo_serials_for,
        )

        relay = None
        cameras = []
        try:
            stereo_set = await asyncio.to_thread(_stereo_serials_for, cfg)
            relay = await asyncio.to_thread(_start_video_relay, cfg, stereo_set)
            if relay is not None:
                teleop.set_video_manager(relay)
            else:
                cameras = await asyncio.to_thread(_connect_zed_cameras, cfg, stereo_set)
                _register_zed_video(teleop, cameras)
        except Exception as exc:  # noqa: BLE001 - video is optional here
            _logger.warning("headset video unavailable (%s)", exc)

        t = threading.Thread(
            target=_guide,
            args=(robot, teleop._vr_server, clicks, labels, out, done),
            daemon=True, name="teach-guide",
        )
        t.start()
        try:
            await teleop.run()
        finally:
            done.set()
            teleop._vr_server.set_banner(None)
            if relay is not None:
                await asyncio.to_thread(relay.shutdown)
            for _name, cam in cameras:
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001
                    pass


def main(argv: list[str]) -> None:
    from .config import TeleopCmdConfig, normalize_bool_flags, parse

    ap = argparse.ArgumentParser(prog="axol teach-poses", description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="ordered pose labels to record")
    ap.add_argument("--out", default="poses.json",
                    help="WaypointSet file to write (default poses.json)")
    args, vendor_argv = ap.parse_known_args(argv)
    if vendor_argv[:1] == ["--"]:
        vendor_argv = vendor_argv[1:]

    cfg = parse(TeleopCmdConfig, normalize_bool_flags(vendor_argv, "sim", "cart_only"))
    if getattr(cfg, "sim", False):
        raise SystemExit("teach-poses records a real arm; --sim is not supported")
    if getattr(cfg.cart, "enabled", False):
        raise SystemExit("teach-poses uses the thumbstick clicks the cart lift "
                         "owns; run without --cart.enabled")

    asyncio.run(_run(cfg, list(args.labels), args.out))
