"""
axol calibration.cameras

Measure where the head and wrist cameras sit on this robot, and save it as
the robot's camera calibration file (``~/.almond/camera_calibration.json``).
MCAP episodes recorded by Collect data then place the cameras in
``/tf_static`` from the measurement instead of from CAD.

    axol calibration.cameras --print_board ~/calibration_board.pdf
    axol calibration.cameras
    axol calibration.cameras --board.square_mm 39.6
    axol calibration.cameras --from_report ~/.almond/camera_calibration/run-20261124T181500Z

1. Print the board at 100% (actual size) and tape it flat to something
   rigid. Measure one square; if it isn't 40 mm, pass its size with
   ``--board.square_mm``.
2. Run the command. The arms go into gravity compensation. Lay the board
   flat on the table where the head camera sees it and press Enter.
3. Move one arm at a time by hand until its wrist camera sees the board, then
   let go. A view is taken when the arm is still. Tilt the camera toward the
   board from different sides, and turn it, between views.
4. Press Enter, move the board to another place on the table, and press Enter
   again. Use at least three places; four is better.
5. Type ``done`` and press Enter. The tool solves, prints the result per
   camera, and saves the file when every camera passes.

The cameras come from the control panel's camera settings (the same slots
teleop uses): ``overhead`` is the head camera, ``left_arm`` and
``right_arm`` the wrist cameras. ``--from_report`` solves again from a saved
run without the robot (it writes nothing unless ``--save_from_report true``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..camera_calibration.board import BoardConfig
from ..camera_calibration.checks import ProcedureLimits
from ..camera_calibration.file import DEFAULT_PATH, REPORTS_DIR, VALID_DAYS
from ..camera_calibration.session import CaptureRules, SessionOptions
from ..camera_calibration.solve import Limits
from ..constants import CAN_LEFT, CAN_RIGHT
from ..robot.config import AxolConfig
from .config import LogLevel, parse

_logger = logging.getLogger(__name__)
# Shared settings this command reads (from teleop's: the cameras and the arms).
_SETTINGS_KEYS = ("axol", "left_channel", "right_channel", "cameras")


@dataclass
class CameraCalibrationCmdConfig:
    """Config for ``axol calibration.cameras``."""

    axol: AxolConfig = field(default_factory=AxolConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    cameras: dict[str, int] = field(default_factory=dict)
    """Camera slot to ZED serial (overhead, left_arm, right_arm). Defaults to
    the control panel's camera settings."""
    board: BoardConfig = field(default_factory=BoardConfig)
    limits: Limits = field(default_factory=Limits)
    rules: CaptureRules = field(default_factory=CaptureRules)
    print_board: str | None = None
    """Write the printable board to this file (PDF or PNG) and exit."""
    from_report: str | None = None
    """Solve again from a saved run's folder, without the robot."""
    save_from_report: bool = False
    """With from_report: save the calibration file if it passes."""
    output: str = str(DEFAULT_PATH)
    """Where to save the camera calibration file."""
    sim: bool = False
    """Run the calibration procedure in the sim: record Recording A and B of
    the sim's robot, check them and write the evidence package. Nothing is
    saved as this robot's calibration."""
    packages: str = str(REPORTS_DIR)
    """Folder for the evidence packages (one per calibration ID)."""
    checks: ProcedureLimits = field(default_factory=ProcedureLimits)
    """The procedure's pass/fail limits."""
    valid_days: float = VALID_DAYS
    """Days until a calibration expires (0: never)."""
    sim_workers: int = 4
    """With sim: processes that render video."""
    kd: float = 0.5
    """Gravity compensation damping (Nm·s/rad)."""
    fps: int = 30
    log_level: LogLevel = "INFO"


def _settings_overlay() -> dict[str, Any]:
    try:
        from ..settings import shared_overlay

        overlay = shared_overlay("teleop")
    except Exception as exc:  # noqa: BLE001 - settings are a convenience
        _logger.debug("shared settings unavailable: %s", exc)
        return {}
    return {k: v for k, v in overlay.items() if k in _SETTINGS_KEYS}


def _options(cfg: CameraCalibrationCmdConfig) -> SessionOptions:
    from importlib.metadata import PackageNotFoundError, version

    from ..robot.identity import hub_serial

    try:
        software = version("almond-axol")
    except PackageNotFoundError:
        software = ""
    return SessionOptions(
        board=cfg.board,
        limits=cfg.limits,
        rules=cfg.rules,
        output=Path(cfg.output).expanduser(),
        kd=cfg.kd,
        fps=cfg.fps,
        robot_serial=hub_serial() or "",
        software_version=software,
    )


def _from_report(cfg: CameraCalibrationCmdConfig) -> None:
    from ..camera_calibration.board import Board
    from ..camera_calibration.session import finish, load_views

    folder = Path(cfg.from_report or "").expanduser()
    specs, views, board_config, images = load_views(folder / "views.json")
    options = _options(cfg)
    print(f"Solving {len(views)} views from {folder} ...")
    finish(
        specs,
        views,
        Board(board_config),
        images,
        options,
        folder / "resolved",
        write_file=cfg.save_from_report,
    )


def _open_cameras(
    specs: dict[str, Any], fps: int, stereo: set[int]
) -> tuple[list[Any], list[Any]]:
    """Open each camera's raw 1200p stream; returns (feeds, devices to close)."""
    from ..camera_calibration.session import CAPTURE_SIZE, _Feed
    from ..video.zed_sdk import ZedSdkCamera, ZedSdkCameraConfig, ZedSdkStereoCamera

    feeds, devices = [], []
    for spec in specs.values():
        config = ZedSdkCameraConfig(
            serial=int(spec.serial),
            fps=fps,
            width=CAPTURE_SIZE[0],
            height=CAPTURE_SIZE[1],
            stereo=int(spec.serial) in stereo,
            unrectified=True,
        )
        if config.stereo:
            device = ZedSdkStereoCamera(config)
            device.connect()
            devices.append(device)
            feeds.append(_Feed(spec, device.left_view))
        else:
            device = ZedSdkCamera(config)
            device.connect()
            devices.append(device)
            feeds.append(_Feed(spec, device))
    return feeds, devices


def _run(cfg: CameraCalibrationCmdConfig) -> None:
    from ..camera_calibration.board import Board
    from ..camera_calibration.session import (
        LiveSession,
        _Detector,
        camera_specs,
        run_live,
    )
    from ..constants import AxolModel, urdf_arm_joint_names, urdf_path
    from ..recording.mcap.urdf_tree import UrdfTree
    from ..robot import Axol
    from ..settings import resolve_gripper, resolve_robot_model
    from ..zed import stereo_serials
    from ..zed.calibration import SETTINGS_DIR

    if not cfg.cameras:
        raise SystemExit(
            "No cameras are set. Assign the head and wrist cameras in the control "
            "panel's camera settings, or pass --cameras."
        )
    if cfg.left_channel is None and cfg.right_channel is None:
        raise SystemExit("Both arms are off; the wrist cameras need the arms.")
    options = _options(cfg)
    if not options.robot_serial:
        _logger.warning(
            "This robot's serial is unknown (no Axol hub found). MCAP recordings "
            "can only use a calibration file that names the robot."
        )
    model = resolve_robot_model()
    gripper = resolve_gripper(cfg.axol.gripper)
    tree = UrdfTree.load(
        urdf_path(model, gripper=gripper if model is AxolModel.MOBILE else "parallel")
    )
    stereo = stereo_serials()
    specs, problems = camera_specs(tree, dict(cfg.cameras), stereo, SETTINGS_DIR)
    for problem in problems:
        print(problem)
    if not any(spec.moves for spec in specs.values()):
        raise SystemExit(
            "No wrist camera to calibrate with. Check the camera settings."
        )
    board = Board(cfg.board)
    print(
        f"Board: {board.describe()}. Cameras: {', '.join(s.label for s in specs.values())}."
    )
    feeds, devices = _open_cameras(specs, cfg.fps, stereo)
    detector = _Detector(feeds, board)
    detector.start()
    try:
        session = LiveSession(
            options,
            specs,
            feeds,
            tree,
            {
                side: urdf_arm_joint_names(is_left=side == "left")
                for side in ("left", "right")
            },
        )
        robot = Axol(
            config=cfg.axol,
            left_channel=cfg.left_channel,
            right_channel=cfg.right_channel,
        )
        asyncio.run(run_live(session, robot))
    finally:
        detector.stop.set()
        detector.join(timeout=2.0)
        for device in devices:
            try:
                device.disconnect()
            except Exception:  # noqa: BLE001 - best effort on the way out
                pass


def _run_sim(cfg: CameraCalibrationCmdConfig) -> None:
    from dataclasses import replace

    from ..camera_calibration.board import Board
    from ..camera_calibration.procedure import report
    from ..camera_calibration.sim import SimOptions, run_session

    board_config = cfg.board
    if not board_config.id:
        board_config = replace(board_config, id="SIM-BOARD")
    board = Board(board_config)
    print(f"Calibrating the sim's robot with {board.describe()}.")
    result = run_session(
        Path(cfg.packages).expanduser() / "sim",
        board,
        options=SimOptions(workers=max(1, cfg.sim_workers)),
        limits=cfg.checks,
        calibration_path=None,
        valid_days=cfg.valid_days,
    )
    for line in report(result):
        print(line)


def main(argv: list[str]) -> None:
    cfg = parse(CameraCalibrationCmdConfig, argv, fallback_overlay=_settings_overlay())
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    if cfg.print_board:
        from ..camera_calibration.board import Board

        board = Board(cfg.board)
        path = board.save_page(Path(cfg.print_board))
        print(
            f"Saved the board to {path}. Print it at 100% (actual size), then check "
            f"that one square measures {cfg.board.square_mm:g} mm."
        )
        return
    if cfg.from_report:
        _from_report(cfg)
        return
    if cfg.sim:
        _run_sim(cfg)
        return
    try:
        _run(cfg)
    except KeyboardInterrupt:
        print("\nStopped. Nothing was saved.")
