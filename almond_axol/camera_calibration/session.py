"""Hand-guided camera calibration on the robot (``axol calibration.cameras``).

The arms are in gravity compensation, so the operator moves them by hand. The
board lies still on the table. Whenever an arm is still and its wrist camera
sees the board from a new angle, the session records a view: the detected
corners, the arm's joint readings (averaged over the still moment) and, if it
also sees the board, the head camera's corners. The operator moves the board
to a few places on the table, pressing Enter before and after each move so no
view is taken while it moves.

``done`` solves (:mod:`.solve`), prints the result per camera, and writes the
calibration file when every camera passes. Each run keeps a report folder
under ``~/.almond/camera_calibration/<id>/``: the views (``views.json``,
enough to solve again offline with ``--from_report``), the captured images,
and the same images with the solved corners drawn on them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .board import Board, BoardConfig, Detection
from .file import DEFAULT_PATH, REPORTS_DIR, document, transform_entry, write
from .optics import Lens
from .solve import CameraSpec, Limits, Solution, View, rotation_spread, solve

_logger = logging.getLogger(__name__)

# The control panel's camera slots, as calibration cameras.
SLOTS = {
    "overhead": ("head", "Head camera"),
    "left_arm": ("left_wrist", "Left wrist camera"),
    "right_arm": ("right_wrist", "Right wrist camera"),
}
CAPTURE_RESOLUTION = "HD1200"
CAPTURE_SIZE = (1920, 1200)
# Corners a wrist view needs (fewer give a poor pose at the edge of the image).
_MIN_VIEW_CORNERS = 8
# Head views kept per board position (the head and board are both still, so
# more add little).
_HEAD_VIEWS_PER_PLACE = 8
_STATUS_EVERY_S = 3.0


@dataclass
class CaptureRules:
    """When a still arm's view counts as a new one."""

    # Joint motion (rad) allowed over the still window.
    still_rad: float = 0.002
    still_s: float = 0.4
    # A view is new when the mount turned or moved this much from every
    # earlier view of the same board position.
    new_view_deg: float = 6.0
    new_view_m: float = 0.03


@dataclass
class SessionOptions:
    board: BoardConfig = field(default_factory=BoardConfig)
    limits: Limits = field(default_factory=Limits)
    rules: CaptureRules = field(default_factory=CaptureRules)
    output: Path = DEFAULT_PATH
    reports_dir: Path = REPORTS_DIR
    kd: float = 0.5
    fps: int = 30
    robot_serial: str = ""
    software_version: str = ""


# -- camera specs ------------------------------------------------------------


def rigid_mount(tree: Any, frame: str) -> str:
    """The highest link ``frame`` is rigidly fixed to (fixed joints only)."""
    link = frame
    while True:
        joint = tree.parent_joint.get(link)
        if joint is None or joint.moving:
            return link
        link = joint.parent


def camera_specs(
    tree: Any,
    cameras: dict[str, int],
    stereo: set[int],
    calibration_dir: Path,
) -> tuple[dict[str, CameraSpec], list[str]]:
    """Specs for the configured camera slots, and why any are left out."""
    from ..recording.mcap.scene import camera_frame_id
    from ..recording.mcap.zed_conf import camera_intrinsics, load_calibration

    specs: dict[str, CameraSpec] = {}
    problems: list[str] = []
    for slot, serial in sorted(cameras.items()):
        if slot not in SLOTS or not serial:
            continue
        key, label = SLOTS[slot]
        is_stereo = int(serial) in stereo
        frame = camera_frame_id(slot, "left" if is_stereo else None)
        if not tree.has_link(frame):
            problems.append(f"{label}: this robot model has no {frame} frame")
            continue
        conf = load_calibration(int(serial), calibration_dir)
        if conf is None:
            problems.append(
                f"{label} (SN{serial}): no factory calibration file in "
                f"{calibration_dir}. Open the camera once with the ZED SDK to "
                "download it."
            )
            continue
        intrinsics = camera_intrinsics(
            conf,
            eye="left",
            capture_resolution=CAPTURE_RESOLUTION,
            width=CAPTURE_SIZE[0],
            height=CAPTURE_SIZE[1],
            fisheye=not is_stereo,
        )
        if intrinsics is None:
            problems.append(
                f"{label} (SN{serial}): its calibration file has no 1200p entry"
            )
            continue
        mount = rigid_mount(tree, frame)
        specs[key] = CameraSpec(
            key=key,
            label=label,
            frame=frame,
            mount=mount,
            cad=tree.fk(mount, frame, {}),
            lens=Lens.from_intrinsics(intrinsics),
            serial=int(serial),
            moves=mount != "root" and slot != "overhead",
        )
    return specs, problems


# -- report ------------------------------------------------------------------


def _spec_json(spec: CameraSpec) -> dict[str, Any]:
    lens = spec.lens
    return {
        "key": spec.key,
        "label": spec.label,
        "frame": spec.frame,
        "mount": spec.mount,
        "cad": spec.cad.tolist(),
        "serial": spec.serial,
        "moves": spec.moves,
        "lens": {
            "width": lens.width,
            "height": lens.height,
            "k": lens.k.tolist(),
            "d": lens.d.tolist(),
            "fisheye": lens.fisheye,
        },
    }


def _spec_from_json(data: dict[str, Any]) -> CameraSpec:
    lens = data["lens"]
    return CameraSpec(
        key=data["key"],
        label=data["label"],
        frame=data["frame"],
        mount=data["mount"],
        cad=np.asarray(data["cad"], dtype=np.float64),
        lens=Lens(
            width=int(lens["width"]),
            height=int(lens["height"]),
            k=np.asarray(lens["k"], dtype=np.float64),
            d=np.asarray(lens["d"], dtype=np.float64),
            fisheye=bool(lens["fisheye"]),
        ),
        serial=data.get("serial"),
        moves=bool(data["moves"]),
    )


def save_views(
    path: Path,
    specs: dict[str, CameraSpec],
    views: list[View],
    board: BoardConfig,
    images: dict[int, str],
) -> None:
    """Everything a solve needs, so it can run again offline."""
    data = {
        "board": asdict(board),
        "cameras": [_spec_json(s) for s in specs.values()],
        "views": [
            {
                "camera": v.camera,
                "placement": v.placement,
                "capture": v.capture,
                "mount_pose": v.mount_pose.tolist(),
                "ids": v.detection.ids.tolist(),
                "pixels": v.detection.pixels.tolist(),
                "image": images.get(id(v), ""),
            }
            for v in views
        ],
    }
    path.write_text(json.dumps(data) + "\n")


def load_views(
    path: Path,
) -> tuple[dict[str, CameraSpec], list[View], BoardConfig, list[str]]:
    """The saved views of a report folder (``views.json``)."""
    data = json.loads(Path(path).read_text())
    specs = {s["key"]: _spec_from_json(s) for s in data["cameras"]}
    views = [
        View(
            camera=v["camera"],
            placement=int(v["placement"]),
            mount_pose=np.asarray(v["mount_pose"], dtype=np.float64),
            detection=Detection(
                ids=np.asarray(v["ids"], dtype=np.int64),
                pixels=np.asarray(v["pixels"], dtype=np.float64).reshape(-1, 2),
            ),
            capture=int(v.get("capture", -1)),
        )
        for v in data["views"]
    ]
    images = [str(v.get("image", "")) for v in data["views"]]
    return specs, views, BoardConfig(**data["board"]), images


def describe(solution: Solution) -> list[str]:
    """One block of lines per camera, for the operator."""
    lines = []
    for result in solution.cameras.values():
        label = result.spec.label
        if result.transform is None:
            lines.append(f"{label}: FAIL. {'. '.join(result.problems)}.")
            continue
        status = "PASS" if result.passed else "FAIL"
        detail = (
            f"{label}: {status}. {result.views} views"
            + (f" ({result.rejected} rejected)" if result.rejected else "")
            + f", error {result.rms_px:.2f} px, {result.change_mm:.1f} mm and "
            f"{result.change_deg:.2f} degrees from CAD"
        )
        if result.spec.moves:
            detail += f", wrist turned {result.turn_deg[0]:.0f} and {result.turn_deg[1]:.0f} degrees"
        lines.append(detail + ".")
        lines += [f"  {p}." for p in result.problems]
    return lines


def calibration_document(
    solution: Solution,
    *,
    robot_serial: str,
    calibrated_at: datetime,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """The calibration file for a passing solution."""
    import hashlib

    entries = []
    for result in solution.cameras.values():
        if result.transform is None:
            continue
        spec = result.spec
        entries.append(
            transform_entry(
                spec.mount,
                spec.frame,
                result.transform,
                camera_serial=spec.serial,
                views=result.views,
                rms_px=round(result.rms_px, 4),
                change_from_cad_mm=round(result.change_mm, 2),
                change_from_cad_deg=round(result.change_deg, 3),
            )
        )
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    stamp = calibrated_at.strftime("%Y%m%dT%H%M%SZ")
    return document(
        robot_serial=robot_serial,
        calibration_id=f"cam-{stamp}-{digest[:8]}",
        calibrated_at=calibrated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        status="pass" if solution.passed else "fail",
        transforms=entries,
        **extra,
    )


def draw_views(
    solution: Solution,
    specs: dict[str, CameraSpec],
    views: list[View],
    board: Board,
    images: list[str],
    out_dir: Path,
) -> int:
    """Draw the detected (green) and solved (red) corners on each saved image."""
    import cv2

    from .solve import reprojection

    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for view, image_path, rms in zip(views, images, solution.view_rms, strict=True):
        if not image_path:
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        for x, y in view.detection.pixels:
            cv2.circle(image, (round(x), round(y)), 6, (0, 200, 0), 2)
        projected = reprojection(solution, specs[view.camera], view, board)
        if projected is not None:
            for x, y in projected:
                if abs(x) < 1e5 and abs(y) < 1e5:
                    cv2.drawMarker(
                        image,
                        (round(x), round(y)),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        14,
                        2,
                    )
        label = f"{specs[view.camera].label}, board place {view.placement}: " + (
            f"error {rms:.2f} px" if np.isfinite(rms) else "not used"
        )
        cv2.putText(
            image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6
        )
        cv2.putText(image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
        cv2.imwrite(str(out_dir / Path(image_path).name), image)
        drawn += 1
    return drawn


def finish(
    specs: dict[str, CameraSpec],
    views: list[View],
    board: Board,
    images: list[str],
    options: SessionOptions,
    report_dir: Path,
    *,
    write_file: bool = True,
) -> tuple[Solution, Path | None]:
    """Solve, print the result, write the report and (on a pass) the file."""
    solution = solve(specs, views, board, options.limits)
    for line in describe(solution):
        print(line)
    now = datetime.now(UTC)
    doc = calibration_document(
        solution,
        robot_serial=options.robot_serial,
        calibrated_at=now,
        extra={
            "software_version": options.software_version,
            "board": asdict(board.config),
            "report": str(report_dir),
        },
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "calibration.json").write_text(json.dumps(doc, indent=2) + "\n")
    report = {
        "passed": solution.passed,
        "cameras": {
            key: {
                "views": r.views,
                "rejected": r.rejected,
                "rms_px": r.rms_px,
                "change_mm": r.change_mm,
                "change_deg": r.change_deg,
                "turn_deg": list(r.turn_deg),
                "problems": r.problems,
            }
            for key, r in solution.cameras.items()
        },
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    drawn = draw_views(solution, specs, views, board, images, report_dir / "checked")
    if drawn:
        print(f"Images with the solved corners: {report_dir / 'checked'}")
    written = None
    if solution.passed and write_file:
        written = write(doc, options.output)
        print(
            f"Saved {doc['calibration_id']} to {written}. MCAP episodes use it from the next episode."
        )
    elif solution.passed:
        print(
            "Every camera passed. Nothing was saved: add --save_from_report true to save it."
        )
    else:
        print(
            "Calibration failed, so nothing was saved. Fix the items above and run it again."
        )
    return solution, written


# -- the live session ---------------------------------------------------------


@dataclass
class _Feed:
    """One open camera, its latest frame and its latest board detection."""

    spec: CameraSpec
    camera: Any  # ZedSdkCamera or StereoEyeView
    frame: np.ndarray | None = None
    perf: float = 0.0
    detection: Detection | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Detector(threading.Thread):
    """Detects the board in each camera's newest frame, round robin."""

    def __init__(self, feeds: list[_Feed], board: Board) -> None:
        super().__init__(name="calibration-detect", daemon=True)
        self.feeds = feeds
        self.board = board
        self.stop = threading.Event()
        self._last: dict[str, float] = {}

    def run(self) -> None:
        while not self.stop.is_set():
            busy = False
            for feed in self.feeds:
                try:
                    frame, perf, _recv = feed.camera.read_latest_with_ts()
                except Exception:  # noqa: BLE001 - camera warming up or recovering
                    continue
                if self._last.get(feed.spec.key) == perf:
                    continue
                self._last[feed.spec.key] = perf
                busy = True
                detection = self.board.detect(frame)
                with feed.lock:
                    feed.frame, feed.perf, feed.detection = frame, perf, detection
            if not busy:
                self.stop.wait(0.01)


class LiveSession:
    """The robot, the cameras and the operator's prompts."""

    def __init__(
        self,
        options: SessionOptions,
        specs: dict[str, CameraSpec],
        feeds: list[_Feed],
        tree: Any,
        arm_joints: dict[str, list[str]],
    ) -> None:
        self.options = options
        self.board = Board(options.board)
        self.specs = specs
        self.feeds = {f.spec.key: f for f in feeds}
        self.tree = tree
        self.arm_joints = arm_joints
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.report_dir = Path(options.reports_dir).expanduser() / f"run-{stamp}"
        self.images_dir = self.report_dir / "images"
        self.views: list[View] = []
        self.images: dict[int, str] = {}
        self.placement = 0
        self.capturing = False
        self.captures = 0
        history = int(4 * 250 * max(options.rules.still_s, 0.5))
        self._joints: dict[str, deque[tuple[float, np.ndarray]]] = {
            side: deque(maxlen=history) for side in ("left", "right")
        }
        self._seen: dict[str, float] = {}
        self._head_used: set[float] = set()
        # Images are written off the control loop.
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="calib-jpeg"
        )

    # -- joints ----------------------------------------------------------

    def record_joints(self, axol: Any) -> None:
        now = time.perf_counter()
        for side in ("left", "right"):
            arm = getattr(axol, side, None)
            if arm is None:
                continue
            with contextlib.suppress(Exception):
                self._joints[side].append(
                    (now, np.array(arm.positions[:7], dtype=np.float64))
                )

    def _still_joints(self, side: str, at: float) -> np.ndarray | None:
        """The arm's mean joints around ``at`` if it held still, else None."""
        rules = self.options.rules
        rows = [q for t, q in self._joints[side] if at - rules.still_s <= t <= at + 0.1]
        if len(rows) < 10:
            return None
        q = np.asarray(rows)
        if float(np.max(np.ptp(q, axis=0))) > rules.still_rad:
            return None
        return q.mean(axis=0)

    def _mount_pose(self, spec: CameraSpec, side: str, q: np.ndarray) -> np.ndarray:
        values = dict(zip(self.arm_joints[side], (float(v) for v in q), strict=True))
        return self.tree.fk("root", spec.mount, values)

    # -- views -----------------------------------------------------------

    def _is_new(self, camera: str, mount_pose: np.ndarray) -> bool:
        rules = self.options.rules
        for view in self.views:
            if view.camera != camera or view.placement != self.placement:
                continue
            r = view.mount_pose[:3, :3].T @ mount_pose[:3, :3]
            angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
            shift = np.linalg.norm(view.mount_pose[:3, 3] - mount_pose[:3, 3])
            if angle < rules.new_view_deg and shift < rules.new_view_m:
                return False
        return True

    def _save_image(self, frame: np.ndarray, name: str) -> str:
        import cv2

        self.images_dir.mkdir(parents=True, exist_ok=True)
        path = self.images_dir / f"{name}.jpg"

        def write_jpeg() -> None:
            # Frames are RGB; OpenCV writes BGR.
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

        self._writer.submit(write_jpeg)
        return str(path)

    def flush_images(self) -> None:
        """Wait for every image to be written."""
        self._writer.shutdown(wait=True)

    def check_views(self) -> None:
        """Take a view from each still wrist camera that sees the board anew."""
        if not self.capturing:
            return
        now = time.perf_counter()
        for key, spec in self.specs.items():
            if not spec.moves:
                continue
            feed = self.feeds.get(key)
            if feed is None:
                continue
            with feed.lock:
                frame, perf, detection = feed.frame, feed.perf, feed.detection
            if frame is None or self._seen.get(key) == perf or now < perf + 0.1:
                continue
            self._seen[key] = perf
            if detection is None or len(detection) < _MIN_VIEW_CORNERS:
                continue
            side = "left" if key.startswith("left") else "right"
            q = self._still_joints(side, perf)
            if q is None:
                continue
            mount_pose = self._mount_pose(spec, side, q)
            if not self._is_new(key, mount_pose):
                continue
            self.captures += 1
            view = View(key, self.placement, mount_pose, detection, self.captures)
            self.views.append(view)
            self.images[id(view)] = self._save_image(
                frame, f"{self.captures:03d}_{key}"
            )
            count = sum(1 for v in self.views if v.camera == key)
            turn = rotation_spread(
                [v.mount_pose[:3, :3] for v in self.views if v.camera == key]
            )
            print(
                f"{spec.label}: view {count} ({len(detection)} corners). "
                f"Wrist turned {turn[0]:.0f} and {turn[1]:.0f} degrees so far."
            )
            self._add_head_view()

    def _add_head_view(self) -> None:
        spec = self.specs.get("head")
        feed = self.feeds.get("head")
        if spec is None or feed is None:
            return
        with feed.lock:
            frame, perf, detection = feed.frame, feed.perf, feed.detection
        here = [
            v
            for v in self.views
            if v.camera == "head" and v.placement == self.placement
        ]
        if (
            frame is None
            or detection is None
            or perf in self._head_used
            or len(here) >= _HEAD_VIEWS_PER_PLACE
        ):
            return
        self._head_used.add(perf)
        view = View("head", self.placement, np.eye(4), detection, self.captures)
        self.views.append(view)
        self.images[id(view)] = self._save_image(frame, f"{self.captures:03d}_head")

    def status(self) -> str:
        parts = [f"Board place {self.placement}"]
        for key, spec in self.specs.items():
            feed = self.feeds.get(key)
            seen = feed is not None and feed.detection is not None
            count = sum(1 for v in self.views if v.camera == key)
            parts.append(
                f"{spec.label.lower()}: {count} views, board {'seen' if seen else 'not seen'}"
            )
        return " | ".join(parts)

    # -- prompts ---------------------------------------------------------

    def prompt(self) -> None:
        if self.capturing:
            print(
                f"Board place {self.placement}: move one arm at a time until its "
                "wrist camera sees the board, then let go. Tilt the camera toward "
                "the board from different sides and turn it. A view is taken when "
                "the arm is still. Press Enter before you move the board, or type "
                "done and press Enter to finish."
            )
        else:
            print(
                "Put the board flat on the table where the head camera sees it, "
                "then press Enter. Type done and press Enter to finish."
            )

    def command(self, line: str) -> str | None:
        """Handle one typed line; returns ``"done"`` or ``"quit"`` to stop."""
        word = line.strip().lower()
        if word in ("done", "d"):
            return "done"
        if word in ("quit", "q", "exit"):
            return "quit"
        if self.capturing:
            self.capturing = False
            print("Move the board now. Views are paused until you press Enter.")
        else:
            self.placement += 1
            self.capturing = True
            self._seen.clear()
        self.prompt()
        return None


def _stdin_lines(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]) -> None:
    for line in sys.stdin:
        loop.call_soon_threadsafe(queue.put_nowait, line)
    loop.call_soon_threadsafe(queue.put_nowait, "quit")


async def run_live(
    session: LiveSession, robot: Any, *, rate_hz: float = 250.0
) -> tuple[Solution | None, Path | None]:
    """Gravity compensation and view capture until the operator finishes."""
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[str] = asyncio.Queue()
    threading.Thread(target=_stdin_lines, args=(loop, lines), daemon=True).start()
    dt = 1.0 / rate_hz
    session.prompt()
    last_status = time.perf_counter()
    async with robot as axol:
        outcome = None
        while outcome is None:
            start = time.monotonic()
            await axol.gravity_compensate(kd=session.options.kd)
            session.record_joints(axol)
            session.check_views()
            while not lines.empty():
                outcome = session.command(lines.get_nowait())
                if outcome is not None:
                    break
            if (
                session.capturing
                and time.perf_counter() - last_status > _STATUS_EVERY_S
            ):
                last_status = time.perf_counter()
                print(session.status())
            spent = time.monotonic() - start
            if spent < dt:
                await asyncio.sleep(dt - spent)
        if outcome == "quit":
            print("Stopped without solving. Nothing was saved.")
            return None, None
        print(f"Solving from {len(session.views)} views ...")
        session.report_dir.mkdir(parents=True, exist_ok=True)
        session.flush_images()
        images = [session.images.get(id(v), "") for v in session.views]
        save_views(
            session.report_dir / "views.json",
            session.specs,
            session.views,
            session.options.board,
            session.images,
        )
        # Keep the arms in gravity compensation while solving.
        task = asyncio.create_task(
            asyncio.to_thread(
                finish,
                session.specs,
                session.views,
                session.board,
                images,
                session.options,
                session.report_dir,
            )
        )
        while not task.done():
            start = time.monotonic()
            await axol.gravity_compensate(kd=session.options.kd)
            spent = time.monotonic() - start
            if spent < dt:
                await asyncio.sleep(dt - spent)
        solution, written = task.result()
        print(
            "Rest both arms, then press Enter to stop. The arms go limp when the tool stops."
        )
        while lines.empty():
            start = time.monotonic()
            await axol.gravity_compensate(kd=session.options.kd)
            spent = time.monotonic() - start
            if spent < dt:
                await asyncio.sleep(dt - spent)
        return solution, written
