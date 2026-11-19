"""Save lidar frames and IMU samples to a file, and play them back.

A recording is one ``.npz`` file. Played back, it behaves like a
:class:`~almond_axol.lidar.LidarStream` (``frames()`` and
``imu_between()``), so a map can be built offline from a drive recorded
earlier (``axol nav.map NAME --from FILE``), or a problem reproduced away from
the robot.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.state_files import secure_atomic_write_bytes
from .stream import ImuSample, LidarFrame

FORMAT_VERSION = 1


@dataclass
class LidarRecording:
    frames_: list[LidarFrame] = field(default_factory=list)
    imu: list[ImuSample] = field(default_factory=list)
    _stamps: list[float] = field(default_factory=list, repr=False)

    def add_frame(self, frame: LidarFrame) -> None:
        self.frames_.append(frame)

    def add_imu(self, samples: list[ImuSample]) -> None:
        self.imu.extend(samples)

    def __len__(self) -> int:
        return len(self.frames_)

    @property
    def duration(self) -> float:
        if not self.frames_:
            return 0.0
        return self.frames_[-1].stamp - self.frames_[0].stamp

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        f = self.frames_
        counts = np.array([len(x) for x in f], np.int64)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            version=np.array(FORMAT_VERSION),
            counts=counts,
            stamps=np.array([x.stamp for x in f], np.float64),
            sensor_ns=np.array([x.sensor_ns for x in f], np.int64),
            durations=np.array([x.duration for x in f], np.float64),
            xyz=np.concatenate([x.xyz for x in f])
            if f
            else np.zeros((0, 3), np.float32),
            reflectivity=np.concatenate([x.reflectivity for x in f])
            if f
            else np.zeros(0, np.uint8),
            tag=np.concatenate([x.tag for x in f]) if f else np.zeros(0, np.uint8),
            times=np.concatenate([x.times for x in f])
            if f
            else np.zeros(0, np.float32),
            imu_stamps=np.array([s.stamp for s in self.imu], np.float64),
            imu_gyro=np.array([s.gyro for s in self.imu], np.float64).reshape(-1, 3),
            imu_accel=np.array([s.accel for s in self.imu], np.float64).reshape(-1, 3),
        )
        secure_atomic_write_bytes(p, buf.getvalue(), mode=0o644)
        return p

    @classmethod
    def load(cls, path: str | Path) -> LidarRecording:
        with np.load(Path(path).expanduser()) as z:
            if int(z["version"]) != FORMAT_VERSION:
                raise ValueError(
                    f"{path} is a version {int(z['version'])} lidar recording"
                )
            counts = z["counts"]
            ends = np.cumsum(counts)
            starts = ends - counts
            frames = [
                LidarFrame(
                    stamp=float(z["stamps"][i]),
                    sensor_ns=int(z["sensor_ns"][i]),
                    xyz=z["xyz"][a:b],
                    reflectivity=z["reflectivity"][a:b],
                    tag=z["tag"][a:b],
                    times=z["times"][a:b],
                    duration=float(z["durations"][i]),
                    seq=i,
                )
                for i, (a, b) in enumerate(zip(starts, ends, strict=True))
            ]
            imu = [
                ImuSample(float(t), g, a)
                for t, g, a in zip(
                    z["imu_stamps"], z["imu_gyro"], z["imu_accel"], strict=True
                )
            ]
        return cls(frames, imu)

    # -- playback (LidarStream-like) ----------------------------------------

    def frames(
        self, timeout: float = 2.0, realtime: bool = False
    ) -> Iterator[LidarFrame]:
        start = time.monotonic()
        t0 = self.frames_[0].stamp if self.frames_ else 0.0
        for frame in self.frames_:
            if realtime:
                time.sleep(max(0.0, (frame.stamp - t0) - (time.monotonic() - start)))
            yield frame

    def imu_between(self, t0: float, t1: float) -> list[ImuSample]:
        import bisect

        stamps = (
            [s.stamp for s in self.imu]
            if len(self._stamps) != len(self.imu)
            else self._stamps
        )
        self._stamps = stamps
        return self.imu[bisect.bisect_left(stamps, t0) : bisect.bisect_left(stamps, t1)]
