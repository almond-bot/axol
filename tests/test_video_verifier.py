"""The per-episode video verifier decodes only while no episode is recording."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.recording.record_proc import _EpisodeVideoVerifier

try:
    import av
except ImportError:  # pragma: no cover - exercised only where PyAV is missing
    av = None


def _write_mp4(path: Path, frames: int) -> None:
    """A tiny all-intra mp4 so ``_probe`` has real packets to count and decode."""
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "1"}
        for i in range(frames):
            img = np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _episode_row(root: Path, cameras: dict[str, Path]) -> dict[str, object]:
    """The subset of a LeRobot episode metadata row ``submit`` reads."""
    row: dict[str, object] = {}
    for key in cameras:
        row[f"videos/{key}/chunk_index"] = 0
        row[f"videos/{key}/file_index"] = 0
    return row


class VerifierGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())

    def _make_video(self, key: str, frames: int) -> Path:
        path = self.root / "videos" / key / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True)
        _write_mp4(path, frames)
        return path

    @unittest.skipIf(av is None, "PyAV not installed")
    def test_probe_parks_while_suspended_and_finishes_on_resume(self) -> None:
        mp4 = self._make_video("observation.images.cam", frames=12)
        gate = threading.Event()  # cleared: "an episode is recording"
        progress: list[int] = []
        result: list[tuple[int, int]] = []

        def probe() -> None:
            # Count frames through the gate by observing each wait.
            original_wait = gate.wait

            def counting_wait(*args: object, **kwargs: object) -> bool:
                progress.append(1)
                return original_wait(*args, **kwargs)

            gate.wait = counting_wait  # type: ignore[method-assign]
            result.append(_EpisodeVideoVerifier._probe(mp4, gate))

        worker = threading.Thread(target=probe, daemon=True)
        worker.start()
        # Suspended: the decode stops after its first frame and does not finish.
        worker.join(timeout=0.5)
        self.assertTrue(worker.is_alive(), "decode must park while the gate is down")
        self.assertEqual(result, [])
        self.assertEqual(len(progress), 1)  # exactly one frame decoded, then parked
        gate.set()
        worker.join(timeout=10.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [(12, 12)])

    @unittest.skipIf(av is None, "PyAV not installed")
    def test_suspend_defers_a_submitted_episode_until_resume(self) -> None:
        cam = "observation.images.cam"
        self._make_video(cam, frames=8)
        verifier = _EpisodeVideoVerifier(self.root)
        try:
            verifier.suspend()  # an episode starts before the previous one verified
            with self.assertLogs("almond_axol.recording.record_proc", "INFO") as logs:
                verifier.submit(_episode_row(self.root, {cam: Path()}))
                time.sleep(0.5)
                self.assertEqual(verifier.pending_files, 1)
                self.assertFalse(
                    any("fully decodable" in line for line in logs.output),
                    "nothing may be verified while an episode records",
                )
                verifier.resume()
                deadline = time.monotonic() + 10.0
                while verifier.pending_files and time.monotonic() < deadline:
                    time.sleep(0.02)
            self.assertEqual(verifier.pending_files, 0)
            self.assertTrue(
                any("fully decodable (8 frames)" in line for line in logs.output)
            )
        finally:
            verifier.close(timeout=5.0)

    def test_close_lifts_the_gate_so_shutdown_never_waits_on_it(self) -> None:
        verifier = _EpisodeVideoVerifier(self.root)
        released = threading.Event()

        def fake_probe(mp4: Path, proceed: threading.Event | None = None):
            if proceed is not None:
                proceed.wait()
            released.set()
            return 1, 1

        with patch.object(_EpisodeVideoVerifier, "_probe", staticmethod(fake_probe)):
            verifier.suspend()
            verifier.submit(_episode_row(self.root, {"cam": Path()}))
            self.assertFalse(released.wait(0.2))
            t0 = time.monotonic()
            verifier.close(timeout=5.0)
        self.assertTrue(released.is_set())
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertFalse(verifier._thread.is_alive())
        self.assertEqual(verifier.pending_files, 0)


if __name__ == "__main__":
    unittest.main()
