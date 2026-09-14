from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from almond_axol.cli import mantis_latency


class _Server:
    def __init__(self) -> None:
        self.set_on_frame = Mock()

    async def __aenter__(self) -> _Server:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    def get_frame(self) -> object:
        return object()


class MantisLatencyLifecycleTests(unittest.TestCase):
    def _camera_patches(self, camera: object, server: _Server):
        return (
            patch(
                "almond_axol.lerobot.camera.ZedCamera",
                return_value=camera,
                create=True,
            ),
            patch(
                "almond_axol.lerobot.camera.ZedCameraConfig",
                return_value=SimpleNamespace(),
            ),
            patch("almond_axol.vr.server.VRServer", return_value=server),
            patch.object(mantis_latency, "_make_aruco_detector", return_value=Mock()),
        )

    def test_connect_interrupt_disconnects_partially_owned_camera(self) -> None:
        camera = SimpleNamespace(
            connect=Mock(side_effect=KeyboardInterrupt),
            disconnect=Mock(),
            is_connected=True,
            thread=None,
        )
        server = _Server()
        patches = self._camera_patches(camera, server)
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        with self.assertRaises(KeyboardInterrupt):
            asyncio.run(mantis_latency._run("left", 123, 0.0))

        camera.disconnect.assert_called_once_with()

    def test_live_analysis_thread_is_rejected_before_camera_disconnect(self) -> None:
        camera = SimpleNamespace(
            connect=Mock(),
            disconnect=Mock(),
            is_connected=True,
            thread=None,
        )
        server = _Server()
        analysis_thread = SimpleNamespace(
            start=Mock(),
            join=Mock(),
            is_alive=Mock(return_value=True),
        )
        patches = self._camera_patches(camera, server)
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        with (
            patch.object(mantis_latency.asyncio, "to_thread", new=AsyncMock()),
            patch.object(
                mantis_latency.threading,
                "Thread",
                return_value=analysis_thread,
            ),
            self.assertRaisesRegex(RuntimeError, "analysis thread did not stop"),
        ):
            asyncio.run(mantis_latency._run("left", 123, 0.0))

        server.set_on_frame.assert_any_call(None)
        camera.disconnect.assert_called_once_with()

    def test_callback_clear_failure_still_joins_analysis_thread(self) -> None:
        camera = SimpleNamespace(
            connect=Mock(),
            disconnect=Mock(),
            is_connected=True,
            thread=None,
        )
        server = _Server()
        server.set_on_frame.side_effect = [None, RuntimeError("callback clear failed")]
        analysis_thread = SimpleNamespace(
            start=Mock(),
            join=Mock(),
            is_alive=Mock(return_value=False),
        )
        patches = self._camera_patches(camera, server)
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        with (
            patch.object(mantis_latency.asyncio, "to_thread", new=AsyncMock()),
            patch.object(
                mantis_latency.threading,
                "Thread",
                return_value=analysis_thread,
            ),
            self.assertRaisesRegex(RuntimeError, "callback clear failed"),
        ):
            asyncio.run(mantis_latency._run("left", 123, 0.0))

        analysis_thread.join.assert_called_once_with(timeout=5.0)
        camera.disconnect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
