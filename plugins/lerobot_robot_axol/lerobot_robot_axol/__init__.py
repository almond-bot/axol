"""LeRobot plugin for the Almond Axol dual-arm robot.

LeRobot auto-imports any installed distribution whose name starts with
``lerobot_robot_`` (see ``lerobot.utils.import_utils.register_third_party_plugins``),
so installing this package makes the Axol devices available to the stock
LeRobot CLI tools by type name:

- robot ``axol`` (:class:`~almond_axol.lerobot.robot.AxolRobot`)
- teleoperator ``axol_vr`` (:class:`~almond_axol.lerobot.teleop.AxolVRTeleop`)
- camera ``zed`` (:class:`~almond_axol.lerobot.camera.ZedCamera`)

The classes themselves live in the ``almond-axol`` SDK; this package only
triggers their draccus type registration at CLI startup.
"""

from almond_axol.lerobot.camera.configuration_zed import ZedCameraConfig
from almond_axol.lerobot.robot import AxolRobot, AxolRobotConfig
from almond_axol.lerobot.teleop import AxolVRTeleop, AxolVRTeleopConfig

__all__ = [
    "AxolRobot",
    "AxolRobotConfig",
    "AxolVRTeleop",
    "AxolVRTeleopConfig",
    "ZedCameraConfig",
]

try:
    # pyzed ships with the ZED SDK (installed on the robot via `axol zed.install`),
    # not from PyPI. Keep the plugin importable without it so type registration —
    # and with it the whole plugin — never breaks on machines without the SDK;
    # opening a camera without pyzed still fails with a clear error.
    from almond_axol.lerobot.camera import ZedCamera  # noqa: F401

    __all__.append("ZedCamera")
except ImportError:  # pragma: no cover - depends on the ZED SDK being provisioned
    pass
