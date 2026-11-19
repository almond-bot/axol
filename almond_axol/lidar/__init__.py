"""Livox Mid-360 lidar support: driver, streaming and network setup.

- :mod:`.livox`: the Mid-360 Ethernet protocol (discovery, parameters,
  point and IMU packets), in pure Python.
- :mod:`.stream`: :class:`LidarStream`, a background receiver that cuts the
  point stream into frames on the host's monotonic clock.
- :mod:`.config`: :class:`LidarConfig` (``~/.almond/lidar.json``), written by
  ``axol lidar.setup``.
- :mod:`.network`: finding the lidar on a wired port and configuring it.

Mapping, localization and navigation build on this in :mod:`almond_axol.nav`.
"""

from .config import LidarConfig, LidarMount
from .livox import LidarStatus, LivoxClient, LivoxDevice, discover
from .stream import ImuSample, LidarFrame, LidarStream

__all__ = [
    "ImuSample",
    "LidarConfig",
    "LidarFrame",
    "LidarMount",
    "LidarStatus",
    "LidarStream",
    "LivoxClient",
    "LivoxDevice",
    "discover",
]
