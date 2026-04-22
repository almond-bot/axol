"""Abstract base class for the Axol robot and simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RobotBase(ABC):
    """Common interface for the Axol hardware robot and the viser simulation.

    All position values are in radians. Gripper is normalised
    [0.0 closed, 1.0 open].
    """

    @abstractmethod
    async def enable(self) -> None:
        """Enable the robot, or start the simulation server."""
        ...

    @abstractmethod
    async def disable(self) -> None:
        """Disable the robot, or stop the simulation server."""
        ...

    @abstractmethod
    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return current joint positions (rad) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order: 7 arm joints then gripper,
        or ``None`` if that arm is absent.
        """
        ...

    @abstractmethod
    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Send control commands to one or both arms (impedance for arm joints, position-force for gripper).

        Args:
            left:  Shape (8,) array of target positions (rad) in Joint enum order.
                   Pass ``None`` to skip the left arm.
            right: Same for the right arm.
        """
        ...
