"""Abstract base class for the Axol robot and simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..motor import JointValues


class RobotBase(ABC):
    """Common interface for the Axol hardware robot and the viser simulation.

    All position values are in revolutions. Gripper is normalised
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
    async def get_positions(self) -> tuple[JointValues, JointValues]:
        """Return current joint positions (rev) for both arms as (left, right)."""
        ...

    @abstractmethod
    async def set_positions(
        self,
        left: JointValues | None = None,
        right: JointValues | None = None,
    ) -> None:
        """Command joint positions (rev) on one or both arms.

        Args:
            left:  Target positions keyed by joint. Omit or pass ``None`` to
                skip the left arm.
            right: Target positions keyed by joint. Omit or pass ``None`` to
                skip the right arm.
        """
        ...
