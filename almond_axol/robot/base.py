"""Abstract base class for the Axol robot and simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class HardwareCleanupError(RuntimeError):
    """Hardware teardown failed, so another owner must not reacquire its bus."""


_CLEANUP_UNCERTAIN_ATTR = "_axol_hardware_cleanup_uncertain"


def mark_hardware_cleanup_uncertain(
    error: BaseException, cleanup_error: BaseException
) -> None:
    """Preserve ``error`` while telling the runner that teardown also failed."""
    setattr(error, _CLEANUP_UNCERTAIN_ATTR, True)
    error.add_note(
        "Robot cleanup also failed; hardware ownership is uncertain: "
        f"{type(cleanup_error).__name__}: {cleanup_error}"
    )


def is_hardware_cleanup_uncertain(error: BaseException) -> bool:
    """Whether an exception proves hardware handoff did not complete."""
    return isinstance(error, HardwareCleanupError) or bool(
        getattr(error, _CLEANUP_UNCERTAIN_ATTR, False)
    )


class RobotBase(ABC):
    """Common interface for the Axol hardware robot and the viser simulation.

    All position values are in radians. Gripper is normalised
    [0.0 closed, 1.0 open]. Arrays keep their (8,) shape on the gripperless
    SKU (``AxolConfig.has_gripper = False``); the gripper element is simply
    ignored on write and reported as 0.0 on read.

    Subclasses inherit ``__aenter__`` / ``__aexit__`` which call
    :meth:`enable` and :meth:`disable`, so they can be used with
    ``async with``.
    """

    @abstractmethod
    async def enable(self) -> None:
        """Enable the robot, or start the simulation server."""
        ...

    @abstractmethod
    async def disable(self) -> None:
        """Disable the robot, or stop the simulation server."""
        ...

    async def __aenter__(self) -> Self:
        """Enter the async context, enabling the robot via :meth:`enable`."""
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit the async context, disabling the robot via :meth:`disable`."""
        try:
            await self.disable()
        except HardwareCleanupError:
            raise
        except BaseException as exc:
            raise HardwareCleanupError(
                "robot disable failed; hardware ownership is uncertain"
            ) from exc

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
