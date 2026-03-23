from __future__ import annotations

from abc import ABC, abstractmethod


class _MotorDriver(ABC):
    @abstractmethod
    async def enable(self) -> None: ...

    @abstractmethod
    async def disable(self) -> None: ...

    @abstractmethod
    async def clear_errors(self) -> None: ...

    @abstractmethod
    async def get_position(self) -> float:
        """Return current shaft position in revolutions."""
        ...
