"""One MCAP file: channels registered on first use, crash-safe finish.

The file is written as ``<name>.mcap.partial`` and renamed to ``<name>.mcap``
only after :meth:`EpisodeFile.finish` has written the summary and flushed it
to disk, so a reader never sees a half-written episode under its final name.

Chunks are uncompressed: almost all of the bytes are already-compressed video,
and compressing them again would only spend recorder CPU.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

from . import messages

PARTIAL_SUFFIX = ".partial"
_CHUNK_SIZE = 4 * 1024 * 1024


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("almond-axol")
    except Exception:  # noqa: BLE001 - version is informational
        return "unknown"


class EpisodeFile:
    """Writes one episode's MCAP file.

    ``clock_offset_ns`` maps the host's ``perf_counter`` clock (seconds, the
    clock every sample is stamped on) to Unix time: ``unix_ns = perf_s * 1e9
    + clock_offset_ns``. Message log and publish times are both that instant,
    which is also each message's header stamp.

    ``topic_names`` renames topics as they are written (default name to the
    customer's name; see :mod:`.topics`). Callers always pass default names.
    """

    def __init__(
        self,
        final_path: Path,
        *,
        clock_offset_ns: int,
        topic_names: Mapping[str, str] | None = None,
    ) -> None:
        from mcap.writer import CompressionType, Writer

        self.final_path = Path(final_path)
        self.partial_path = self.final_path.with_name(
            self.final_path.name + PARTIAL_SUFFIX
        )
        self.clock_offset_ns = int(clock_offset_ns)
        self.topic_names = dict(topic_names or {})
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[bytes] | None = open(self.partial_path, "wb")  # noqa: SIM115
        try:
            self._writer: Any = Writer(
                self._file,
                chunk_size=_CHUNK_SIZE,
                compression=CompressionType.NONE,
            )
            self._writer.start(
                profile="ros2", library=f"almond-axol/{_package_version()}"
            )
        except BaseException:
            self._file.close()
            self._file = None
            with contextlib.suppress(OSError):
                self.partial_path.unlink()
            raise
        self._schemas: dict[str, int] = {}
        self._channels: dict[str, int] = {}
        self._sequence: dict[int, int] = {}
        self.message_counts: dict[str, int] = {}
        self.first_ns: int | None = None
        self.last_ns: int | None = None

    def stamp(self, perf_s: float) -> int:
        """Unix nanoseconds for a ``perf_counter`` instant."""
        return int(round(perf_s * 1e9)) + self.clock_offset_ns

    def _channel(self, topic: str, msg_type: str) -> int:
        channel = self._channels.get(topic)
        if channel is not None:
            return channel
        schema = self._schemas.get(msg_type)
        if schema is None:
            schema = self._writer.register_schema(
                name=messages.schema_name(msg_type),
                encoding="ros2msg",
                data=messages.schema_text(msg_type).encode(),
            )
            self._schemas[msg_type] = schema
        channel = self._writer.register_channel(
            topic=self.topic_names.get(topic, topic),
            message_encoding="cdr",
            schema_id=schema,
        )
        self._channels[topic] = channel
        self._sequence[channel] = 0
        return channel

    def write(self, topic: str, msg_type: str, data: bytes, stamp_ns: int) -> None:
        """Append one encoded message."""
        if self._file is None:
            raise RuntimeError(f"{self.final_path.name} is already closed")
        channel = self._channel(topic, msg_type)
        sequence = self._sequence[channel]
        self._sequence[channel] = sequence + 1
        self._writer.add_message(
            channel_id=channel,
            log_time=stamp_ns,
            data=data,
            publish_time=stamp_ns,
            sequence=sequence,
        )
        self.message_counts[topic] = self.message_counts.get(topic, 0) + 1
        if self.first_ns is None or stamp_ns < self.first_ns:
            self.first_ns = stamp_ns
        if self.last_ns is None or stamp_ns > self.last_ns:
            self.last_ns = stamp_ns

    def metadata(self, name: str, values: Mapping[str, Any]) -> None:
        """Add a metadata record (every value written as a string)."""
        if self._file is None:
            raise RuntimeError(f"{self.final_path.name} is already closed")
        self._writer.add_metadata(
            name, {str(k): "" if v is None else str(v) for k, v in values.items()}
        )

    def finish(self) -> Path:
        """Write the summary, flush to disk, and move into place."""
        if self._file is None:
            raise RuntimeError(f"{self.final_path.name} is already closed")
        try:
            self._writer.finish()
            self._file.flush()
            os.fsync(self._file.fileno())
        finally:
            self._file.close()
            self._file = None
        os.replace(self.partial_path, self.final_path)
        _fsync_dir(self.final_path.parent)
        return self.final_path

    def discard(self) -> None:
        """Close without finishing and delete the partial file."""
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
            self._file = None
        with contextlib.suppress(FileNotFoundError):
            self.partial_path.unlink()

    @property
    def closed(self) -> bool:
        return self._file is None


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
