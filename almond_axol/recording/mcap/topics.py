"""MCAP topic names: the defaults, and a customer's renames.

Every episode is written with the default topic names below. The **MCAP topic
names** setting (``recording.mcap_topics``, collect-data's ``mcap_topics``)
renames any of them, as comma-separated ``default=new`` pairs::

    /head/left/video=/top-left-camera/image-raw, /joint_states=/robot/joints

Only names change; message types and contents stay the same. Each file
records its renames in the ``recording`` metadata (``topic_names``), so a
reader can map them back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# Topics every episode can have, whatever cameras are recorded.
FIXED_TOPICS: tuple[str, ...] = (
    "/tf",
    "/tf_static",
    "/robot_description",
    "/joint_states",
    "/joint_command",
    "/joint_command/info",
    "/ee_pose/left",
    "/ee_pose/right",
    "/ee_target/left",
    "/ee_target/right",
    "/teleop/left/pose",
    "/teleop/right/pose",
    "/teleop/joy",
    "/base/odom",
    "/base/wheel_states",
    "/base/lift",
    "/base/command",
    "/base/cmd_vel",
)
CAMERA_TOPICS = ("video", "camera_info")

# A topic name: a leading slash, then letters, digits and _ - . ~ / (no
# trailing slash, no empty segment).
_TOPIC = re.compile(r"^(/[A-Za-z0-9_.~-]+)+$")


def default_topics(
    camera_prefixes: Iterable[str], imu_topics: Iterable[str] = ()
) -> list[str]:
    """Every default topic for a session recording these cameras and IMUs."""
    topics = list(FIXED_TOPICS)
    for prefix in camera_prefixes:
        topics += [f"{prefix}/{name}" for name in CAMERA_TOPICS]
    topics += [t for t in imu_topics if t not in topics]
    return topics


def parse_topic_names(text: str | None) -> dict[str, str]:
    """``default=new`` pairs (comma or newline separated) as a mapping.

    Raises ``ValueError`` with a message for the operator when a pair is
    malformed, a name is not a valid topic, or a default appears twice.
    """
    out: dict[str, str] = {}
    if not text or not text.strip():
        return out
    for raw in re.split(r"[,\n]", text):
        pair = raw.strip()
        if not pair:
            continue
        default, sep, new = (part.strip() for part in pair.partition("="))
        if not sep or not default or not new:
            raise ValueError(
                f"MCAP topic names: {pair!r} is not a default=new pair, such as "
                "/joint_states=/robot/joints."
            )
        for name in (default, new):
            if not _TOPIC.match(name):
                raise ValueError(
                    f"MCAP topic names: {name!r} is not a topic name. Start it "
                    "with / and use letters, digits, _ - . or ~."
                )
        if default in out:
            raise ValueError(f"MCAP topic names: {default} is renamed twice.")
        if new != default:
            out[default] = new
    return out


def topic_names_problem(
    renames: Mapping[str, str], available: Iterable[str]
) -> str | None:
    """Why ``renames`` can't apply to these topics, or ``None``."""
    available = list(available)
    unknown = sorted(set(renames) - set(available))
    if unknown:
        return (
            f"MCAP topic names: this session has no {', '.join(unknown)} topic. "
            f"Its topics are {', '.join(available)}."
        )
    final: dict[str, str] = {}
    for topic in available:
        name = renames.get(topic, topic)
        if name in final:
            return (
                f"MCAP topic names: {final[name]} and {topic} would both be "
                f"named {name}."
            )
        final[name] = topic
    return None


def format_topic_names(renames: Mapping[str, str]) -> str:
    """The setting's text for a mapping (sorted by default name)."""
    return ", ".join(f"{k}={v}" for k, v in sorted(renames.items()))
