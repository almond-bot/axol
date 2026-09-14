from __future__ import annotations

import importlib
import pkgutil

import almond_axol


def test_all_core_modules_import_without_hardware() -> None:
    failures: list[str] = []
    optional_prefixes = (
        "almond_axol.lerobot",
        "almond_axol.recording.record_proc",
        "almond_axol.cli.collect_",
        "almond_axol.cli.inference_server",
        "almond_axol.cli.replay_dataset",
        "almond_axol.cli.run_policy",
    )
    for module in pkgutil.walk_packages(
        almond_axol.__path__, almond_axol.__name__ + "."
    ):
        if module.name.startswith(optional_prefixes):
            continue
        try:
            importlib.import_module(module.name)
        except ImportError as exc:
            failures.append(f"{module.name}: {exc}")

    assert not failures, "core module import failures:\n" + "\n".join(failures)
