"""Compile and run the native optical regression; never opens tracker hardware.

Example:
    python tests/native/run_libsurvive_optical_outliers.py \
        --source ~/.almond/libsurvive --build ~/.almond/libsurvive/build

The source/build must include 0010-filter-established-pose-optical-outliers.patch. The negative
control disables only its filter, using the same library, inputs and solver.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    source, build = args.source.resolve(), args.build.resolve()
    includes = [
        source / "include/libsurvive",
        source / "include",
        source / "redist",
        source / "libs/cnmatrix/include",
        source / "libs/cnkalman/include",
    ]
    with tempfile.TemporaryDirectory(prefix="axol-optical-regression-") as directory:
        binary = Path(directory) / "optical-regression"
        subprocess.run(
            [
                os.environ.get("CC", "cc"),
                "-O2",
                *(f"-I{path}" for path in includes),
                str(Path(__file__).with_name("libsurvive_optical_outliers.c")),
                f"-L{build}",
                f"-Wl,-rpath,{build}",
                "-lsurvive",
                "-lm",
                "-o",
                str(binary),
            ],
            check=True,
        )
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(build)
        results = {}
        for name, enabled in (("filtered", "1"), ("original_policy", "0")):
            result = subprocess.run(
                [str(binary), enabled], capture_output=True, text=True, env=env
            )
            report = json.loads(result.stdout)
            expected_failures = 0 if enabled == "1" else 100
            if (
                report["cases"] != 300
                or report["failures"] != expected_failures
                or report["clean_rejections"] != 0
                or result.returncode != (expected_failures != 0)
            ):
                raise RuntimeError(f"{name} failed: {report}\n{result.stderr}")
            results[name] = report
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
