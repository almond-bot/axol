"""
axol diag.offline

Offline analysis suites over a teleop flight-recorder capture (``axol
teleop --teleop.record PREFIX``). Each suite isolates one stage of
the teleop pipeline and answers one question from the recording alone —
no hardware needed:

* ``wifi``       — is the transport jittery? VR frame inter-arrival
                   statistics, gaps and bursts, separated from tracking noise.
* ``filtering``  — what does the pose filter stack pass through? Band RMS
                   per stage (raw → filtered → EE target) and the lag each
                   stage adds.
* ``kinematics`` — does the IK add motion the hand didn't make? EE tracking
                   error (FK of solved joints vs the world target), per-joint
                   churn, and injected mid-band jitter.

Results print as a report and (with ``--save-run``) persist as tuning-run
artifacts for charting and comparison in the diagnostics UI.

Examples:
    axol teleop --teleop.record /tmp/rec    # record a session first
    axol diag.offline wifi /tmp/rec
    axol diag.offline filtering /tmp/rec --save-run
    axol diag.offline kinematics /tmp/rec --save-run --label "solver v2"
"""

from __future__ import annotations

import argparse


def _add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "suite",
        choices=["wifi", "filtering", "kinematics"],
        help="Which pipeline stage to analyze",
    )
    ap.add_argument(
        "prefix",
        help="Flight-recorder prefix used with --teleop.record",
    )
    ap.add_argument(
        "--save-run",
        action="store_true",
        help="Persist the analysis as a tuning-run artifact "
        "(~/.almond/diagnostics/tuning/) for the diagnostics UI",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="Free-form note stored on the run artifact",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``diag.offline`` on a subparser tree (serve-UI introspection).

    The ``axol`` CLI dispatches this command lazily via ``_DIAG_COMMANDS``
    (so the heavy imports stay off the common path); this registrar exists
    for the serve layer's argparse schema introspection.
    """
    p = subparsers.add_parser(
        "diag.offline",
        help="Offline wifi/filtering/kinematics analysis of a recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(p)
    p.set_defaults(func=lambda args: main_args(args))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="axol diag.offline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(ap)
    args = ap.parse_args(argv)
    main_args(args)


def main_args(args: argparse.Namespace) -> None:
    """Run the selected suite over the capture and print its report."""
    from ..tuning import save_run
    from ..tuning.offline import (
        filtering_analysis,
        kinematics_analysis,
        print_filtering_report,
        print_kinematics_report,
        print_wifi_report,
        wifi_analysis,
    )

    analyze, report = {
        "wifi": (wifi_analysis, print_wifi_report),
        "filtering": (filtering_analysis, print_filtering_report),
        "kinematics": (kinematics_analysis, print_kinematics_report),
    }[args.suite]

    print(f"Analyzing {args.suite} from {args.prefix} ...")
    metrics, series, params = analyze(args.prefix)
    report(metrics)

    if args.save_run:
        run_id = save_run(args.suite, series, metrics, params=params, label=args.label)
        print(f"\nSaved tuning run {run_id} (kind={args.suite})")


if __name__ == "__main__":
    main()
