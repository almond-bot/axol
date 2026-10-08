# Tool catalogue

Run from the robot's checkout with its venv (`~/axol/.venv/bin/axol` or
`uv run axol`) — an older system-wide `/usr/local/bin/axol` may lack the
flags. **A4** = uses the MyActuator 0xA4 firmware position loop: avoid
(ground rule 1). "Moves" = drives the arms.

## Read-only / analysis

| Tool | Use it for |
|---|---|
| `scripts/robot_report.py [--bundle F]` | Intake snapshot (Phase 0) |
| `scripts/can_trace.py record -o F` / `summary F` / `decode F --out D` | Passive bus recording; per-joint verdict; per-joint CSVs |
| `axol motor.health` / `motor.info` / `motor.dump-config` | Are the motors reachable; firmware; stored parameters (reads) |
| `axol diag.teleop-jitter PREFIX` | Which teleop stage adds jitter, from a `teleop --teleop.record` or `tune.motion --record` capture |
| `axol diag.offline {wifi,filtering,kinematics}` | Offline analysis of a recording |
| `axol tune.tf SIDE.JOINT RUN…` | Fit a joint's closed-loop tracking model from chirp runs (`--save` → `~/.almond/tracking_models.json`) |
| `axol tune.filter` | Offline test of the teleop filter stack |
| `scripts/explain_learned.py` | Decompose a `--learn` correction into feedforward terms |
| Diagnostics dashboard (`/diagnostics` under `axol serve`) | Past runs, charts, A/B overlay of two runs of the same kind |

## Calibration (moves)

| Tool | Use it for | Persists |
|---|---|---|
| `axol tune.factory [--arms left\|right] [--keep-gravity] [--mass/--com] [--raw-dir D]` | Friction + Stribeck + gravity, all joints (~1 h 40 both arms) | calibration.json (+ cloud with `AXOL_SUPABASE_KEY`) |
| `axol tune.friction --l\|--r --joint J --profile slow [--raw-csv F] [--save]` | One joint's friction; `--fit-csv F` re-fits without moving | `--save` → calibration.json |
| `axol tune.gravity --l\|--r --joint J [--save]` | One link's CoM (run distal → proximal) | `--save` |
| `axol tune.breakaway --l\|--r --joint J --poses …` | Static vs sliding friction; ceiling for `stiction_gain` | CSV only |
| `axol calibration.pull` / `calibration.push [--dry-run]` | Fetch / upload calibration by hub serial | factory_calibration.json / cloud |

All calibration tools are impedance-only: they hold every joint at fixed
calibration gains with gravity fed forward before anything moves. They
have **no tracking guard** — stay at the e-stop; Ctrl-C **once**.

## Evaluation and A/B (moves)

**`axol tune.motion --arms left|right --motion NAME|PATH`** — replays a
reference motion through the production controller and scores it.
- Runs the panel settings over the calibration, as teleop does (gains,
  link mass / CoM, stiffness, `has_gripper`); the first line says
  `Config: panel settings, gripper yes|no`. `--defaults` runs the bare
  calibrated defaults; `--no-gripper` / `--stiffness` override the panel.
  (Before this, it ran the bare defaults with a gripper: on an older
  checkout pass `--no-gripper` on a gripperless robot.)
- `--repeat N` (a saved run per pass, `[k/N]`), `--label`, `--hold
  SIDE.JOINT[=DEG]` (freeze joints).
- `--gain [SIDE.]JOINT.FIELD=V` — one run only, nothing persists. Fields:
  kp, kd, kd_host, kd_host_hz, kd_host_q, j_eff, stiction_gain,
  stiction_load_gain, stiction_err_deg, dither_nm, dither_hz,
  stribeck_gain/dfs/load_gain/vs/pole, friction.fc/k/fv/fo/fl, mass,
  com.x/y/z (firmware.* = **A4**).
- Guard on by default: `--guard-dev-deg 10` (+0.1 s × commanded speed),
  `--guard-osc-deg 1.5` (1.5–15 Hz RMS), `--guard-vib-deg 0.15` (> 15 Hz).
  For creeps use `--guard-dev-deg 4 --guard-osc-deg 1`. A trip restores the
  base gains, holds, returns home; exit 3 (4 = arm left holding). Never
  `--no-guard`.
- `--record PREFIX` → the core's per-tick trace with every feedforward term
  (`~/.almond/recordings/PREFIX_rt.npz`).
- Prints per joint RMS / lagfree / lag ms / jitter / buzz@Hz (healthy buzz
  ≈ 0.005°) and the wrist IMU line `vertical X mm = 1-3 Hz a + 3-15 Hz b`,
  accel, peak Hz (only with a wrist camera; `--no-imu` skips it). Runs go
  to `~/.almond/diagnostics/tuning/<id>/`.
- Research flags (don't ship results from them): `--learn N`,
  `--learn-imu`, `--correction RUN`, `--invert`, `--notch`, `--imu-damp*`,
  `--gyro-damp`, `--torque-probe`, `--enc2`, `--fast-impedance`.
- **A4:** `--a4 SIDE.JOINT`, `--controller position`.

**`scripts/tuning_queue.py plan|run|summary`** — interleaved rounds of
variants (`--gain` overrides or `NAME@calib=PATH`), halt rules, and a
verdict. `summary` decides on the slow_osc sway (`--metric auto`):
`imu.low_mm` when the runs have the wrist IMU, else `enc.low_mm` — the same
sway from the encoders (flange forward kinematics), needing ≥ 15%. It
calls a variant that buys sway with 3–15 Hz shake / accel a `trade`;
`--metric rip3_rms_mdeg` for creep screening. `plan --no-imu` for a robot
without wrist cameras. Every saved run is scored by
`almond_axol/tuning/creep_score.py` (`score_run`: ripple, `imu`, `enc`).

**`axol tune.pid --l|--r --joint J`** — one joint, step or sine, a grid of
`--kp/--kd` candidates, ranked (overshoot, settling, ring Hz, holder
wobble, IMU). `--host-kd/--host-kd-hz/--host-kd-q`, `--pose J=DEG`,
`--pose-by-hand` (drag the arm to the worst pose, step-probe there). Other
joints hold on impedance. `--save` writes kp/kd to calibration.json —
prefer panel settings. **A4:** `--a4-holders`.

**`axol motion.chirp SIDE.JOINT [--carrier DPS]`** writes a sine-sweep
motion (0.3–8 Hz) to fly with `tune.motion`, then fit with `tune.tf`.
**`axol motion.build PREFIX`** turns a teleop / gravity-comp recording into
a reference motion.

**`scripts/ext_imu.py check LOG` / `attach LOG --session DIR | --run ID…`**
— an IMU the operator mounted on the end-effector (no wrist camera). CSV
`t, ax, ay, az[, gx, gy, gz]`, Unix time, raw acceleration with gravity
(m/s² or g, guessed), ≥ 100 Hz. `attach` matches each run (its `t0_wall`,
then a ± 2 s cross-correlation with the flange's acceleration from the
encoders; `--search`), scores it like the wrist IMU and saves
`imu_external.json` beside the run; `tuning_queue.py summary` and
`score_run` use it as `imu` (`source: external`) when the run has no wrist
camera IMU. Runs from before `t0_wall` existed are searched ± 10 s.

**`scripts/creep_motions.py --all`** — per-joint creeps, both arms, plus
`slow_osc_left.npz`, into `~/.almond/motions/`.

## Avoid (A4) unless Almond asks

`axol tune.a4`, `tune.motion --a4 / --controller position`, `tune.pid
--a4-holders`, `scripts/creep_test.py`, `scripts/fw_gains.py` (writes motor
ROM with `--persist` / `--accel`), config `wire_mode: "a4"`. Firmware gains
stay stock.

## Reference motions

| Motion | What | Judge on |
|---|---|---|
| `slow_osc` (right; `~/.almond/motions/slow_osc_left.npz` for the left) | 28 s smoothed teleop | wrist IMU `low_mm` (1–3 Hz sway), or `enc.low_mm` without a camera — **the acceptance test** |
| `*_creep[_left]` (generated) | one joint at 3 and 6°/s | `rip3_rms_mdeg`, screening only |
| `hold` | 40 s still | noise floor, parked buzz, limit cycles |
| `shoulder_1_no_load`, `wirst_swing` | fast swings (≤ 173°/s) | no new buzz on fast motion |

## Where values live (later wins)

coded defaults (`robot/config.py`) → `factory_calibration.json` (pulled
from the cloud) → `calibration.json` (this machine, scoped to the hub
serial) → panel settings `settings.json` (Advanced → Axol) → a `--gain` for
one run. Persist a tuning decision in the panel settings.
