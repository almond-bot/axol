# What normal looks like (jelly, 2026-10-07)

Measured on Almond's jelly robot with the shipped gains and a fresh
`tune.factory` calibration. Use them as orders of magnitude, not targets —
compare a robot mainly with itself (interleaved) and its two arms with each
other.

## slow_osc (right arm, wrist IMU) — the acceptance test

| | untuned | shipped |
|---|---|---|
| vertical (mm, 2 s p2p, 1–15 Hz) | 4.3–4.5 | ~2.5 (2.2–3.0 day to day) |
| 1–3 Hz sway `low_mm` | ~4.0 | 1.8–2.1 |
| 3–15 Hz `high_mm` | — | 1.5–1.7 |
| accel RMS (m/s²), peak | — | 0.5–0.7, ~5.3 Hz |

- The same runs by the encoders (`enc`, no camera needed): `low_mm`
  1.35–1.52 shipped, ~1.8 on the Sep 29 – Oct 2 configs, 2.4–3.1 untuned;
  `high_mm` 0.80–0.93. It reads ~25% under the IMU (it can't see flex past
  the encoders) and tracks the IMU's A/B changes at r ≈ 0.75 (12/12
  directions where the IMU moved > 5%).
- An operator's own IMU (`scripts/ext_imu.py`) reads differently by where
  it is mounted (further out = more sway): compare runs with each other.
- These are with the stock gripper and wrist camera (~0.75 kg on wrist_3).
  Another end-effector changes them; compare such a robot with itself.
- Camera / IMU floor: ~0.03 mm still camera; `hold` motion 0.09–0.13 mm.
- Noise: ±10–15% run to run, 10–20% across a day. A real change on the
  shipped gains is ~5–10%: interleave ≥ 4 rounds.
- Buzz cost: accel up > ~10% or its peak moving to 8–12 Hz = added buzz.
- Tracking: lag 55–65 ms; `lagfree` RMS s1 0.09–0.16°, s2 ~0.33°, elbow
  0.28–0.39°; closed-loop modes s1 ~2.4 Hz, elbow ~3.2 Hz, ζ ≈ 0.5–0.6.
- Fast replays (`wirst_swing`, `shoulder_1_no_load`): 4–15 mm, mostly
  1–3 Hz — only check that a change adds no buzz there.

## Creep ripple at 3 °/s (mdeg RMS, `rip3_rms_mdeg`) and wrist-IMU sway

| Joint | right ripple | left ripple | right IMU 1–3 Hz (mm) | left IMU (mm) |
|---|---|---|---|---|
| shoulder_1 | 16–34 | 15–17 | ~0.95 | ~0.54 |
| shoulder_2 | 23–50 | 18–30 | ~0.8 | ~0.5 |
| shoulder_3 | 19–32 | 19–27 | ~0.5 | ~0.19 |
| elbow | 20–23 | ~28 | ~1.0 | ~0.85 |
| wrist_1 | ~35 | ~22 | 0.37–0.59 | ~0.15 |
| wrist_2 | ~21 | ~28 | — | — |
| wrist_3 | ~27 | — | — | — |

A joint at 2× these, or one arm at 2× the other on the same joint, is worth
a look. Shoulder ripple swings run to run (see known-issues).

## Bus recording (`can_trace.py summary`)

- Quiet joint in teleop or a creep: `err_band_mdeg` (1.5–15 Hz) ~5–40.
- Verdict threshold 100 mdeg RMS; `STRONG` at ≥ 300 (visible at the hand).
- Homing on impedance tracks within ~0.4°; holds within ~0.2°.

## Calibration (`tune.factory`)

- ~6 min per joint, ~1 h 40 min both arms; ~2.5 s torque-off at start.
- Friction fit vs measured: within ~10% at 2–15 °/s; −22…−36% at 1 °/s and
  +4…+33% at 30 °/s are normal.
- Typical `fc` (Nm): shoulders 0.35–1.2, shoulder_3 / elbow 0.3–0.6, wrists
  0.1–0.5. `Fo` up to ~±1 Nm on the shoulders (it absorbs the gravity model's
  constant error, sign mirrored between the arms).
- Gravity: 0–7 of 14 joints rejected is normal.

## Shipped gains (impedance, 240 Hz)

| Joint | kp | kd | host damper | Stribeck gain |
|---|---|---|---|---|
| shoulder_1 | 450 | 5 | 110 at 2.6 Hz, Q 1 | 0.8 |
| shoulder_2 | 500 | 5 | 70 (pose-tracked) | 0.8 |
| shoulder_3 | 180 | 5 | — | 0 |
| elbow | 200 | 5 | 10 at 1.6 Hz, Q 1.5 | 0.8 |
| wrist_1 | 180 | 1.7 | — | 0 |
| wrist_2 / wrist_3 | 130 | 2.25 / 2.0 | — | 0 |

Calibration tools hold and sweep at fixed calibration gains instead
(s1/s2 250/3.5, s3 180/5, elbow 130/5, wrists as above). `robot_report.py`
→ `effective_config` shows what a given robot actually runs.
