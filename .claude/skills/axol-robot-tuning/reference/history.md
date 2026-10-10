# What was tried, what won, what shipped (jelly, 2026-09-22 → 10-08)

Two weeks of tuning Almond's jelly robot (right arm mostly), plus a robot
in the field. Read this before proposing a change: most obvious ideas were already
tried, with numbers. Gains below are the shipped defaults unless marked.

## The problem and what causes it

- **Symptom:** slow teleop wobbles at the hand — a 1–3 Hz sway (what an
  operator feels) plus 3–15 Hz shake. Untuned, `slow_osc` read **4.3–4.5 mm**
  vertical at the wrist IMU.
- **Mechanism:** separate 1.2–2 Hz rings (ζ ≈ 0.5–0.6) kicked by stick-slip,
  not a sustained oscillation — one cycle predicts only 0–8% of the next.
  The ripple frequency stays near 2 Hz whatever the speed, so it is
  stick-slip, not cogging (cogging scales with speed).
- **Who:** shoulder_1 carries 65–75% of the 1–3 Hz sway, then shoulder_2 and
  the elbow; the elbow dominates 3–15 Hz.
- **Ceiling:** encoders explain 67–83% of the IMU wobble before tuning but
  only ~40% after learning — the rest is flex past the motor encoders that
  joint control can't see. The gearbox isn't it: the output encoder (0x60)
  agrees with the control position within ~7 mdeg.

## How the shipped configuration was reached

1. **Impedance at 240 Hz, firmware loops abandoned** (Sep 22–23).
   - Position controller (every joint on its firmware loop, 400 Hz): 85 Hz
     wrist buzz, then a CAN TX stall.
   - 0xA4 on shoulder_1/elbow inside impedance: ~0.6 mm tip shake at
     position_kp 1.0/1.4, but on the edge of stability (pkp 1.4 passed twice,
     then vibrated three times; 2.0 vibrated at once); lag 60–65 ms; no
     compliance, no feedforward, contact watchdog blind; `speed_kp` is its
     only damping and also the buzz knob.
   - The planner (`planner_accel` 60000) aborted on fast moves (lag 135–213
     ms); `planner_accel` 0 written by mistake made homing "go wild".
   - Damiao wrists on position-velocity: +14–39% tip shake, a 4 Hz ring.
   - Loop rate 240/400/480 made no measurable difference.
   - On another robot the stock 0xA4 loop later oscillated a shoulder
     (see known-issues). Firmware gains are now left stock and 0xA4 is
     opt-in everywhere.
2. **Friction calibrated in the runtime's own law** (`tune.friction
   --profile slow`, then `tune.factory`): sliding `(fc + fl·|g|)`, `fo`, and
   a Stribeck excess with `vs ≤ 5.7°/s` (left free, it swallowed the whole
   curve: fc 0, ±0.7 Nm flipping at rest). jelly's fits are every robot's
   fallback.
3. **shoulder_1** (s1_creep 3°/s ripple p2p 282 → 62 mdeg over 6 rounds):
   Stribeck 0.5 (pole 40) was the first big win on slow_osc (4.41 → 3.01
   mm); 0.7–0.9 at pole 20 were worse, 0.8 at pole 40 wins on creep;
   kd 5; kp 450 (kp 500: no gain); host damper 70 at 3.2 Hz Q 1 → **110 at
   2.6 Hz Q 1** (Oct 6: creep −29%, slow_osc 1–3 Hz −15% alone).
4. **elbow:** kp 130 → 200, kd 5, host damper 10 at 1.6 Hz Q 1.5,
   Stribeck 0.8. Damper centres at 3 / 4.5 Hz and kp 150–350 later: within
   noise at the tool, more acceleration.
5. **shoulder_2** (Oct 6): kp 250 → **500**, kd 3.5 → 5, host damper 35 →
   **70** (pose-tracked band), Stribeck **0.8** (creep −46…−59%).
6. **Result:** slow_osc **~2.5 mm** (−43% vs untuned); the Oct 6 sweep added
   ~−7% 1–3 Hz sway (per round −13…0%) with accel +6%.

Shipped defaults (both arms):

| joint | kp | kd | host damper | Stribeck (pole) |
|---|---|---|---|---|
| shoulder_1 | 450 | 5 | 110 at 2.6 Hz, Q 1 | 0.8 (40) |
| shoulder_2 | 500 | 5 | 70, pose-tracked band | 0.8 (40) |
| shoulder_3 | 180 | 5 | 0 — it pumped the 8.7–11 Hz mast mode | 0 |
| elbow | 200 | 5 | 10 at 1.6 Hz, Q 1.5 | 0.8 (40) |
| wrist_1 | 180 | 1.7 | 0 | 0 |
| wrist_2 | 130 | 2.25 — kd 5 buzzed at 110 Hz, never raise | 0 | 0 |
| wrist_3 | 130 | 2.0 | 0 | 0 |

## Per-joint lever results (creep = encoder ripple; tool = slow_osc IMU)

| joint | lever | creep | at the tool | status |
|---|---|---|---|---|
| s1 | Stribeck 0.5 → 0.8 (pole 40) | −25%…−35% | 1–3 Hz −30% (vs none) | shipped |
| s1 | Stribeck 1.0 | worse (130 vs 79) | — | rejected |
| s1 | kp 500 | worse (113 vs 79) | — | rejected |
| s1 | damper 110 at 2.6 Hz | −29% | 1–3 Hz −15% | shipped |
| s1 | damper 140 at 2.6 / at 2.2 Hz | — | −6% / 3–15 Hz +8–11% | not shipped |
| s1 | 480 Hz (with kp 450) | −25% once | no change | not shipped |
| s1 | cogging map | no effect | — | removed |
| s1 | dither 1 Nm | — | +3% vert, accel +12% | rejected |
| s2 | kp 500 kd 5 damper 70 Stribeck 0.8 | −46…−59% | −2% alone | shipped |
| s2 | Stribeck at old kp 250 | ripple +30% | — | was off until kp 500 |
| elbow | kp 300–350, damper 20 | −19…−34% | IMU vert 1.10 → 1.22–1.31 (worse) | not shipped |
| elbow | cogging map + damper | −56% on el_creep | no difference on slow_osc | removed |
| s3 | kp 300–400 | −28…−51% | IMU +8% | not shipped |
| s3 | Stribeck 0.8 (Sep 29 fits) | ripple +18%, IMU sway +39% | — | off |
| s3 | Stribeck 0.4 / 0.8 (Oct 7 fits) | right −23%, left none | right IMU −34% (2 rounds) | **open — needs slow_osc + IMU on both arms** |
| w1 | kp 400 kd 3.5 | −48% | IMU 0.16 → 0.21 | not shipped |
| w1 | Stribeck 0.4–0.8 | right −12…−15%, left none | unchanged | open |
| all | every creep winner together | — | 3–15 Hz +23%, accel +32% | rejected |

**Lesson:** per-joint creep wins (−30…−57%) became at most −7% sway at the
tool, and stiffer or more damped settings trade 1–3 Hz sway for 3–15 Hz
shake. Screen on creep; decide on slow_osc with the wrist IMU.

**Without a camera (Oct 8):** the same sway from the encoders (wrist flange
by forward kinematics, `enc.low_mm`), re-scored on all 740 saved slow_osc
runs: per variant it followed the IMU's change at r = 0.75 and in direction
on 12/12 variants where the IMU moved > 5%. It over-credited dither (−10…−14%
vs IMU 0…−4%), shoulder_2 alone (−14% vs −2%) and one deployed config
(−12% vs +1%, with 3–15 Hz +13%). At a 15% bar none of those pass; the
five that do were all IMU winners (−11…−15%), but the IMU's smaller wins
(−5…−10%) don't show. So without a camera only big improvements are
detectable, and an encoder `better` is a candidate, not proof.
All of this was with the stock gripper and camera on the wrist.

## Rejected — don't re-propose without new evidence

- **Feedback from the wrist camera IMU** (Python and in the realtime core):
  up to −40% vertical, but accel +35–39% — high-frequency oscillation the
  user called "not acceptable"; the camera's mount drifts (gyro vs FK R²
  0.996 → 0.918 overnight); and the fix must work without a camera. Removed.
- **Encoder-only / gyro "tip" damping:** collocated (same as more kd_host);
  1–3 Hz worse at every gain and band.
- **Disturbance observer in the core:** no better than session drift, more
  3–15 Hz (waterbed). Removed.
- **Cogging cancellation:** helped only an elbow-only creep; nothing on
  slow_osc (2.84 vs 2.95 mm, inside noise); per-motor maps made robots tune
  differently. Removed.
- **Command notch** (2.1 / 5.5 Hz): no effect on slow_osc.
- **Learned corrections** (`--learn`, `--learn-imu`): joint error −94%, IMU
  −28…−45% — but ~85% motion-specific; a model predicting them from the
  motion had leave-one-out R² ≈ 0. Research only.
- **Tracking-model inversion** (`--invert`): lag −40…−80%, sway only −5…−8%.
- **Gravity correction from a wrist-mass fit** (wrist_3 +0.11 kg): mean
  errors better, elbow worse; vertical 3.77 vs 3.01.
- **Friction-law changes:** wider `vs` changes nothing; negative `fv` fixes
  30°/s but extrapolates to zero friction at 60–112°/s (needs a runtime
  saturation). A position-indexed friction map: friction ripple doesn't
  repeat by angle (r ≈ 0.02–0.06). LuGre: no better than static.
- **Higher `fc`, stiction 0.3, dither:** noise or worse.
- **Backlash compensation / mechanical changes:** excluded by the user.

## Calibration findings (Oct 7, both arms)

- Friction fit within ~10% at 2–15°/s; low at 1°/s (runtime smooths through
  zero), high at 30°/s (can't fall with speed). The runtime cancels 53–62%
  of 1°/s friction on Stribeck joints, 11–21% where Stribeck is 0.
- Gravity: 7/14 CoM fits rejected by the 60 mm cap — the one-link,
  fixed-mass fit can't absorb a distributed ~5% load error. Harmless with
  `Fo` (≤ 0.3 Nm; left s1 0.74). Proper fix (not built): a joint mass + CoM
  regression over all of an arm's sweeps.
- Old vs new calibration on creeps: elbow new better (−10%), s3 new worse
  (+28%, fixed by Stribeck), others equal.

## A robot in the field (Oct 6–8)

- A left shoulder_1 oscillated during `tune.factory`: the old tools homed on
  0xA4 one joint at a time; reproduced only with the rest of the arm stiff
  on firmware loops (~2 Hz, growing). Fixed by impedance-only calibration.
- Gravity tuned by hand in the panel settings is kept with
  `tune.factory --keep-gravity`.
- Lesson for later reports of "one arm oscillates in teleop": a phone video
  can't tell input jitter from control ringing — record the bus (Phase 1).
