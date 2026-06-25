# Almond Axol v0.1.1

Changes landed since the v0.1.0 release, focused on data-collection reliability and operational tooling.

## New

- **`motor.health` command** — Probes all 16 motors (8 per arm), running the same status reads as `motor.info` on each joint and printing `OK` or the error string per motor. Exits with status `1` if any motor fails to respond, so it works in scripts and provisioning checks. Also available from the web control panel under **Calibrate**.

## Improvements

- **`motor.info` for IDs outside 1–8** — Querying a motor ID outside the known arm range no longer errors out; pass `--type myactuator|damiao` to probe arbitrary IDs (useful when configuring a fresh motor).
- **Jetson setup / power-mode hardening** — `jetson.setup` now reports the real cause when it can't set the MAXN power mode or pin engine/CPU clocks (a genuine command/write failure vs. missing root) instead of always blaming root. It also declines `nvpmodel`'s reboot prompt so the mid-install SSH session and the robot aren't restarted (the mode applies on the next natural reboot), and feeds stdin to prompts so interactive runs don't block.

## Fixes

- **Drop-free episode recording** — Fixed an mp4 DTS crash (`non monotonically increasing dts to muxer`) that lost any episode at index ≥ 1 by re-basing each segment's timestamps when concatenating NVENC mp4 segments. Also fixed a sustained NVENC frame-drop storm during `collect-data` by pipelining the gst stages with queue elements, deepening per-camera feed queues (30 → 90 frames), sampling per-frame image stats at ~10 Hz instead of every frame, and re-partitioning recorder cores (control 3 / relay 1 / dataset 4). Recording now runs drop-free at 960×600@60fps across three cameras with the control loop steady at 120 Hz.
