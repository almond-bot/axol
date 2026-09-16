//! Motor bring-up and teardown shared by `hold` and `serve`, mirroring the
//! Python drivers' cold-enable sequences.

use std::io;
use std::time::Duration;

use crate::can::CanSock;
use crate::proto;
use crate::safety::purge_tx_queue;
use crate::txn;

pub const TIMEOUT: Duration = Duration::from_millis(100);
const SEND_TIMEOUT: Duration = Duration::from_millis(20);
/// Post-0x76 reboot settle; the Python driver measures ~1.12 s and waits 2.
pub const RESET_SETTLE: Duration = Duration::from_millis(2200);

/// Static per-joint inputs to bring-up (from a params file or CONFIG).
#[derive(Clone, Debug)]
pub struct MotorSpec {
    pub joint: String,
    pub motor_id: u8,
    /// Gains for the initial hold phase (production config values).
    pub kp: f64,
    pub kd: f64,
    /// The gripper is special-cased throughout: POSITION_FORCE mode instead
    /// of MIT, brought up (enabled + calibrated) by the Python side before
    /// the core arms, exempt from the max-step gate and feedback-health
    /// tracking (stalling against an object is its job).
    pub gripper: bool,
    /// Target-tuple index this motor plays (arm joints 0-6, gripper 7);
    /// decouples the wire layout from bring-up iteration order.
    pub slot: usize,
    /// In-core target-tracker limits (rad/s, rad/s²) for tracked-mode
    /// targets — see `filter::Trapezoid`. Unused by the gripper and `hold`.
    pub max_vel: f64,
    pub max_accel: f64,
    /// Tanh friction-model parameters (`filter::friction`), applied in-core
    /// against the tracker velocity in tracked mode. Zero for the gripper.
    pub fc: f64,
    pub k: f64,
    pub fv: f64,
    pub fo: f64,
    /// MyActuator MIT `(p_max, t_max)` the client detected for this motor
    /// (its own 0xB2/0xB5 reads, with retries, before arming). When present
    /// it is authoritative: the wire is encoded against it whatever this
    /// side's reads say, so one dropped capability reply here can never
    /// silently scale every `t_ff` by the legacy/V4.4 ratio (2.5-5.4x on an
    /// X6/X8). `None` for Damiao motors and the gripper.
    pub mit_ranges: Option<(f64, f64)>,
}

/// Attempts at the MyActuator capability reads (0xB2 version, 0xB5 model)
/// before the bring-up gives up on them.
const CAPABILITY_READ_ATTEMPTS: usize = 3;

#[derive(Clone, Copy, PartialEq)]
pub enum Vendor {
    MyActuator,
    Damiao,
}

/// A motor that passed bring-up prep: identified, fault-free, ranges known.
#[derive(Clone)]
pub struct ReadyMotor {
    pub id: u8,
    pub joint: String,
    pub vendor: Vendor,
    pub ranges: proto::MitRanges,
    /// Measured position at prep time (motor frame, rad).
    pub hold_pos: f64,
    /// Already enabled and holding torque when found (a previous session
    /// died or disconnected while live). Such a motor is *attached to*, not
    /// brought up: no reset, no brake release, no enable — it keeps holding
    /// its pose until the core's hold stream takes over at that same pose.
    pub holding: bool,
    pub kp: f64,
    pub kd: f64,
    pub gripper: bool,
    pub slot: usize,
    /// Tracker limits + friction params, carried over from the spec.
    pub max_vel: f64,
    pub max_accel: f64,
    pub fc: f64,
    pub k: f64,
    pub fv: f64,
    pub fo: f64,
}

/// Status-probe attempts before a silent motor fails the bring-up.
const HOLDING_PROBE_ATTEMPTS: usize = 3;

/// Read-only "enabled and holding torque" probe — the Python drivers'
/// `is_holding`, per vendor.
///
/// Silence is an error, never "not holding": a motor that does not answer
/// cannot be classified, and guessing cold would send a possibly-holding
/// joint the 0x76 reset (the Python `is_holding` likewise raises on a
/// timeout instead of answering). A few attempts ride out a single missed
/// frame; an unpowered arm still fails loudly here, as it did in `prepare`.
pub fn is_holding(sock: &CanSock, spec: &MotorSpec) -> io::Result<bool> {
    for _ in 0..HOLDING_PROBE_ATTEMPTS {
        if spec.motor_id <= 5 {
            let reply = txn::ma_request(
                sock,
                spec.motor_id,
                proto::ma_cmd(proto::MA_READ_STATUS1),
                TIMEOUT,
            )?;
            if let Some((d, _)) = reply {
                return Ok(proto::ma_is_holding(&d));
            }
        } else {
            let reply = txn::dm_request_feedback(sock, spec.motor_id as u16, TIMEOUT)?;
            // Only the status nibble is needed; the ranges do not affect it.
            if let Some((fb, _)) = reply {
                return Ok(fb[0] >> 4 == proto::DM_STATUS_ENABLED);
            }
        }
    }
    Err(io::Error::other(format!(
        "{} (0x{:02X}): no status reply — cannot tell whether it is holding, \
         not resetting it",
        spec.joint, spec.motor_id
    )))
}

/// Phase 1 of a cold bring-up: MyActuator 0x76 system reset (all motors at
/// once, one settle) and Damiao clear-errors. Torque-neutral on a disabled
/// motor. Runs *before* the Python side resolves joint offsets, so the
/// multi-turn wrap state Python verifies is the post-reset one.
///
/// Idempotent per motor, like the Python `Axol.enable()`: a joint found
/// already enabled and holding (a previous session died or disconnected
/// while live) is skipped — the 0x76 reset reboots a MyActuator, dropping
/// torque for ~2 s, so it must never reach a holding joint. A joint whose
/// state cannot be read fails the prep before any reset is sent to it.
/// Returns the joint names that were left holding, for the caller's log.
pub fn prep(sock: &CanSock, specs: &[MotorSpec]) -> io::Result<Vec<String>> {
    let mut any_ma = false;
    let mut held = Vec::new();
    for spec in specs {
        if spec.gripper {
            // The gripper's bring-up (enable, calibration, mode switch) is
            // Python's, and it may be holding an object — never touched here.
            continue;
        }
        if is_holding(sock, spec)? {
            held.push(spec.joint.clone());
            continue;
        }
        if spec.motor_id <= 5 {
            sock.send(
                proto::MA_REQ + spec.motor_id as u16,
                &proto::ma_cmd(proto::MA_RESET),
            )?;
            any_ma = true;
        } else {
            sock.send(spec.motor_id as u16, &proto::DM_CLEAR_ERRORS)?;
        }
    }
    if any_ma {
        std::thread::sleep(RESET_SETTLE);
    }
    sock.drain()?;
    Ok(held)
}

/// Phase 2: capability detection, fault checks, range reads, and position
/// reads. Read-only — the motors stay torque-off until [`enable`].
///
/// `notes` collects non-fatal observations for the caller's log (a
/// capability read that disagreed with the client's, a read that only
/// succeeded on retry).
pub fn prepare(
    sock: &CanSock,
    iface: &str,
    specs: &[MotorSpec],
    notes: &mut Vec<String>,
) -> io::Result<Vec<ReadyMotor>> {
    let err = |msg: String| io::Error::other(format!("{iface}: {msg}"));
    let mut motors = Vec::new();

    for spec in specs.iter().filter(|s| s.motor_id <= 5) {
        let id = spec.motor_id;
        let (p_max, t_max) = ma_ranges_for(sock, spec, notes).map_err(err)?;

        let Some((s1, _)) =
            txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_STATUS1), TIMEOUT)?
        else {
            return Err(err(format!("{} (0x{id:02X}): no status reply", spec.joint)));
        };
        let (_, errors) = proto::ma_decode_status1(&s1);
        if errors != 0 {
            return Err(err(format!(
                "{} (0x{id:02X}): latched fault 0x{errors:04X} — not enabling",
                spec.joint
            )));
        }
        let holding = proto::ma_is_holding(&s1);
        let Some((pos_frame, _)) =
            txn::ma_request(sock, id, proto::ma_cmd(proto::MA_MULTI_TURN_ANGLE), TIMEOUT)?
        else {
            return Err(err(format!(
                "{} (0x{id:02X}): no position reply",
                spec.joint
            )));
        };
        motors.push(ReadyMotor {
            id,
            joint: spec.joint.clone(),
            vendor: Vendor::MyActuator,
            ranges: proto::MitRanges {
                p_max,
                v_max: proto::MA_V_MAX,
                kp_max: proto::MA_KP_MAX,
                kd_max: proto::MA_KD_MAX,
                t_max,
            },
            hold_pos: proto::ma_decode_position(&pos_frame),
            holding,
            kp: spec.kp,
            kd: spec.kd,
            gripper: false,
            slot: spec.slot,
            max_vel: spec.max_vel,
            max_accel: spec.max_accel,
            fc: spec.fc,
            k: spec.k,
            fv: spec.fv,
            fo: spec.fo,
        });
    }

    for spec in specs.iter().filter(|s| s.motor_id >= 6) {
        let id = spec.motor_id as u16;
        let mode = read_dm_register(sock, id, proto::DM_REG_CTRL_MODE)?;
        // Wrists run MIT (1); the gripper must already be in POSITION_FORCE
        // (4), set by the Python side's calibration flow before arming.
        let expected = if spec.gripper { 4.0 } else { 1.0 };
        if mode != expected {
            return Err(err(format!(
                "{} (0x{id:02X}): control mode {mode} (expected {expected}) — not enabling",
                spec.joint
            )));
        }
        let p_max = read_dm_register(sock, id, proto::DM_REG_PMAX)?;
        let v_max = read_dm_register(sock, id, proto::DM_REG_VMAX)?;
        let t_max = read_dm_register(sock, id, proto::DM_REG_TMAX)?;

        let Some((fb, _)) = txn::dm_request_feedback(sock, id, TIMEOUT)? else {
            return Err(err(format!(
                "{} (0x{id:02X}): no feedback reply",
                spec.joint
            )));
        };
        let decoded = proto::dm_decode_feedback(&fb, p_max, v_max, t_max);
        motors.push(ReadyMotor {
            id: spec.motor_id,
            joint: spec.joint.clone(),
            vendor: Vendor::Damiao,
            ranges: proto::MitRanges {
                p_max,
                v_max,
                kp_max: 500.0,
                kd_max: 5.0,
                t_max,
            },
            hold_pos: decoded.position,
            holding: decoded.status == proto::DM_STATUS_ENABLED,
            kp: spec.kp,
            kd: spec.kd,
            gripper: spec.gripper,
            slot: spec.slot,
            max_vel: spec.max_vel,
            max_accel: spec.max_accel,
            fc: spec.fc,
            k: spec.k,
            fv: spec.fv,
            fo: spec.fo,
        });
    }
    Ok(motors)
}

/// The MIT `(p_max, t_max)` a MyActuator's wire frames are encoded against.
///
/// The client's detected ranges (`spec.mit_ranges`) win when present; this
/// side's own 0xB2/0xB5 reads then only cross-check them, and a
/// disagreement is reported through `notes` rather than acted on (the
/// client read with retries on a quiet bus; both cannot be right, and
/// arming against the client's value keeps Python's feedback decode and the
/// wire consistent with each other). Without shipped ranges the own reads
/// are authoritative and are retried; a motor that never answers them is a
/// bring-up error, exactly like one that never answers the status or
/// position read — never a silent fall-back to the legacy ranges.
fn ma_ranges_for(
    sock: &CanSock,
    spec: &MotorSpec,
    notes: &mut Vec<String>,
) -> Result<(f64, f64), String> {
    let id = spec.motor_id;
    let mut version = None;
    let mut model = None;
    let mut attempts = 0;
    while attempts < CAPABILITY_READ_ATTEMPTS && (version.is_none() || model.is_none()) {
        attempts += 1;
        if version.is_none() {
            version = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_VERSION), TIMEOUT)
                .map_err(|e| e.to_string())?
                .map(|(d, _)| proto::ma_decode_version(&d));
        }
        if model.is_none() {
            model = read_ma_model(sock, id).map_err(|e| e.to_string())?;
        }
    }
    let own = match (version, model.as_deref()) {
        (Some(v), Some(m)) => Some(proto::ma_mit_ranges(Some(v), Some(m))),
        _ => None,
    };
    if attempts > 1 && own.is_some() {
        notes.push(format!(
            "{} (0x{id:02X}): capability read succeeded on attempt {attempts}",
            spec.joint
        ));
    }
    let label = format!("{} (0x{id:02X})", spec.joint);
    let detail = format!(
        "firmware {}, model {:?}",
        version.map_or("?".to_string(), |v| v.to_string()),
        model.as_deref().unwrap_or("?"),
    );
    let (ranges, note) = resolve_ma_ranges(&label, spec.mit_ranges, own, &detail)?;
    notes.extend(note);
    Ok(ranges)
}

/// The range decision, separated from the bus reads so it can be tested:
/// `shipped` is the client's detection, `own` this side's (both `None` when
/// unavailable). Returns the ranges to arm against plus an optional note.
fn resolve_ma_ranges(
    label: &str,
    shipped: Option<(f64, f64)>,
    own: Option<(f64, f64)>,
    detail: &str,
) -> Result<((f64, f64), Option<String>), String> {
    match (shipped, own) {
        (Some(shipped), Some(read)) => {
            let note = if (shipped.0 - read.0).abs() > 1e-6 || (shipped.1 - read.1).abs() > 1e-6 {
                Some(format!(
                    "{label}: MIT ranges disagree — client detected p_max {} / t_max {}, this \
                     read gives {} / {} ({detail}); arming against the client's",
                    shipped.0, shipped.1, read.0, read.1,
                ))
            } else {
                None
            };
            Ok((shipped, note))
        }
        (Some(shipped), None) => Ok((
            shipped,
            Some(format!(
                "{label}: no capability reply in {CAPABILITY_READ_ATTEMPTS} attempts; arming \
                 against the client's detected ranges (p_max {} / t_max {})",
                shipped.0, shipped.1
            )),
        )),
        (None, Some(read)) => Ok((read, None)),
        (None, None) => Err(format!(
            "{label}: no firmware version / model reply in {CAPABILITY_READ_ATTEMPTS} \
             attempts — cannot tell which MIT ranges the motor decodes against, not enabling"
        )),
    }
}

pub fn read_ma_model(sock: &CanSock, motor_id: u8) -> io::Result<Option<String>> {
    let mut raw = Vec::new();
    for block in [0x01u8, 0x02] {
        let req = [proto::MA_READ_MODEL, 0x01, block, 0, 0, 0, 0, 0];
        match txn::ma_request(sock, motor_id, req, TIMEOUT)? {
            Some((d, _)) => raw.extend_from_slice(&proto::ma_decode_model_block(&d)),
            None => return Ok(None),
        }
    }
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    Ok(Some(String::from_utf8_lossy(&raw[..end]).into_owned()))
}

/// Register read with the Python driver's retry (single unacked reply frame).
pub fn read_dm_register(sock: &CanSock, motor_id: u16, rid: u8) -> io::Result<f64> {
    for _ in 0..5 {
        if let Some((value, _)) = txn::dm_read_register(sock, motor_id, rid, TIMEOUT)? {
            return Ok(value);
        }
    }
    Err(io::Error::other(format!(
        "damiao 0x{motor_id:02X}: register {rid} read timed out"
    )))
}

/// Enable every cold motor. Motors found holding by [`prepare`] are left
/// exactly as they are (no brake release / enable frame), and are excluded
/// from the rollback: a failed cold bring-up must never drop a pre-existing
/// hold, only the motors this call itself touched.
pub fn enable(sock: &CanSock, iface: &str, motors: &[ReadyMotor]) -> io::Result<()> {
    // A full socket buffer must not strand motors that were enabled earlier in
    // this batch or prevent the rollback below from running to completion.
    sock.set_send_timeout(SEND_TIMEOUT)?;
    // Record a motor before its enable write: a failed request may still have
    // reached the controller, even when its acknowledgement did not return.
    let mut attempted = Vec::new();
    for m in motors {
        if m.gripper || m.holding {
            // Gripper: enabled by the Python side's calibration flow.
            // Holding: attached to, keeps its pose until the hold stream.
            continue;
        }
        attempted.push(m.clone());
        let result = match m.vendor {
            Vendor::MyActuator => {
                match txn::ma_request(sock, m.id, proto::ma_cmd(proto::MA_RELEASE_BRAKE), TIMEOUT) {
                    Ok(Some(_)) => Ok(()),
                    Ok(None) => Err(io::Error::other(format!(
                        "{}: brake release not acknowledged",
                        m.joint
                    ))),
                    Err(err) => Err(err),
                }
            }
            Vendor::Damiao => sock.send(m.id as u16, &proto::DM_ENABLE),
        };
        if let Err(err) = result {
            if !disable_inner(sock, &attempted) {
                let purged = purge_tx_queue(iface);
                eprintln!(
                    "{iface}: motor-enable rollback had failed writes{}",
                    if purged {
                        "; stale TX queue purged"
                    } else {
                        "; QUEUE PURGE FAILED, flap the interface before re-powering"
                    }
                );
            }
            return Err(err);
        }
    }
    Ok(())
}

/// Best-effort disable of every motor; Damiao gets the repeated-send
/// treatment the Python driver uses.
pub fn disable(sock: &CanSock, motors: &[ReadyMotor]) {
    let _ = disable_inner(sock, motors);
}

/// Return false if any rollback frame could not be written. Callers performing
/// partial-enable rollback use that signal to purge potentially stale enables.
fn disable_inner(sock: &CanSock, motors: &[ReadyMotor]) -> bool {
    let mut complete = true;
    for m in motors {
        for _ in 0..3 {
            let sent = match m.vendor {
                Vendor::MyActuator => sock.send(
                    proto::MA_REQ + m.id as u16,
                    &proto::ma_cmd(proto::MA_SHUTDOWN),
                ),
                Vendor::Damiao => sock.send(m.id as u16, &proto::DM_DISABLE),
            };
            if let Err(err) = sent {
                complete = false;
                eprintln!("disable {}: {err}", m.joint);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    complete
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The failure this guards against: a dropped 0xB2/0xB5 reply used to
    /// select the legacy (12.5, 24) ranges silently, scaling every t_ff on a
    /// V4.4 X6/X8 by 2.5x/5.4x for the session.
    #[test]
    fn range_resolution_never_falls_back_silently() {
        let v44_x8 = (proto::MA_P_MAX_V44, 129.0);
        let legacy = (proto::MA_P_MAX_LEGACY, proto::MA_T_MAX_LEGACY);
        // Client and core agree: no note.
        let (r, note) = resolve_ma_ranges("s1", Some(v44_x8), Some(v44_x8), "").unwrap();
        assert_eq!(r, v44_x8);
        assert!(note.is_none());
        // Core's read missing: the client's ranges are used, with a note.
        let (r, note) = resolve_ma_ranges("s1", Some(v44_x8), None, "").unwrap();
        assert_eq!(r, v44_x8);
        assert!(note.unwrap().contains("client's detected ranges"));
        // Core's read disagrees (it saw legacy): the client's win, loudly.
        let (r, note) = resolve_ma_ranges("s1", Some(v44_x8), Some(legacy), "fw ?").unwrap();
        assert_eq!(r, v44_x8);
        let note = note.unwrap();
        assert!(note.contains("disagree"), "{note}");
        assert!(note.contains("129"), "{note}");
        assert!(note.contains("24"), "{note}");
        // No client detection (hold tool): the core's own read is used.
        let (r, note) = resolve_ma_ranges("s1", None, Some(v44_x8), "").unwrap();
        assert_eq!(r, v44_x8);
        assert!(note.is_none());
        // Neither: refuse to arm rather than guess.
        let err = resolve_ma_ranges("s1", None, None, "").unwrap_err();
        assert!(err.contains("not enabling"), "{err}");
    }
}
