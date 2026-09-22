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
    /// Error-sign stiction compensation (`filter::stiction`): gain as a
    /// fraction of `fc`, and the error (rad) it saturates at. Zero gain is
    /// the production law; zero for the gripper.
    pub stiction_gain: f64,
    pub stiction_err: f64,
    /// Load-proportional stiction push, Nm per Nm of gravity feedforward.
    pub stiction_load_gain: f64,
    /// Torque dither (`filter::dither_step`): peak Nm (0 off) and frequency.
    pub dither_nm: f64,
    pub dither_hz: f64,
    /// Command frame for tracked ticks (MyActuator joints only).
    pub wire: WireMode,
    /// Stribeck cancellation on measured velocity (`filter::stribeck_excess`):
    /// gain, zero-load excess (Nm), excess per Nm of gravity, 1/e speed.
    pub stribeck_gain: f64,
    pub stribeck_dfs: f64,
    pub stribeck_load_gain: f64,
    pub stribeck_vs: f64,
    /// Load-proportional Coulomb friction, Nm per Nm of gravity feedforward:
    /// the tracked-mode friction term uses `fc + fl·|t_ff|`.
    pub fl: f64,
    /// Low-pass pole (rad/s) of the measured velocity the Stribeck term
    /// follows; `<= 0` falls back to the control derivative pole.
    pub stribeck_pole: f64,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Vendor {
    MyActuator,
    Damiao,
}

/// Which frame a MyActuator arm joint is commanded with in tracked mode.
/// Damiao joints, the gripper, and every limp / gravity-comp tick (kp = 0)
/// use MIT regardless. An a4 joint's *holds* (bring-up, stalled stream) are
/// 0xA4 too: the X6-P20's 2025070202 firmware ignores 0xA4 after an MIT
/// frame until reset, so an a4 joint must see the position frame from its
/// first tick (see `serve::a4_wire`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WireMode {
    /// The 0x400 impedance frame: the production control law.
    Mit,
    /// 0xA4 absolute position closed-loop: the firmware's own position PI
    /// (and speed PI beneath it, at its kHz rate) tracks the streamed target
    /// under a speed cap. No host feedforward reaches the motor; gravity and
    /// friction are the firmware integrator's job. Paired with a 0x92 read
    /// per tick for 0.01° position; the reply's torque channel is iq in
    /// amps, so measured torque is reported as NaN on these joints.
    A4,
    /// Damiao position-velocity mode (0x100 + id, control-mode register 2):
    /// the wrist firmware's own position → speed cascade (KP_APR/KP_ASR
    /// registers, ACC/DEC ramps) tracks the streamed target under a speed
    /// cap. The feedback frame is the MIT one, so position, velocity and
    /// torque all come back with each command. Like `A4`, no host
    /// feedforward reaches the motor.
    Pv,
}

impl WireMode {
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "mit" => Some(Self::Mit),
            "a4" => Some(Self::A4),
            "pv" => Some(Self::Pv),
            _ => None,
        }
    }

    /// The Damiao control-mode register value a wrist must be in for this
    /// wire's command frame to be acted on.
    pub fn dm_mode(self) -> u32 {
        match self {
            Self::Pv => proto::DM_MODE_POS_VEL,
            Self::Mit | Self::A4 => proto::DM_MODE_MIT,
        }
    }
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
    pub stiction_gain: f64,
    pub stiction_err: f64,
    pub stiction_load_gain: f64,
    pub dither_nm: f64,
    pub dither_hz: f64,
    pub wire: WireMode,
    pub stribeck_gain: f64,
    pub stribeck_dfs: f64,
    pub stribeck_load_gain: f64,
    pub stribeck_vs: f64,
    pub fl: f64,
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
pub fn prepare(sock: &CanSock, iface: &str, specs: &[MotorSpec]) -> io::Result<Vec<ReadyMotor>> {
    let err = |msg: String| io::Error::other(format!("{iface}: {msg}"));
    let mut motors = Vec::new();

    for spec in specs.iter().filter(|s| s.motor_id <= 5) {
        let id = spec.motor_id;
        let version = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_VERSION), TIMEOUT)?
            .map(|(d, _)| proto::ma_decode_version(&d));
        let model = read_ma_model(sock, id)?;
        let (p_max, t_max) = proto::ma_mit_ranges(version, model.as_deref());

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
            stiction_gain: spec.stiction_gain,
            stiction_err: spec.stiction_err,
            stiction_load_gain: spec.stiction_load_gain,
            dither_nm: spec.dither_nm,
            dither_hz: spec.dither_hz,
            wire: spec.wire,
            stribeck_gain: spec.stribeck_gain,
            stribeck_dfs: spec.stribeck_dfs,
            stribeck_load_gain: spec.stribeck_load_gain,
            stribeck_vs: spec.stribeck_vs,
            fl: spec.fl,
        });
    }

    for spec in specs.iter().filter(|s| s.motor_id >= 6) {
        let id = spec.motor_id as u16;
        let mode = read_dm_register(sock, id, proto::DM_REG_CTRL_MODE)?;
        if spec.gripper {
            // The gripper must already be in POSITION_FORCE (4), set by the
            // Python side's calibration flow before arming.
            let expected = proto::DM_MODE_POS_FORCE as f64;
            if mode != expected {
                return Err(err(format!(
                    "{} (0x{id:02X}): control mode {mode} (expected {expected}) — not enabling",
                    spec.joint
                )));
            }
        } else {
            // A wrist runs in the mode its wire wants — MIT (1) or, for
            // `wire_mode pv`, position-velocity (2). Put it there (RAM
            // write, effective at once) rather than refusing: a wrist left
            // in the other mode by the previous session is the normal case
            // when the controller choice changes between runs.
            let wanted = spec.wire.dm_mode();
            if mode != wanted as f64 {
                write_dm_register(sock, id, proto::DM_REG_CTRL_MODE, wanted.to_le_bytes())?;
                let now = read_dm_register(sock, id, proto::DM_REG_CTRL_MODE)?;
                if now != wanted as f64 {
                    return Err(err(format!(
                        "{} (0x{id:02X}): control mode {now} after asking for {wanted} — not enabling",
                        spec.joint
                    )));
                }
            }
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
            stiction_gain: spec.stiction_gain,
            stiction_err: spec.stiction_err,
            stiction_load_gain: spec.stiction_load_gain,
            dither_nm: spec.dither_nm,
            dither_hz: spec.dither_hz,
            wire: spec.wire,
            stribeck_gain: spec.stribeck_gain,
            stribeck_dfs: spec.stribeck_dfs,
            stribeck_load_gain: spec.stribeck_load_gain,
            stribeck_vs: spec.stribeck_vs,
            fl: spec.fl,
        });
    }
    Ok(motors)
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

/// Register write (RAM, 0x55). The motor does not acknowledge; callers
/// read the register back. A short settle lets the firmware apply it before
/// the readback.
pub fn write_dm_register(sock: &CanSock, motor_id: u16, rid: u8, value: [u8; 4]) -> io::Result<()> {
    sock.send(
        proto::DM_REG_ARB,
        &proto::dm_write_register(motor_id, rid, value),
    )?;
    std::thread::sleep(Duration::from_millis(5));
    Ok(())
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
