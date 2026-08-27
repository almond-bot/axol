//! Wire protocol for the two motor vendors on the Axol arm buses, ported
//! bit-for-bit from `almond_axol/motor/myactuator.py` and `damiao.py`.
//!
//! Per arm bus: MyActuator RMD at IDs 0x01-0x05 (shoulder_1..wrist_1),
//! Damiao at IDs 0x06-0x08 (wrist_2, wrist_3, gripper; feedback on 0x10+id).

/// MyActuator request arbitration ID base (request -> 0x140 + motor_id).
pub const MA_REQ: u16 = 0x140;
/// MyActuator response arbitration ID base (response <- 0x240 + motor_id).
pub const MA_RESP: u16 = 0x240;
/// MyActuator motion-control (MIT) request / response bases.
#[allow(dead_code)] // staged for the command path
pub const MA_MC_REQ: u16 = 0x400;
#[allow(dead_code)] // staged for the command path
pub const MA_MC_RESP: u16 = 0x500;

pub const MA_READ_STATUS1: u8 = 0x9A;
pub const MA_READ_VERSION: u8 = 0xB2;
pub const MA_MULTI_TURN_ANGLE: u8 = 0x92;
pub const MA_MOTOR_STATUS_2: u8 = 0x9C;

/// Damiao register access + feedback requests all go to this arbitration ID.
pub const DM_REG_ARB: u16 = 0x7FF;
/// Damiao feedback frames arrive on 0x10 + motor_id.
pub const DM_FEEDBACK_BASE: u16 = 0x10;

pub const DM_REG_CTRL_MODE: u8 = 10;
pub const DM_REG_PMAX: u8 = 21;
pub const DM_REG_VMAX: u8 = 22;
pub const DM_REG_TMAX: u8 = 23;
pub const DM_REG_VBUS: u8 = 60;
/// Registers that decode as uint32 rather than float32.
pub const DM_UINT32_REGS: [u8; 10] = [7, 8, 9, 10, 13, 14, 15, 16, 35, 36];

/// Motor IDs on each arm bus.
pub const MA_IDS: [u8; 5] = [1, 2, 3, 4, 5]; // shoulder_1..wrist_1
pub const DM_IDS: [u8; 3] = [6, 7, 8]; // wrist_2, wrist_3, gripper

pub const JOINT_NAMES: [&str; 8] = [
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
];

// ---------------------------------------------------------------- MyActuator

/// Single-command-byte request frame (`_cmd` in the Python driver).
pub fn ma_cmd(cmd: u8) -> [u8; 8] {
    [cmd, 0, 0, 0, 0, 0, 0, 0]
}

/// Decode a 0xB2 reply: firmware VersionDate (e.g. 2026042402).
pub fn ma_decode_version(data: &[u8; 8]) -> u32 {
    u32::from_le_bytes([data[4], data[5], data[6], data[7]])
}

/// Decode a 0x92 reply: multi-turn angle in radians (0.01 deg/LSB).
pub fn ma_decode_position(data: &[u8; 8]) -> f64 {
    let raw = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    raw as f64 * (0.01 * std::f64::consts::PI / 180.0)
}

/// Decode a 0x9C reply: (temp °C, current A, speed rad/s).
pub fn ma_decode_status2(data: &[u8; 8]) -> (f64, f64, f64) {
    let temp = data[1] as i8 as f64;
    let current = i16::from_le_bytes([data[2], data[3]]) as f64 * 0.01;
    let speed_dps = i16::from_le_bytes([data[4], data[5]]) as f64;
    (temp, current, speed_dps * std::f64::consts::PI / 180.0)
}

/// Decode a 0x9A reply: (bus voltage V, error bitmask).
pub fn ma_decode_status1(data: &[u8; 8]) -> (f64, u16) {
    let volts = u16::from_le_bytes([data[4], data[5]]) as f64 * 0.1;
    let errors = u16::from_le_bytes([data[6], data[7]]);
    (volts, errors)
}

// ------------------------------------------------------------------- Damiao

/// Register-read request (`0x33`) for `motor_id`, sent to 0x7FF.
pub fn dm_read_register(motor_id: u16, rid: u8) -> [u8; 8] {
    let (lo, hi) = ((motor_id & 0xFF) as u8, (motor_id >> 8) as u8);
    [lo, hi, 0x33, rid, 0, 0, 0, 0]
}

/// Feedback request (`0xCC`) for `motor_id`, sent to 0x7FF. The motor answers
/// with a normal feedback frame on its MST_ID (0x10 + id) — read-only.
pub fn dm_request_feedback(motor_id: u16) -> [u8; 8] {
    let (lo, hi) = ((motor_id & 0xFF) as u8, (motor_id >> 8) as u8);
    [lo, hi, 0xCC, 0, 0, 0, 0, 0]
}

/// True when `data` is a register-read reply for (`motor_id`, `rid`).
pub fn dm_is_register_reply(data: &[u8; 8], motor_id: u16, rid: u8) -> bool {
    (data[0] as u16 | ((data[1] as u16) << 8)) == motor_id
        && data[2] == 0x33
        && data[3] == rid
}

/// Decode the value of a register-read reply as f64 (uint32 regs widened).
pub fn dm_decode_register(data: &[u8; 8], rid: u8) -> f64 {
    let raw = [data[4], data[5], data[6], data[7]];
    if DM_UINT32_REGS.contains(&rid) {
        u32::from_le_bytes(raw) as f64
    } else {
        f32::from_le_bytes(raw) as f64
    }
}

/// Decoded Damiao feedback frame.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // velocity/torque consumed by the command path
pub struct DmFeedback {
    pub status: u8,    // frame byte 0 high nibble
    pub position: f64, // rad
    pub velocity: f64, // rad/s
    pub torque: f64,   // Nm
    pub t_mos: f64,    // °C
    pub t_rotor: f64,  // °C
}

/// Decode a Damiao feedback frame against the motor's (p_max, v_max, t_max).
pub fn dm_decode_feedback(data: &[u8; 8], p_max: f64, v_max: f64, t_max: f64) -> DmFeedback {
    let pos_int = ((data[1] as u32) << 8) | data[2] as u32;
    let vel_int = ((data[3] as u32) << 4) | ((data[4] as u32) >> 4);
    let torq_int = (((data[4] & 0xF) as u32) << 8) | data[5] as u32;
    DmFeedback {
        status: data[0] >> 4,
        position: uint_to_float(pos_int, -p_max, p_max, 16),
        velocity: uint_to_float(vel_int, -v_max, v_max, 12),
        torque: uint_to_float(torq_int, -t_max, t_max, 12),
        t_mos: data[6] as f64,
        t_rotor: data[7] as f64,
    }
}

/// MIT-protocol fixed-point decode (identical in both Python drivers).
pub fn uint_to_float(x: u32, x_min: f64, x_max: f64, bits: u32) -> f64 {
    x as f64 * (x_max - x_min) / ((1u32 << bits) - 1) as f64 + x_min
}

/// MIT-protocol fixed-point encode (identical in both Python drivers).
#[allow(dead_code)] // staged for the command path (tested)
pub fn float_to_uint(x: f64, x_min: f64, x_max: f64, bits: u32) -> u32 {
    let x = x.clamp(x_min, x_max);
    ((x - x_min) * ((1u32 << bits) - 1) as f64 / (x_max - x_min)) as u32
}

/// Command ranges an MIT impedance frame is scaled against. Firmware- and
/// motor-dependent: MyActuator legacy vs V4.4 differ in `p_max`/`t_max`
/// (see `mit_ranges` in the Python driver); Damiao reads its ranges from
/// registers 21-23 at enable time.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // staged for the command path (tested)
pub struct MitRanges {
    pub p_max: f64,
    pub v_max: f64,
    pub kp_max: f64,
    pub kd_max: f64,
    pub t_max: f64,
}

/// Encode an MIT impedance command frame — the byte layout is identical for
/// both vendors (`set_impedance` in `myactuator.py`, `_send_cmd` IMPEDANCE
/// branch in `damiao.py`); only the scaling ranges differ.
#[allow(dead_code)] // staged for the command path (tested)
pub fn mit_encode(
    p_des: f64,
    v_des: f64,
    kp: f64,
    kd: f64,
    t_ff: f64,
    r: &MitRanges,
) -> [u8; 8] {
    let p_u = float_to_uint(p_des, -r.p_max, r.p_max, 16);
    let v_u = float_to_uint(v_des, -r.v_max, r.v_max, 12);
    let kp_u = float_to_uint(kp, 0.0, r.kp_max, 12);
    let kd_u = float_to_uint(kd, 0.0, r.kd_max, 12);
    let t_u = float_to_uint(t_ff, -r.t_max, r.t_max, 12);
    [
        (p_u >> 8) as u8,
        p_u as u8,
        (v_u >> 4) as u8,
        (((v_u & 0xF) << 4) | ((kp_u >> 8) & 0xF)) as u8,
        kp_u as u8,
        (kd_u >> 4) as u8,
        (((kd_u & 0xF) << 4) | ((t_u >> 8) & 0xF)) as u8,
        t_u as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector generated by the Python driver's encoder
    /// (`damiao._float_to_uint` + the IMPEDANCE frame layout) for
    /// p=1.2345, v=-0.5, kp=130, kd=3, t_ff=2.75 at Damiao default ranges.
    #[test]
    fn mit_encode_matches_python() {
        let ranges = MitRanges {
            p_max: 12.5,
            v_max: 45.0,
            kp_max: 500.0,
            kd_max: 5.0,
            t_max: 18.0,
        };
        let frame = mit_encode(1.2345, -0.5, 130.0, 3.0, 2.75, &ranges);
        assert_eq!(frame, [140, 163, 126, 132, 40, 153, 153, 56]);
    }

    #[test]
    fn uint_roundtrip() {
        let encoded = float_to_uint(1.2345, -12.5, 12.5, 16);
        assert_eq!(encoded, 36003);
        let decoded = uint_to_float(encoded, -12.5, 12.5, 16);
        assert!((decoded - 1.2345).abs() < 25.0 / 65535.0);
    }
}
