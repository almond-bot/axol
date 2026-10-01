//! In-core tip damping: the wrist IMU's vertical velocity against a reference
//! tool height, fed back as joint torque through the height Jacobian.
//!
//! The Python original (`almond_axol.tuning.imu_damping.TipDamper`, run by
//! `tune.motion --imu-damp`) closes this loop from the 240 Hz Python motion
//! loop: an IMU sample waits for that loop's poll, its torque for the next
//! target to reach the core. On jelly the loop's 15-25 ms round trip is what
//! capped the gain — the damping's phase wraps near 7-8 Hz on shoulder_1 and
//! ~12 Hz on the elbow, where it starts to drive the modes it should damp.
//! Here the IMU's datagrams come straight to the bus thread and the torque
//! joins the joint's feedforward on the tick it is computed.
//!
//! Every piece mirrors its Python original sample for sample (the golden
//! tests at the bottom pin them): [`VerticalVelocity`], [`HeightVelocity`]
//! (Python's `EncoderVelocity`), [`TrackingFilter`], the lead-lag, notch and
//! per-joint low-pass, and [`TipDamper`]'s force law, clamp, ramp, stale
//! cut-off and runaway trip. [`PoeChain`] is the arm's forward kinematics as a
//! product of exponentials — exported by Python from the same URDF model the
//! solver uses — for the tool height and its Jacobian.
//!
//! Nothing here allocates after construction: the delay buffers are sized at
//! configure time and only rotate.

use std::collections::VecDeque;
use std::f64::consts::PI;

pub const G: f64 = 9.80665;

/// Second-order Butterworth high-pass biquad (bilinear, prewarped) —
/// `_butter_hp2` in the Python original. Returns `(b, a)`, `a[0] = 1`.
pub fn butter_hp2(fc: f64, fs: f64) -> ([f64; 3], [f64; 3]) {
    let k = (PI * fc / fs).tan();
    let q = 1.0 / 2f64.sqrt();
    let norm = 1.0 / (1.0 + k / q + k * k);
    let b = [norm, -2.0 * norm, norm];
    let a = [
        1.0,
        2.0 * (k * k - 1.0) * norm,
        (1.0 - k / q + k * k) * norm,
    ];
    (b, a)
}

/// Band-passed vertical velocity (m/s, up positive) of the wrist camera
/// from its accelerometer (m/s², gravity included) and gyro (deg/s, camera
/// frame): gravity carried through the camera's rotation and relaxed toward
/// the accelerometer, the vertical acceleration less g leak-integrated,
/// high-passed and low-passed. Mirrors `VerticalVelocity`.
pub struct VerticalVelocity {
    hp_hz: f64,
    lp_hz: f64,
    up_tau_s: f64,
    leak_hz: f64,
    b: [f64; 3],
    a: [f64; 3],
    up: Option<[f64; 3]>,
    t: Option<f64>,
    vi: f64,
    hx: [f64; 2],
    hy: [f64; 2],
    lp: f64,
    pub value: f64,
    // The vertical acceleration above HF_HZ, as a mean square over HF_TAU_S
    // (`hf_rms`): where a damper whose loop phase has wrapped drives the arm.
    hf_x: f64,
    hf_y: f64,
    hf_ms: f64,
}

const HF_HZ: f64 = 7.0;
const HF_TAU_S: f64 = 0.3;

impl VerticalVelocity {
    /// RMS vertical acceleration above 7 Hz (m/s²).
    pub fn hf_rms(&self) -> f64 {
        self.hf_ms.sqrt()
    }

    pub fn new(hp_hz: f64, lp_hz: f64) -> Self {
        let (b, a) = butter_hp2(hp_hz, 200.0);
        Self {
            hp_hz,
            lp_hz,
            up_tau_s: 2.0,
            leak_hz: 0.2,
            b,
            a,
            up: None,
            t: None,
            vi: 0.0,
            hx: [0.0; 2],
            hy: [0.0; 2],
            lp: 0.0,
            value: 0.0,
            hf_x: 0.0,
            hf_y: 0.0,
            hf_ms: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.hp_hz, self.lp_hz);
    }

    pub fn update(&mut self, t: f64, acc: [f64; 3], gyro_deg_s: [f64; 3]) -> f64 {
        let (Some(mut up), Some(t_prev)) = (self.up, self.t) else {
            self.up = Some(acc);
            self.t = Some(t);
            return self.value;
        };
        let dt = t - t_prev;
        self.t = Some(t);
        if !(0.0 < dt && dt < 0.1) {
            return self.value;
        }
        let w = gyro_deg_s.map(f64::to_radians);
        // A fixed world vector seen from a frame rotating at ω turns at −ω.
        let c = cross(w, up);
        for i in 0..3 {
            up[i] -= c[i] * dt;
        }
        let relax = (dt / self.up_tau_s).min(1.0);
        for i in 0..3 {
            up[i] += (acc[i] - up[i]) * relax;
        }
        self.up = Some(up);
        let mut g = norm(up);
        if g == 0.0 {
            g = G;
        }
        let a_v = dot(acc, up) / g - g;
        let k = 1.0 / (1.0 + 2.0 * PI * HF_HZ * dt);
        self.hf_y = k * (self.hf_y + a_v - self.hf_x);
        self.hf_x = a_v;
        self.hf_ms += (self.hf_y * self.hf_y - self.hf_ms) * (dt / HF_TAU_S).min(1.0);
        self.vi += (a_v - 2.0 * PI * self.leak_hz * self.vi) * dt;
        let (b, a) = (self.b, self.a);
        let y = b[0] * self.vi + b[1] * self.hx[0] + b[2] * self.hx[1]
            - a[1] * self.hy[0]
            - a[2] * self.hy[1];
        self.hx = [self.vi, self.hx[0]];
        self.hy = [y, self.hy[0]];
        let alpha = 1.0 - (-2.0 * PI * self.lp_hz * dt).exp();
        self.lp += (y - self.lp) * alpha;
        self.value = self.lp;
        self.value
    }
}

/// The band velocity of a height signal (m) through the same chain as
/// [`VerticalVelocity`] — derivative, the leak as a first-order high-pass,
/// the high-pass biquad (designed at 240 Hz) and the low-pass — held back by
/// `delay_s` so it lines up with the IMU's latency. Python's
/// `EncoderVelocity`.
pub struct HeightVelocity {
    hp_hz: f64,
    lp_hz: f64,
    leak_hz: f64,
    pub delay_s: f64,
    b: [f64; 3],
    a: [f64; 3],
    buf: VecDeque<(f64, f64)>,
    t: Option<f64>,
    z: f64,
    hp1: f64,
    v_prev: f64,
    hx: [f64; 2],
    hy: [f64; 2],
    lp: f64,
    pub value: f64,
}

/// Samples a height chain may hold back (2 s at 240 Hz — far past any delay).
const HEIGHT_BUF: usize = 512;

impl HeightVelocity {
    pub fn new(hp_hz: f64, lp_hz: f64, delay_s: f64) -> Self {
        let (b, a) = butter_hp2(hp_hz, 240.0);
        Self {
            hp_hz,
            lp_hz,
            leak_hz: 0.2,
            delay_s,
            b,
            a,
            buf: VecDeque::with_capacity(HEIGHT_BUF),
            t: None,
            z: 0.0,
            hp1: 0.0,
            v_prev: 0.0,
            hx: [0.0; 2],
            hy: [0.0; 2],
            lp: 0.0,
            value: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.t = None;
        self.z = 0.0;
        self.hp1 = 0.0;
        self.v_prev = 0.0;
        self.hx = [0.0; 2];
        self.hy = [0.0; 2];
        self.lp = 0.0;
        self.value = 0.0;
        let (b, a) = butter_hp2(self.hp_hz, 240.0);
        self.b = b;
        self.a = a;
    }

    pub fn update(&mut self, t: f64, z: f64) -> f64 {
        if self.buf.len() == HEIGHT_BUF {
            // Never grow: a stalled release drops the oldest sample.
            self.buf.pop_front();
        }
        self.buf.push_back((t, z));
        while self.buf.len() > 1 && self.buf[1].0 <= t - self.delay_s {
            let (ts, zs) = self.buf.pop_front().unwrap();
            self.step(ts, zs);
        }
        self.value
    }

    fn step(&mut self, t: f64, z: f64) {
        let Some(t_prev) = self.t else {
            self.t = Some(t);
            self.z = z;
            return;
        };
        let dt = t - t_prev;
        if !(0.0 < dt && dt < 0.1) {
            self.t = Some(t);
            self.z = z;
            return;
        }
        let v = (z - self.z) / dt;
        self.t = Some(t);
        self.z = z;
        let a1 = 1.0 / (1.0 + 2.0 * PI * self.leak_hz * dt);
        self.hp1 = a1 * (self.hp1 + v - self.v_prev);
        self.v_prev = v;
        let (b, a) = (self.b, self.a);
        let x = self.hp1;
        let y = b[0] * x + b[1] * self.hx[0] + b[2] * self.hx[1]
            - a[1] * self.hy[0]
            - a[2] * self.hy[1];
        self.hx = [x, self.hx[0]];
        self.hy = [y, self.hy[0]];
        let alpha = 1.0 - (-2.0 * PI * self.lp_hz * dt).exp();
        self.lp += (y - self.lp) * alpha;
        self.value = self.lp;
    }
}

/// A joint's measured tracking dynamics, `K (1 + s/ωz) ωn² / (s² + 2ζωn s +
/// ωn²) e^{−sτ}`, run causally at `fs` with the DC gain pinned to 1 — the
/// position the joint is expected to reach given its command. Python's
/// `TrackingFilter` (Tustin; the delay in whole samples; at rest at the
/// first sample).
pub struct TrackingFilter {
    b: [f64; 3],
    a: [f64; 3],
    delay: usize,
    buf: VecDeque<f64>,
    x: [f64; 2],
    y: [f64; 2],
    started: bool,
}

impl TrackingFilter {
    /// `wz = None` is a model without its zero.
    pub fn new(wn: f64, zeta: f64, wz: Option<f64>, tau: f64, fs: f64) -> Self {
        let c = 2.0 * fs;
        let r = wz.map_or(0.0, |wz| c / wz);
        let b = [wn * wn * (r + 1.0), wn * wn * 2.0, wn * wn * (1.0 - r)];
        let a = [
            c * c + 2.0 * zeta * wn * c + wn * wn,
            -2.0 * c * c + 2.0 * wn * wn,
            c * c - 2.0 * zeta * wn * c + wn * wn,
        ];
        let delay = (tau * fs).round().max(0.0) as usize;
        Self {
            b: [b[0] / a[0], b[1] / a[0], b[2] / a[0]],
            a: [1.0, a[1] / a[0], a[2] / a[0]],
            delay,
            buf: VecDeque::with_capacity(delay + 1),
            x: [0.0; 2],
            y: [0.0; 2],
            started: false,
        }
    }

    pub fn reset(&mut self) {
        self.started = false;
        self.buf.clear();
    }

    pub fn step(&mut self, x: f64) -> f64 {
        if !self.started {
            let gain = (self.b[0] + self.b[1] + self.b[2]) / (self.a[0] + self.a[1] + self.a[2]);
            self.x = [x, x];
            self.y = [x * gain, x * gain];
            self.buf.clear();
            for _ in 0..self.delay {
                self.buf.push_back(x);
            }
            self.started = true;
        }
        let x = if self.delay > 0 {
            self.buf.push_back(x);
            self.buf.pop_front().unwrap()
        } else {
            x
        };
        let (b, a) = (self.b, self.a);
        let y =
            b[0] * x + b[1] * self.x[0] + b[2] * self.x[1] - a[1] * self.y[0] - a[2] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }
}

/// The arm's forward kinematics as a product of exponentials: joint `i`
/// rotates about the unit axis `w[i]` through the point `r[i]`, both in the
/// world frame at zero joint angles, and `m` is the gripper mount's pose
/// there. Exported by Python (`almond_axol.rt.tipdamp.poe_chain`) from the
/// solver's own URDF model and checked against its FK.
#[derive(Clone)]
pub struct PoeChain {
    pub w: [[f64; 3]; 7],
    pub r: [[f64; 3]; 7],
    pub m: [[f64; 4]; 4],
}

impl PoeChain {
    /// The mount's height (m) at joint-frame `q`, and ∂height/∂q_i (m/rad).
    pub fn height_and_jz(&self, q: &[f64; 7]) -> (f64, [f64; 7]) {
        let mut g = IDENT;
        let mut axes = [[0.0; 3]; 7];
        let mut points = [[0.0; 3]; 7];
        for i in 0..7 {
            axes[i] = rot_vec(&g, self.w[i]);
            points[i] = xform_point(&g, self.r[i]);
            g = mul(&g, &screw_exp(self.w[i], self.r[i], q[i]));
        }
        let t = mul(&g, &self.m);
        let p = [t[0][3], t[1][3], t[2][3]];
        let mut jz = [0.0; 7];
        for i in 0..7 {
            let lever = [
                p[0] - points[i][0],
                p[1] - points[i][1],
                p[2] - points[i][2],
            ];
            jz[i] = cross(axes[i], lever)[2];
        }
        (p[2], jz)
    }

    pub fn height(&self, q: &[f64; 7]) -> f64 {
        self.height_and_jz(q).0
    }
}

const IDENT: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn screw_exp(w: [f64; 3], r: [f64; 3], th: f64) -> [[f64; 4]; 4] {
    let (s, c) = th.sin_cos();
    let v = 1.0 - c;
    let [x, y, z] = w;
    let rot = [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ];
    // Rotation about the axis through r: p ↦ R(p − r) + r.
    let mut t = IDENT;
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = rot[i][j];
        }
        t[i][3] = r[i] - (rot[i][0] * r[0] + rot[i][1] * r[1] + rot[i][2] * r[2]);
    }
    t
}

fn mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn rot_vec(t: &[[f64; 4]; 4], v: [f64; 3]) -> [f64; 3] {
    [
        t[0][0] * v[0] + t[0][1] * v[1] + t[0][2] * v[2],
        t[1][0] * v[0] + t[1][1] * v[1] + t[1][2] * v[2],
        t[2][0] * v[0] + t[2][1] * v[1] + t[2][2] * v[2],
    ]
}

fn xform_point(t: &[[f64; 4]; 4], p: [f64; 3]) -> [f64; 3] {
    let r = rot_vec(t, p);
    [r[0] + t[0][3], r[1] + t[1][3], r[2] + t[2][3]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// What the damper measures the tool's velocity against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Reference {
    /// FK of the measured joints: only the flex past the encoders.
    Encoder,
    /// The commanded joints: the whole deviation from the path.
    Command,
    /// The commanded joints through their tracking models: the deviation
    /// from the path the arm is expected to follow.
    Model,
}

/// A damping joint: its arm slot (0-6), gain scale and optional low-pass.
#[derive(Clone, Copy, Debug)]
pub struct TipColumn {
    pub slot: usize,
    pub weight: f64,
    pub lp_hz: f64,
}

/// Everything the config line carries.
#[derive(Clone, Debug)]
pub struct TipConfig {
    pub gain: f64,
    pub hp_hz: f64,
    pub lp_hz: f64,
    pub lead_hz: f64,
    pub notch_hz: f64,
    pub notch_q: f64,
    pub max_torque: f64,
    pub reference: Reference,
    pub delay_s: f64,
    pub ramp_s: f64,
    pub stale_s: f64,
    pub trip_speed: f64,
    pub trip_s: f64,
    /// High-band vertical acceleration RMS (m/s²) that trips it; 0 = off.
    pub trip_hf_acc: f64,
    pub columns: Vec<TipColumn>,
}

/// `TipDamper`'s torque law with `--imu-damp-ref`'s reference, in-core.
pub struct TipDamper {
    pub cfg: TipConfig,
    pub imu: VerticalVelocity,
    pub height: HeightVelocity,
    pub tripped: bool,
    started: Option<f64>,
    ramp_from: Option<f64>,
    fast_since: Option<f64>,
    last_sample: Option<f64>,
    lead_t: Option<f64>,
    lead_in: f64,
    lead_out: f64,
    nx: [f64; 2],
    ny: [f64; 2],
    clp: [f64; 7],
}

impl TipDamper {
    pub fn new(cfg: TipConfig) -> Self {
        let imu = VerticalVelocity::new(cfg.hp_hz, cfg.lp_hz);
        // The reference waits out the IMU's latency, whichever it is.
        let height = HeightVelocity::new(cfg.hp_hz, cfg.lp_hz, cfg.delay_s);
        Self {
            cfg,
            imu,
            height,
            tripped: false,
            started: None,
            ramp_from: None,
            fast_since: None,
            last_sample: None,
            lead_t: None,
            lead_in: 0.0,
            lead_out: 0.0,
            nx: [0.0; 2],
            ny: [0.0; 2],
            clp: [0.0; 7],
        }
    }

    /// The band tool velocity the reference does not account for (m/s).
    pub fn flex(&self) -> f64 {
        self.imu.value - self.height.value
    }

    /// Arm the damper (a pass begins): every filter starts over.
    pub fn start(&mut self, now: f64) {
        self.imu.reset();
        self.height.reset();
        self.last_sample = None;
        self.ramp_from = Some(now);
        self.started = Some(now);
        self.tripped = false;
        self.fast_since = None;
        self.lead_t = None;
        self.nx = [0.0; 2];
        self.ny = [0.0; 2];
        self.clp = [0.0; 7];
    }

    pub fn running(&self) -> bool {
        self.started.is_some()
    }

    pub fn stop(&mut self) {
        self.started = None;
    }

    /// One IMU sample (time on the monotonic clock, m/s², deg/s).
    pub fn feed_imu(&mut self, t: f64, acc: [f64; 3], gyro: [f64; 3]) {
        self.imu.update(t, acc, gyro);
        self.last_sample = Some(t);
    }

    /// The reference tool height this tick (m).
    pub fn feed_height(&mut self, t: f64, z: f64) {
        self.height.update(t, z);
    }

    fn lead(&mut self, now: f64, x: f64) -> f64 {
        if self.cfg.lead_hz <= 0.0 {
            return x;
        }
        let Some(t_prev) = self.lead_t else {
            self.lead_t = Some(now);
            self.lead_in = x;
            self.lead_out = x;
            return x;
        };
        let dt = now - t_prev;
        self.lead_t = Some(now);
        if !(0.0 < dt && dt < 0.1) {
            return self.lead_out;
        }
        let wz = PI * self.cfg.lead_hz;
        let wp = 4.0 * PI * self.cfg.lead_hz;
        let k = 2.0 / dt;
        let (b0, b1) = (1.0 + k / wz, 1.0 - k / wz);
        let (a0, a1) = (1.0 + k / wp, 1.0 - k / wp);
        let y = (b0 * x + b1 * self.lead_in - a1 * self.lead_out) / a0;
        self.lead_in = x;
        self.lead_out = y;
        y
    }

    fn notch(&mut self, x: f64) -> f64 {
        if self.cfg.notch_hz <= 0.0 {
            return x;
        }
        let w0 = 2.0 * PI * self.cfg.notch_hz / 240.0;
        let alpha = w0.sin() / (2.0 * self.cfg.notch_q);
        let a0 = 1.0 + alpha;
        let (b0, b1, b2) = (1.0 / a0, -2.0 * w0.cos() / a0, 1.0 / a0);
        let (a1, a2) = (-2.0 * w0.cos() / a0, (1.0 - alpha) / a0);
        let y = b0 * x + b1 * self.nx[0] + b2 * self.nx[1] - a1 * self.ny[0] - a2 * self.ny[1];
        self.nx = [x, self.nx[0]];
        self.ny = [y, self.ny[0]];
        y
    }

    /// The joint torques (Nm, arm slots 0-6) for this tick; `jz[i]` is
    /// ∂(tool height)/∂q_i at the measured pose.
    pub fn torque(&mut self, now: f64, jz: &[f64; 7]) -> [f64; 7] {
        let mut tau = [0.0; 7];
        let started = self.started.is_some_and(|s| now - s >= self.cfg.ramp_s);
        // The guard waits out the ramp-in (the filters' start-up transient).
        let fast = self.flex().abs() > self.cfg.trip_speed
            || (self.cfg.trip_hf_acc > 0.0 && self.imu.hf_rms() > self.cfg.trip_hf_acc);
        if started && fast {
            match self.fast_since {
                None => self.fast_since = Some(now),
                Some(since) if now - since > self.cfg.trip_s => self.tripped = true,
                _ => {}
            }
        } else {
            self.fast_since = None;
        }
        if self.tripped || self.started.is_none() {
            return tau;
        }
        match self.last_sample {
            Some(t) if now - t <= self.cfg.stale_s => {}
            _ => {
                self.ramp_from = None; // re-ramp after the gap
                return tau;
            }
        }
        let from = *self.ramp_from.get_or_insert(now);
        let ramp = if self.cfg.ramp_s > 0.0 {
            ((now - from) / self.cfg.ramp_s).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let flex = self.flex();
        let shaped = self.lead(now, flex);
        let force = -self.cfg.gain * self.notch(shaped);
        for k in 0..self.cfg.columns.len() {
            let col = self.cfg.columns[k];
            let mut value = ramp * col.weight * jz[col.slot] * force;
            if col.lp_hz > 0.0 {
                let alpha = 1.0 - (-2.0 * PI * col.lp_hz / 240.0).exp();
                value = self.clp[col.slot] + alpha * (value - self.clp[col.slot]);
                self.clp[col.slot] = value;
            }
            tau[col.slot] = value.clamp(-self.cfg.max_torque, self.cfg.max_torque);
        }
        tau
    }
}

/// One wrist-IMU datagram from `almond_axol.zed.imu_worker`'s live sender:
/// `<d6f` — perf_counter time (s), acceleration (m/s²), gyro (deg/s).
pub const IMU_DATAGRAM: usize = 8 + 6 * 4;

pub fn decode_imu(data: &[u8]) -> Option<(f64, [f64; 3], [f64; 3])> {
    if data.len() != IMU_DATAGRAM {
        return None;
    }
    let t = f64::from_le_bytes(data[0..8].try_into().ok()?);
    let f = |i: usize| -> f64 {
        let o = 8 + 4 * i;
        f32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f64
    };
    Some((t, [f(0), f(1), f(2)], [f(3), f(4), f(5)]))
}

/// `CLOCK_MONOTONIC` in seconds — Python's `time.perf_counter` on Linux, the
/// clock the IMU worker stamps its samples with.
pub fn monotonic_s() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[cfg(test)]
#[path = "tipdamp_golden.rs"]
mod golden;

#[cfg(test)]
mod tests {
    use super::golden::*;
    use super::*;

    fn close(got: f64, want: f64, tol: f64, what: &str) {
        assert!(
            (got - want).abs() <= tol * (1.0 + want.abs()),
            "{what}: got {got}, want {want}"
        );
    }

    #[test]
    fn vertical_velocity_matches_python() {
        let mut est = VerticalVelocity::new(0.3, 40.0);
        let mut vals = Vec::new();
        for k in 0..600 {
            let kf = k as f64;
            let t = kf * 0.005 + 0.0003 * kf.sin();
            let acc = [
                0.5 * (2.0 * PI * 0.7 * t).sin(),
                0.3 * (2.0 * PI * 1.3 * t).cos(),
                G + 0.2 * (2.0 * PI * 2.0 * t).sin(),
            ];
            let gyro = [
                20.0 * (2.0 * PI * 0.4 * t).sin(),
                10.0 * (2.0 * PI * 0.9 * t).cos(),
                5.0 * (2.0 * PI * 1.7 * t).sin(),
            ];
            vals.push(est.update(t, acc, gyro));
        }
        for (n, &i) in IDX.iter().enumerate() {
            close(vals[i], VERTICAL[n], 1e-9, "vertical");
        }
    }

    #[test]
    fn height_velocity_matches_python() {
        let mut h = HeightVelocity::new(0.3, 40.0, 0.008);
        let mut vals = Vec::new();
        for k in 0..600 {
            let kf = k as f64;
            let t = kf / 240.0 + 0.0002 * (1.3 * kf).sin();
            let z = 0.05 * (2.0 * PI * 0.3 * t).sin() + 0.001 * (2.0 * PI * 3.0 * t).sin();
            vals.push(h.update(t, z));
        }
        for (n, &i) in IDX.iter().enumerate() {
            close(vals[i], HEIGHT[n], 1e-9, "height");
        }
    }

    #[test]
    fn tracking_filter_matches_python() {
        let mut f = TrackingFilter::new(14.795, 0.597, Some(36.217), 0.00363, 240.0);
        let vals: Vec<f64> = (0..600)
            .map(|k| {
                let t = k as f64 / 240.0;
                f.step(0.4 * (2.0 * PI * 0.5 * t).sin() + 0.1 * (2.0 * PI * 2.2 * t).sin())
            })
            .collect();
        for (n, &i) in IDX.iter().enumerate() {
            close(vals[i], TRACKING[n], 1e-9, "tracking");
        }
    }

    fn damper() -> TipDamper {
        TipDamper::new(TipConfig {
            gain: 60.0,
            hp_hz: 0.3,
            lp_hz: 40.0,
            lead_hz: 8.0,
            notch_hz: 11.0,
            notch_q: 1.5,
            max_torque: 0.5,
            reference: Reference::Command,
            delay_s: 0.008,
            ramp_s: 1.0,
            stale_s: 0.05,
            trip_speed: 0.3,
            trip_s: 0.15,
            trip_hf_acc: 1.4,
            columns: vec![
                TipColumn {
                    slot: 0,
                    weight: 1.0,
                    lp_hz: 0.0,
                },
                TipColumn {
                    slot: 3,
                    weight: 0.6,
                    lp_hz: 6.0,
                },
            ],
        })
    }

    #[test]
    fn tip_damper_matches_python() {
        let mut d = damper();
        d.start(0.0);
        let jz = [-0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0];
        let mut taus = Vec::new();
        let mut j = 0u64;
        for k in 0..(6 * 240) {
            let now = k as f64 / 240.0;
            d.feed_height(now, 0.001 * (2.0 * PI * 0.25 * now).sin());
            while (j as f64) * 0.005 <= now - 0.008 {
                let t = j as f64 * 0.005;
                let a = -0.002 * (2.0 * PI * 2.0).powi(2) * (2.0 * PI * 2.0 * t).sin();
                d.feed_imu(t, [0.0, 0.0, G + a], [0.0; 3]);
                j += 1;
            }
            taus.push(d.torque(now, &jz));
        }
        for (n, &i) in DAMP_IDX.iter().enumerate() {
            close(taus[i][0], DAMP_S1[n], 1e-9, "shoulder_1 torque");
            close(taus[i][3], DAMP_EL[n], 1e-9, "elbow torque");
        }
        assert!(taus.iter().all(|t| t[1] == 0.0 && t[2] == 0.0));
    }

    #[test]
    fn a_stale_imu_and_a_stopped_damper_give_no_torque() {
        let mut d = damper();
        let jz = [-0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0];
        assert_eq!(d.torque(0.0, &jz), [0.0; 7]); // never started
        d.start(0.0);
        d.feed_imu(0.0, [0.0, 0.0, G], [0.0; 3]);
        assert_eq!(d.torque(1.0, &jz), [0.0; 7]); // 1 s without a sample
    }

    #[test]
    fn a_runaway_trips_the_damper_off() {
        // The runaway on jelly: 1.9-2.7 m/s² of band acceleration at
        // ~11 Hz; ordinary passes 0.5-0.9.
        let run = |amp: f64| {
            let mut d = damper();
            d.start(0.0);
            let jz = [-0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0];
            for k in 0..(4 * 240) {
                let now = k as f64 / 240.0;
                d.feed_height(now, 0.0);
                if k % 6 < 5 {
                    let a = amp * (2.0 * PI * 11.0 * now).sin();
                    d.feed_imu(now, [0.0, 0.0, G + a], [0.0; 3]);
                }
                d.torque(now, &jz);
            }
            d
        };
        let mut d = run(3.0);
        assert!(d.tripped);
        assert_eq!(
            d.torque(4.0, &[-0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0]),
            [0.0; 7]
        );
        assert!(!run(0.8).tripped);
    }

    #[test]
    fn poe_chain_matches_the_solver() {
        let mut w = [[0.0; 3]; 7];
        let mut r = [[0.0; 3]; 7];
        let mut m = [[0.0; 4]; 4];
        for i in 0..7 {
            for k in 0..3 {
                w[i][k] = CHAIN_W[3 * i + k];
                r[i][k] = CHAIN_R[3 * i + k];
            }
        }
        for i in 0..4 {
            for k in 0..4 {
                m[i][k] = CHAIN_M[4 * i + k];
            }
        }
        let chain = PoeChain { w, r, m };
        for p in 0..4 {
            let mut q = [0.0; 7];
            q.copy_from_slice(&POSES[7 * p..7 * p + 7]);
            let (h, jz) = chain.height_and_jz(&q);
            // The solver's FK is float32: ~3 µm.
            assert!(
                (h - POSE_HEIGHT[p]).abs() < 1e-5,
                "height {h} vs {}",
                POSE_HEIGHT[p]
            );
            for i in 0..7 {
                assert!((jz[i] - POSE_JZ[7 * p + i]).abs() < 1e-6, "jz[{i}]");
            }
        }
    }

    #[test]
    fn imu_datagram_round_trips() {
        let mut data = Vec::new();
        data.extend_from_slice(&123.25f64.to_le_bytes());
        for v in [0.5f32, -0.25, 9.81, 1.0, -2.0, 3.5] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let (t, acc, gyro) = decode_imu(&data).unwrap();
        assert_eq!(t, 123.25);
        assert!((acc[2] - 9.81).abs() < 1e-6 && gyro[2] == 3.5);
        assert!(decode_imu(&data[..20]).is_none());
    }
}
