//! Rolling on-wire CAN timing aggregation.
//!
//! This used to run over every raw frame in Python's passive observer.  Keep
//! the frame decoder there for UI motor state, but reduce control timing to a
//! small JSON snapshot here, on the same Rust/kernel receive clock as CAN.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

const JOINTS: usize = 8;
const WINDOW: Duration = Duration::from_secs(2);
const TARGET_HZ: f64 = 240.0;

#[derive(Default)]
struct JointEvents {
    commands: VecDeque<Instant>,
    feedback: VecDeque<Instant>,
    latencies: VecDeque<(Instant, f64)>,
    misses: VecDeque<Instant>,
    pending: Option<Instant>,
}

pub struct TimingAggregator {
    joints: [JointEvents; JOINTS],
}

impl TimingAggregator {
    pub fn new() -> Self {
        Self {
            joints: std::array::from_fn(|_| JointEvents::default()),
        }
    }

    pub fn reset(&mut self) {
        self.joints = std::array::from_fn(|_| JointEvents::default());
    }

    pub fn observe(&mut self, id: u32, data: &[u8; 8], now: Instant) {
        if let Some(slot) = command_slot(id, data) {
            let e = &mut self.joints[slot];
            if e.pending.replace(now).is_some() {
                e.misses.push_back(now);
            }
            e.commands.push_back(now);
        }
        if let Some(slot) = feedback_slot(id) {
            let e = &mut self.joints[slot];
            e.feedback.push_back(now);
            if let Some(sent) = e.pending.take() {
                e.latencies
                    .push_back((now, now.duration_since(sent).as_secs_f64()));
            }
        }
        self.prune(now);
    }

    fn prune(&mut self, now: Instant) {
        let cutoff = now.checked_sub(WINDOW).unwrap_or(now);
        for e in &mut self.joints {
            while e.commands.front().is_some_and(|t| *t < cutoff) {
                e.commands.pop_front();
            }
            while e.feedback.front().is_some_and(|t| *t < cutoff) {
                e.feedback.pop_front();
            }
            while e.latencies.front().is_some_and(|(t, _)| *t < cutoff) {
                e.latencies.pop_front();
            }
            while e.misses.front().is_some_and(|t| *t < cutoff) {
                e.misses.pop_front();
            }
        }
    }

    pub fn snapshot_json(&mut self, now: Instant) -> Option<String> {
        self.prune(now);
        let max_count = self.joints.iter().map(|e| e.commands.len()).max()?;
        if max_count < 2 {
            return None;
        }
        // Stable tick boundary: prefer the first joint within one frame of
        // the busiest, matching the old observer's shoulder_1 preference.
        let slot = self
            .joints
            .iter()
            .position(|e| e.commands.len() + 1 >= max_count)?;
        let probe = &self.joints[slot];
        if probe
            .commands
            .back()
            .is_none_or(|t| now.duration_since(*t) > WINDOW)
        {
            return None;
        }

        let commands: Vec<Instant> = probe.commands.iter().copied().collect();
        let feedback: Vec<Instant> = probe.feedback.iter().copied().collect();
        let command_dt = deltas(&commands);
        let feedback_dt = deltas(&feedback);
        let all_commands = sorted_events(&self.joints, true);
        let all_feedback = sorted_events(&self.joints, false);
        let mut command_batches = Vec::new();
        let mut feedback_batches = Vec::new();
        let mut cycles = Vec::new();
        let mut utilization = Vec::new();
        let mut headroom = Vec::new();
        for pair in commands.windows(2) {
            let start = pair[0];
            let end = pair[1];
            let tc: Vec<Instant> = all_commands
                .iter()
                .copied()
                .filter(|t| *t >= start && *t < end)
                .collect();
            if tc.is_empty() {
                continue;
            }
            command_batches.push(duration(tc[0], *tc.last().unwrap()));
            let tf: Vec<Instant> = all_feedback
                .iter()
                .copied()
                .filter(|t| *t >= tc[0] && *t < end)
                .collect();
            if tf.is_empty() {
                continue;
            }
            feedback_batches.push(duration(tf[0], *tf.last().unwrap()));
            let cycle = duration(tc[0], *tf.last().unwrap());
            let period = duration(start, end);
            cycles.push(cycle);
            if period > 0.0 {
                utilization.push(100.0 * cycle / period);
                headroom.push(period - cycle);
            }
        }
        let nominal = 1.0 / TARGET_HZ;
        let deadline_misses: usize = command_dt
            .iter()
            .map(|dt| ((*dt / nominal + 0.5) as usize).saturating_sub(1))
            .sum();
        let mut missed_feedback = probe.misses.len();
        if probe
            .pending
            .is_some_and(|t| now.duration_since(t).as_secs_f64() > 1.5 * nominal)
        {
            missed_feedback += 1;
        }
        let latencies: Vec<f64> = probe.latencies.iter().map(|(_, v)| *v).collect();
        let names = [
            "SHOULDER_1",
            "SHOULDER_2",
            "SHOULDER_3",
            "ELBOW",
            "WRIST_1",
            "WRIST_2",
            "WRIST_3",
            "GRIPPER",
        ];
        let age = |events: &[Instant]| {
            events
                .last()
                .map(|t| now.duration_since(*t).as_secs_f64() * 1e3)
        };
        Some(format!(
            "{{\"sourceJoint\":\"{}\",\"targetHz\":240.0,\"commandHz\":{},\"feedbackHz\":{},\"commandPeriodMs\":{},\"feedbackPeriodMs\":{},\"commandJitterP95Ms\":{},\"feedbackJitterP95Ms\":{},\"commandGapMaxMs\":{},\"feedbackGapMaxMs\":{},\"commandBatchP50Ms\":{},\"commandBatchP95Ms\":{},\"feedbackBatchP95Ms\":{},\"canCycleP50Ms\":{},\"canCycleP95Ms\":{},\"canUtilizationP95Pct\":{},\"canHeadroomP05Ms\":{},\"roundTripP50Ms\":{},\"roundTripP95Ms\":{},\"deadlineMisses\":{},\"missedFeedback\":{},\"commandAgeMs\":{},\"feedbackAgeMs\":{}}}",
            names[slot], js(rate(&commands)), js(rate(&feedback)), js(median(&command_dt).map(ms)),
            js(median(&feedback_dt).map(ms)), js(jitter(&command_dt)), js(jitter(&feedback_dt)),
            js(command_dt.iter().copied().reduce(f64::max).map(ms)),
            js(feedback_dt.iter().copied().reduce(f64::max).map(ms)),
            js(percentile(&command_batches, 0.50).map(ms)), js(percentile(&command_batches, 0.95).map(ms)),
            js(percentile(&feedback_batches, 0.95).map(ms)), js(percentile(&cycles, 0.50).map(ms)),
            js(percentile(&cycles, 0.95).map(ms)), js(percentile(&utilization, 0.95)),
            js(percentile(&headroom, 0.05).map(ms)), js(percentile(&latencies, 0.50).map(ms)),
            js(percentile(&latencies, 0.95).map(ms)), deadline_misses, missed_feedback,
            js(age(&commands)), js(age(&feedback)),
        ))
    }
}

fn command_slot(id: u32, data: &[u8; 8]) -> Option<usize> {
    match id {
        0x401..=0x405 => Some((id - 0x401) as usize),
        0x141..=0x145 if matches!(data[0], 0xA2 | 0xA4) => Some((id - 0x141) as usize),
        0x006..=0x008 => Some((id - 1) as usize),
        0x106..=0x108 | 0x206..=0x208 | 0x306..=0x308 => Some(((id & 0xff) - 1) as usize),
        _ => None,
    }
}

fn feedback_slot(id: u32) -> Option<usize> {
    match id {
        0x501..=0x505 => Some((id - 0x501) as usize),
        0x016..=0x018 => Some((id - 0x11) as usize),
        _ => None,
    }
}

fn duration(a: Instant, b: Instant) -> f64 {
    b.saturating_duration_since(a).as_secs_f64()
}
fn deltas(v: &[Instant]) -> Vec<f64> {
    v.windows(2)
        .map(|p| duration(p[0], p[1]))
        .filter(|v| *v > 0.0)
        .collect()
}
fn sorted_events(joints: &[JointEvents; JOINTS], commands: bool) -> Vec<Instant> {
    let mut out: Vec<Instant> = joints
        .iter()
        .flat_map(|e| {
            if commands {
                e.commands.iter()
            } else {
                e.feedback.iter()
            }
        })
        .copied()
        .collect();
    out.sort();
    out
}
fn rate(v: &[Instant]) -> Option<f64> {
    (v.len() >= 2)
        .then(|| (v.len() - 1) as f64 / duration(v[0], *v.last().unwrap()))
        .filter(|v| v.is_finite())
}
fn median(v: &[f64]) -> Option<f64> {
    percentile(v, 0.5)
}
fn percentile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let p = q * (s.len() - 1) as f64;
    let lo = p.floor() as usize;
    let hi = p.ceil() as usize;
    Some(if lo == hi {
        s[lo]
    } else {
        s[lo] + (s[hi] - s[lo]) * (p - lo as f64)
    })
}
fn jitter(v: &[f64]) -> Option<f64> {
    let c = median(v)?;
    percentile(&v.iter().map(|x| (x - c).abs()).collect::<Vec<_>>(), 0.95).map(ms)
}
fn ms(v: f64) -> f64 {
    v * 1e3
}
fn js(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map_or_else(|| "null".into(), |v| format!("{v:.9}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recognizes_arm_control_frames() {
        assert_eq!(command_slot(0x401, &[0; 8]), Some(0));
        assert_eq!(command_slot(0x207, &[0; 8]), Some(6));
        assert_eq!(feedback_slot(0x17), Some(6));
    }

    #[test]
    fn emits_dashboard_schema() {
        let mut timing = TimingAggregator::new();
        let start = Instant::now();
        for tick in 0..4 {
            let at = start + Duration::from_secs_f64(tick as f64 / 240.0);
            timing.observe(0x401, &[0; 8], at);
            timing.observe(0x501, &[0; 8], at + Duration::from_millis(1));
        }
        let json = timing
            .snapshot_json(start + Duration::from_millis(14))
            .unwrap();
        assert!(json.contains("\"sourceJoint\":\"SHOULDER_1\""));
        assert!(json.contains("\"commandHz\":240.000000000"));
        assert!(json.contains("\"roundTripP95Ms\":1.000000000"));
    }
}
