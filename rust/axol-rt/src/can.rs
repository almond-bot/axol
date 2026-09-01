//! Raw SocketCAN socket, mirroring the Python stack's `CanBus` at the frame
//! level. Deliberately built on `libc` rather than a wrapper crate: the
//! realtime core needs direct control over receive timeouts, kernel filters,
//! and (later) hardware timestamping.

use std::ffi::CString;
use std::io;
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};

/// Classic CAN frame as the kernel defines it (`struct can_frame`).
#[repr(C)]
#[derive(Clone, Copy)]
struct CanFrameRaw {
    can_id: u32,
    can_dlc: u8,
    _pad: u8,
    _res0: u8,
    _res1: u8,
    data: [u8; 8],
}

#[repr(C)]
struct SockaddrCan {
    can_family: libc::sa_family_t,
    can_ifindex: libc::c_int,
    rx_id: u32,
    tx_id: u32,
}

/// A received frame: arbitration ID, payload, payload length.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub id: u32,
    pub data: [u8; 8],
    #[allow(dead_code)] // consumed by the command path (DLC-checked decode)
    pub len: u8,
}

pub struct CanSock {
    fd: RawFd,
}

impl CanSock {
    /// Open a raw CAN socket bound to `iface` (e.g. `"can_alm_axol_l"`).
    pub fn open(iface: &str) -> io::Result<Self> {
        let fd = unsafe { libc::socket(libc::PF_CAN, libc::SOCK_RAW, libc::CAN_RAW) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let name = CString::new(iface).expect("interface name with NUL byte");
        let ifindex = unsafe { libc::if_nametoindex(name.as_ptr()) };
        if ifindex == 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }
        let addr = SockaddrCan {
            can_family: libc::AF_CAN as libc::sa_family_t,
            can_ifindex: ifindex as libc::c_int,
            rx_id: 0,
            tx_id: 0,
        };
        let rc = unsafe {
            libc::bind(
                fd,
                &addr as *const SockaddrCan as *const libc::sockaddr,
                std::mem::size_of::<SockaddrCan>() as libc::socklen_t,
            )
        };
        if rc < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }
        Ok(Self { fd })
    }

    /// Set the blocking-receive timeout (`SO_RCVTIMEO`) for coarse,
    /// stop-flag-polling receive loops (the proxy).
    ///
    /// Never use this for a deadline the control loop must honour: the kernel
    /// rounds a socket timeout up to whole scheduler jiffies and the timer
    /// wheel then expires it one to two jiffies late. Measured on the Jetson's
    /// HZ=250 kernel, a 3.7 ms `SO_RCVTIMEO` blocked 4-12 ms (6.7 ms on
    /// average), and a `Duration` under 1 µs disables the timeout entirely
    /// (blocks forever). Use [`Self::recv_timeout`] for deadlines.
    pub fn set_recv_timeout(&self, timeout: Duration) -> io::Result<()> {
        self.set_timeout(libc::SO_RCVTIMEO, timeout)
    }

    /// Wait up to `timeout` for one frame; `Ok(None)` when nothing arrived.
    ///
    /// Built on `ppoll` + a non-blocking `recv` so the wait ends within
    /// microseconds of the deadline: `ppoll` timeouts are hrtimer-based
    /// (with zero slack for a SCHED_FIFO caller), unlike `SO_RCVTIMEO`'s
    /// jiffy-granular timer-wheel timeout. This is what lets the 240 Hz bus
    /// loop bound its reply window inside the cycle — with `SO_RCVTIMEO`, one
    /// missing motor reply stretched a 4.17 ms tick to ~9 ms and the next
    /// tick tripped the whole-cycle timing fault. A zero `timeout` is a plain
    /// non-blocking probe.
    pub fn recv_timeout(&self, timeout: Duration) -> io::Result<Option<Frame>> {
        let deadline = Instant::now() + timeout;
        loop {
            if !wait_readable(self.fd, deadline)? {
                return Ok(None);
            }
            // Readiness can be spurious (or signal a socket error, which the
            // receive surfaces); never block here, and re-check the deadline
            // rather than spinning if nothing was actually queued.
            if let Some(frame) = self.recv_nonblocking()? {
                return Ok(Some(frame));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
        }
    }

    /// Set the blocking-send timeout (`SO_SNDTIMEO`). A dead bus (e-stop —
    /// nothing ACKs, qdisc full) normally surfaces as an immediate `ENOBUFS`
    /// write error, but if the socket sndbuf is what fills, a blocking write
    /// would hang the control loop forever; the timeout turns that into
    /// `EAGAIN`, which the stall detection treats like `ENOBUFS`.
    pub fn set_send_timeout(&self, timeout: Duration) -> io::Result<()> {
        self.set_timeout(libc::SO_SNDTIMEO, timeout)
    }

    fn set_timeout(&self, opt: libc::c_int, timeout: Duration) -> io::Result<()> {
        let tv = libc::timeval {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_usec: timeout.subsec_micros() as libc::suseconds_t,
        };
        let rc = unsafe {
            libc::setsockopt(
                self.fd,
                libc::SOL_SOCKET,
                opt,
                &tv as *const libc::timeval as *const libc::c_void,
                std::mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        };
        if rc < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Send one classic frame with a standard (11-bit) ID.
    pub fn send(&self, id: u16, data: &[u8]) -> io::Result<()> {
        assert!(data.len() <= 8);
        let mut frame = CanFrameRaw {
            can_id: id as u32,
            can_dlc: data.len() as u8,
            _pad: 0,
            _res0: 0,
            _res1: 0,
            data: [0; 8],
        };
        frame.data[..data.len()].copy_from_slice(data);
        let n = unsafe {
            libc::write(
                self.fd,
                &frame as *const CanFrameRaw as *const libc::c_void,
                std::mem::size_of::<CanFrameRaw>(),
            )
        };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Blocking receive. Returns `Ok(None)` on timeout (`SO_RCVTIMEO`).
    pub fn recv(&self) -> io::Result<Option<Frame>> {
        self.recv_with_flags(0)
    }

    /// Receive without waiting, independently of the socket's configured
    /// timeout. The realtime loop uses this immediately before sending a new
    /// batch so a reply that arrived after the previous tick's deadline can
    /// never be mistaken for feedback from the new command.
    pub fn recv_nonblocking(&self) -> io::Result<Option<Frame>> {
        self.recv_with_flags(libc::MSG_DONTWAIT)
    }

    fn recv_with_flags(&self, flags: libc::c_int) -> io::Result<Option<Frame>> {
        let mut frame = CanFrameRaw {
            can_id: 0,
            can_dlc: 0,
            _pad: 0,
            _res0: 0,
            _res1: 0,
            data: [0; 8],
        };
        let n = unsafe {
            libc::recv(
                self.fd,
                &mut frame as *mut CanFrameRaw as *mut libc::c_void,
                std::mem::size_of::<CanFrameRaw>(),
                flags,
            )
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            if matches!(
                err.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) {
                return Ok(None);
            }
            return Err(err);
        }
        Ok(Some(Frame {
            id: frame.can_id & 0x7FF,
            data: frame.data,
            len: frame.can_dlc,
        }))
    }

    /// Drop frames already queued without changing the receive timeout.
    pub fn drain_nonblocking(&self) -> io::Result<()> {
        while self.recv_nonblocking()?.is_some() {}
        Ok(())
    }

    /// Drain anything already queued on the socket without blocking.
    pub fn drain(&self) -> io::Result<()> {
        self.drain_nonblocking()
    }
}

impl Drop for CanSock {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

/// Block until `fd` is readable or `deadline` passes; `Ok(false)` on expiry.
///
/// `ppoll` with an absolute-deadline-derived timeout: hrtimer-precise, and a
/// deadline already in the past is a non-blocking probe (unlike a zero
/// `SO_RCVTIMEO`, which the kernel reads as "no timeout"). `EINTR` re-arms
/// against the same deadline.
fn wait_readable(fd: RawFd, deadline: Instant) -> io::Result<bool> {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let ts = libc::timespec {
            tv_sec: remaining.as_secs() as libc::time_t,
            tv_nsec: remaining.subsec_nanos() as libc::c_long,
        };
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let rc = unsafe { libc::ppoll(&mut pfd, 1, &ts, std::ptr::null()) };
        if rc < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        return Ok(rc > 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket;
    use std::os::unix::io::AsRawFd;

    fn idle_socket() -> UdpSocket {
        UdpSocket::bind("127.0.0.1:0").expect("bind loopback udp socket")
    }

    #[test]
    fn wait_readable_expires_close_to_the_deadline() {
        // The whole point of ppoll over SO_RCVTIMEO: a sub-jiffy wait must end
        // near the requested instant, not one or two scheduler ticks later
        // (4-12 ms measured for a 3.7 ms socket timeout on an HZ=250 kernel).
        // Use a median so one preempted iteration cannot fail the test.
        let sock = idle_socket();
        let timeout = Duration::from_micros(2_000);
        let mut waits: Vec<Duration> = (0..15)
            .map(|_| {
                let started = Instant::now();
                let ready = wait_readable(sock.as_raw_fd(), started + timeout).unwrap();
                assert!(!ready);
                started.elapsed()
            })
            .collect();
        waits.sort();
        let median = waits[waits.len() / 2];
        assert!(median >= timeout, "returned early: {median:?}");
        assert!(
            median < timeout + Duration::from_millis(1),
            "wait overran its deadline: {median:?} for {timeout:?}"
        );
    }

    #[test]
    fn wait_readable_treats_a_past_deadline_as_a_probe() {
        // A sub-microsecond remaining window used to become SO_RCVTIMEO = 0,
        // which the kernel treats as "block forever".
        let sock = idle_socket();
        let started = Instant::now();
        let ready = wait_readable(sock.as_raw_fd(), started + Duration::from_nanos(500)).unwrap();
        assert!(!ready);
        assert!(started.elapsed() < Duration::from_millis(50));
        let ready = wait_readable(sock.as_raw_fd(), started).unwrap();
        assert!(!ready);
    }

    #[test]
    fn wait_readable_returns_as_soon_as_data_arrives() {
        let sock = idle_socket();
        let sender = idle_socket();
        sender
            .send_to(b"x", sock.local_addr().unwrap())
            .expect("loopback send");
        let started = Instant::now();
        let ready = wait_readable(sock.as_raw_fd(), started + Duration::from_secs(5)).unwrap();
        assert!(ready);
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
