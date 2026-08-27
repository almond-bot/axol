//! Raw SocketCAN socket, mirroring the Python stack's `CanBus` at the frame
//! level. Deliberately built on `libc` rather than a wrapper crate: the
//! realtime core needs direct control over receive timeouts, kernel filters,
//! and (later) hardware timestamping.

use std::ffi::CString;
use std::io;
use std::os::unix::io::RawFd;
use std::time::Duration;

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

    /// Set the blocking-receive timeout (`SO_RCVTIMEO`).
    pub fn set_recv_timeout(&self, timeout: Duration) -> io::Result<()> {
        let tv = libc::timeval {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_usec: timeout.subsec_micros() as libc::suseconds_t,
        };
        let rc = unsafe {
            libc::setsockopt(
                self.fd,
                libc::SOL_SOCKET,
                libc::SO_RCVTIMEO,
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
        let mut frame = CanFrameRaw {
            can_id: 0,
            can_dlc: 0,
            _pad: 0,
            _res0: 0,
            _res1: 0,
            data: [0; 8],
        };
        let n = unsafe {
            libc::read(
                self.fd,
                &mut frame as *mut CanFrameRaw as *mut libc::c_void,
                std::mem::size_of::<CanFrameRaw>(),
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

    /// Drain anything already queued on the socket without blocking.
    pub fn drain(&self) -> io::Result<()> {
        self.set_recv_timeout(Duration::from_micros(1))?;
        while self.recv()?.is_some() {}
        Ok(())
    }
}

impl Drop for CanSock {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}
