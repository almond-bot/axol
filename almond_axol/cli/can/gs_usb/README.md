# gs_usb out-of-tree kernel module

NVIDIA L4T/tegra kernels (Jetson Orin, ZED Box, etc.) are built without any
USB-CAN drivers, so the Almond Axol Hub adapter (`1d50:606f`, gs_usb protocol)
never enumerates as `can*` network interfaces. `axol can.driver` builds this
module against the running kernel's headers and installs it.

`gs_usb.c` is the upstream stable v5.15.148 driver
(`drivers/net/can/usb/gs_usb.c`) with these local changes:

1. `netdev->dev_id = channel` in `gs_make_candev()` (upstream `acff76fa45b4`)
   — without it both channels report `dev_id 0x0` and the left/right udev
   rules written by `axol can.setup` cannot tell them apart.
2. Bulk endpoint addresses are read from the USB interface descriptor in
   `gs_usb_probe()` instead of being hardcoded (`IN 1` / `OUT 2`) — the Axol
   Hub firmware uses `EP1 IN` / `EP1 OUT`, so the stock 5.15 driver submits
   every TX URB to a nonexistent endpoint (`usb_submit failed (err=-2)`).
3. RX URBs are re-anchored before every resubmit in
   `gs_usb_receive_bulk_callback()` (upstream `7352e1d5932a`, Dec 2025, and
   `79a6d1bfe114` for the unanchor-on-failure). The USB core unanchors a URB
   before running its completion, so in the stock driver every RX URB that
   completed once escaped `usb_kill_anchored_urbs()` in `gs_can_close()`.
   Upstream calls that a leak; here it was a use-after-free: `close()` freed
   the coherent buffers those still-pending URBs pointed at, and on the next
   `open()` they sat ahead of the fresh URBs in the endpoint queue, so the
   first frames of the new session were DMA'd into freed memory and parsed
   from it. A garbage `hf->channel` of 1 on the single-channel wheel/chest
   adapter indexes a NULL `canch[]` slot → `Kernel panic - not syncing: Fatal
   exception in interrupt` (seen on a Jetson Orin NX at the first frame after
   a teleop restart; `pstore` kept the oops at
   `gs_usb_receive_bulk_callback+0x5c`). A garbage channel >= 2 hit upstream's
   `device_detach` instead — interfaces silently dead until replug, the
   "mid-session USB drop" `axol can.setup`'s hotplug unit exists to recover
   from. A garbage channel of 0 delivered the corrupt frame to the motor
   driver as a valid one.
4. `gs_usb_receive_bulk_callback()` validates every frame before touching a
   channel, as defence in depth for 3 and for adapter firmware bugs. Upstream
   indexes `canch[hf->channel]` after only a `>= GS_MAX_INTF` check. Frames
   for an unregistered channel (out of range *or* a NULL slot), runt transfers
   (`actual_length < sizeof(gs_host_frame)`, whose tail is stale buffer
   contents), and frames for a channel that is no longer `netif_running`
   (upstream `5886e4d5ecec`) are dropped with a rate-limited warning and the
   URB is resubmitted; the interfaces stay up. Upstream's "detach the whole
   device on an out-of-range channel" is folded into the same drop path — one
   corrupt frame must not take the arm buses down until a replug either.
5. RX URB buffers are owned by the device (`struct gs_usb`) rather than by
   whichever channel opened first (upstream `2bda24ef95c0`), and a failed
   `gs_can_open()` unwinds its URBs and `close_candev()`s. With the
   dual-channel hub the closing channel is usually not the allocating one, so
   the stock driver leaked all 30 buffers on every close — which is also why
   the hub never hit the use-after-free in 3: its buffers were never freed.
6. `dev->parent` is set before `register_candev()`. The udev rule installed by
   `axol can.setup` brings an interface up the instant it appears, and
   `gs_can_open()` dereferences the parent.

Kernels >= 6.13 ship 1, 2, 5 and the `netif_running` part of 4 in-tree; 3 is
only in kernels that picked up the Dec 2025 / Jan 2026 stable backports
(5.15.y got it as `[PATCH 5.15 493/570]`). The channel guard in 4 is not
upstream in any kernel (mainline still indexes `canch[]` unchecked, with
`GS_MAX_INTF` 3), so a host kept on its native module is exposed to that
panic if an adapter ever reports a channel it does not have. `axol can.driver` keeps that native
module when it advertises every USB ID needed by the adapters currently
attached (and recognizes `weak-updates` symlinks that resolve back to it).
Maintained 6.2+ distro kernels that backported endpoint discovery are also kept
when kmod's imported-symbol metadata proves the capability. Older or
unverifiable modules are replaced conservatively. The optional
`16d0:117e` alias is required only while such a CANable is plugged in, so an
unused adapter type cannot cause a working signed desktop module to be
replaced.

On signature-enforcing hosts, an unsigned build is rejected before the active
module changes. A trusted, signed pre-version-marker Almond install is retained
only when its selected/loaded source identity, canonical path, and complete
three-ID alias fingerprint match. The installer verifies selection/loading and
rolls the prior file and load configuration back if a later step fails.
