# gs_usb out-of-tree kernel module

NVIDIA L4T/tegra kernels (Jetson Orin, ZED Box, etc.) are built without any
USB-CAN drivers, so the Almond Axol Hub adapter (`1d50:606f`, gs_usb protocol)
never enumerates as `can*` network interfaces. `axol can.driver` builds this
module against the running kernel's headers and installs it.

`gs_usb.c` is the upstream stable v5.15.148 driver
(`drivers/net/can/usb/gs_usb.c`) with two backports:

1. `netdev->dev_id = channel` in `gs_make_candev()` (upstream `acff76fa45b4`)
   — without it both channels report `dev_id 0x0` and the left/right udev
   rules written by `axol can.setup` cannot tell them apart.
2. Bulk endpoint addresses are read from the USB interface descriptor in
   `gs_usb_probe()` instead of being hardcoded (`IN 1` / `OUT 2`) — the Axol
   Hub firmware uses `EP1 IN` / `EP1 OUT`, so the stock 5.15 driver submits
   every TX URB to a nonexistent endpoint (`usb_submit failed (err=-2)`).

Kernels >= 6.13 ship both fixes in-tree. `axol can.driver` keeps that native
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
