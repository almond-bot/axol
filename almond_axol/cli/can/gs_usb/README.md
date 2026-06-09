# gs_usb out-of-tree kernel module

NVIDIA L4T/tegra kernels (Jetson Orin, ZED Box, etc.) are built without any
USB-CAN drivers, so the Almond Axol Hub adapter (`1d50:606f`, gs_usb protocol)
never enumerates as `can*` network interfaces. `axol can.driver` builds this
module against the running kernel's headers and installs it.

`gs_usb.c` is the upstream stable v5.15.148 driver
(`drivers/net/can/usb/gs_usb.c`) with two backports:

1. `netdev->dev_id = channel` in `gs_make_candev()` (upstream `04c9b00ba835`)
   — without it both channels report `dev_id 0x0` and the left/right udev
   rules written by `axol can.setup` cannot tell them apart.
2. Bulk endpoint addresses are read from the USB interface descriptor in
   `gs_usb_probe()` instead of being hardcoded (`IN 1` / `OUT 2`) — the Axol
   Hub firmware uses `EP1 IN` / `EP1 OUT`, so the stock 5.15 driver submits
   every TX URB to a nonexistent endpoint (`usb_submit failed (err=-2)`).

Kernels >= 6.4 ship both fixes in-tree; `axol can.driver` is a no-op there.
