# Axol hub firmware

Firmware for the Axol arm hub: an RP2350 that bridges the two arm CAN buses
(PIO-based [can2040](https://github.com/KevinOConnor/can2040), channel 0 = left,
channel 1 = right) to USB as a gs_usb device (`1d50:606f`, manufacturer
"Almond", product "Axol CAN Bus"). The board design lives in `circuits-py`
(`designs/axol_hub`).

The built image ships in the Python package as `almond_axol/hub/axol_hub.uf2`.
Every bring-up (`Axol.connect()` / `enable()`) compares the hub's version (its
USB `bcdDevice`) with it and reflashes an older hub over USB; see
`almond_axol/hub/firmware.py`. `axol hub.info` shows what a robot's hub runs.

## Changing the firmware

1. Edit the sources.
2. Bump `AXOL_HUB_FW_VERSION` in `CMakeLists.txt` and `FIRMWARE_VERSION` in
   `almond_axol/hub/firmware.py` to the same value (BCD: `0x0201` = 2.01).
3. Run `./build.sh`. It builds with the Pico SDK 2.2.0 (`PICO_SDK_PATH`,
   default `~/.pico-sdk/sdk/2.2.0`) and copies the image into the package.
4. Commit the sources and the new `axol_hub.uf2` together.
   `tests/test_hub_firmware.py` fails if the two version numbers disagree or
   if the image does not report the version.

Pass `CMAKE_ARGS=-DFETCHCONTENT_SOURCE_DIR_PICOTOOL=<picotool source>` to
reuse a picotool checkout instead of fetching one.

## Flashing by hand

```
sudo axol hub.flash                # outdated hubs behind the arm channels
sudo axol hub.flash --port 1-3     # one hub by USB port, also one in BOOTSEL
```

Hubs running firmware from before 2026-08-17 can't be rebooted into the
bootloader from the host. Flash those once by hand: hold BOOTSEL while
plugging the hub in, then run `sudo axol hub.flash --port <port>`.

## Version history

- **2.00**: A USB bus reset now ends the host session. Before this, TinyUSB
  never called `tud_umount_cb` on a bus reset, so the host's USB-reset purge
  after an e-stop left the old session's frames queued and still valid, and
  leaked its echoes to the re-enumerated driver ("Unexpected unused echo
  id"). Also: a frame not handed to the CAN engine within 50 ms of arriving
  is dropped. Stopping a channel wipes the engine's transmit queue and parks
  TX recessive. Core1 wakes at least every 2 ms, so dead-bus recovery always
  runs. The version is reported as the USB `bcdDevice`.
- **1.00** (all earlier builds): `bcdDevice` 0x0100.
