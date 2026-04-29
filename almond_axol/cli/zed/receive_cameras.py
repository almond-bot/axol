"""
Stream one ZED camera to the workstation using the ZED SDK's built-in streaming.

Run on the ZED box:
    python receive_cameras.py

Receive on the workstation (requires pyzed):
    python view_cameras.py --zed-box-ip 192.168.10.1
"""

from __future__ import annotations

import asyncio
import threading

import pyzed.sl as sl

SERIAL = 305042468  # left_arm
PORT = 30000


async def main() -> None:
    zed = sl.CameraOne()

    init = sl.InitParametersOne()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30
    init.input.set_from_serial_number(SERIAL)

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open camera (serial {SERIAL}): {err}")

    stream_params = sl.StreamingParameters()
    stream_params.codec = sl.STREAMING_CODEC.H265
    stream_params.bitrate = 8000
    stream_params.port = PORT

    err = zed.enable_streaming(stream_params)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Failed to start streaming on port {PORT}: {err}")

    print(f"Streaming camera {SERIAL} on port {PORT}. Press Ctrl+C to stop.")

    stop = threading.Event()

    def grab_loop() -> None:
        while not stop.is_set():
            zed.grab()

    thread = threading.Thread(target=grab_loop, daemon=True)
    thread.start()

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        stop.set()
        thread.join(timeout=3.0)
        zed.disable_streaming()
        zed.close()
        print("Stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
