"""Shared constants for live camera streaming."""

# Headset video is deliberately fixed at 30 fps. Camera capture may run faster
# for recording or policy observations; the WebRTC branch is rate-limited
# independently so changing those data rates never changes network load.
HEADSET_STREAM_FPS = 30
