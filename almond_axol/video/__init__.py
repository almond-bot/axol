"""Camera capture, encoding, and WebRTC video transport for the Axol.

The GPU-resident ZED pipeline and WebRTC relay shared by **both** VR teleop and
data collection — extracted from :mod:`almond_axol.vr` because the recorder
(:mod:`almond_axol.recording`) and the LeRobot adapters
(:mod:`almond_axol.lerobot`) consume it independently of VR.

Modules
───────
    gst_zed      GPU-resident zed-gstreamer capture (grab + NVENC encode)
    hw_video     Jetson NVENC hardware H.264 encoder for aiortc
    video        aiortc WebRTC relay (``WebRTCManager``) streaming to the headset
    video_proc   Out-of-process video relay subprocess (``VideoRelayProcess``)
    shm_frames   Shared-memory transport of raw frames to the recorder subprocess

These pull in heavy optional deps (aiortc, numpy, GStreamer), so import the
submodule you need lazily rather than re-exporting here.
"""
