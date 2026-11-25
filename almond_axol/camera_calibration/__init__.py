"""Where the cameras sit on Axol, measured on each robot.

``axol calibration.cameras`` measures the head and wrist camera mounts with a
printed ChArUco board (:mod:`.board`), the arms' joint readings and each
camera's factory intrinsics, and writes them to the robot's camera
calibration file (:mod:`.file`). The MCAP recorder places the cameras in
``/tf_static`` from that file when it exists and matches the robot.

- :mod:`.file`: the calibration file format, shared with the recorder.
- :mod:`.board`: the printed board, its detection, and the printable page.
- :mod:`.optics`: projection and pose estimation for the ZED camera models.
- :mod:`.solve`: the joint least-squares solve and its pass/fail checks.
- :mod:`.session`: the hand-guided capture on the robot.
"""
