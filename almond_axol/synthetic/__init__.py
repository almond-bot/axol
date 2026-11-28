"""Synthetic sensor data from the sim: rendered camera images, robot motion
and whole MCAP episodes written by the real recorder.

Nothing here needs hardware. :mod:`.render` draws the robot model and a
table scene from any camera pose through the ZED lens models, :mod:`.motion`
plans arm motions with the URDF's kinematics, and :mod:`.episode` plays them
into :class:`~almond_axol.recording.mcap.recorder.McapSession`.
"""
