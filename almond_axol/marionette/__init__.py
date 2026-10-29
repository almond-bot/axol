"""Teleoperation from iPhones through a Marionette gateway.

Marionette relays each phone's ARKit pose and button presses; this package
is Axol's side, built from the Marionette SDK's parts (the ``marionette``
extra): its phone screen (``screen.html``), what the screen's controls mean,
how an arm follows a phone, and the session that runs it all. See
:mod:`almond_axol.marionette.session`.
"""
