"""Terminal quit from both the idle gate and active remote DAgger episodes."""

from __future__ import annotations

from .run_policy import _GATE_READY, _StdinPolicyControl


class DaggerStdinControl(_StdinPolicyControl):
    """One terminal reader at a time; only an explicit ready-state q parks.

    EOF, reader failures and declining a contact retry never request parking.
    Gate decisions remain available after joining the reader so a quit racing
    a VR Record event cannot be lost during the transition into an episode.
    """

    def __init__(self) -> None:
        super().__init__(eof_choice="abort", immediate_quit=True)
        self.abort_requested = False
        self._gate_active = False
        self._gate_result: str | None = None
        self._gate_phase = _GATE_READY

    def begin_gate(
        self, message: str, label: str = "Start episode", phase: str = _GATE_READY
    ) -> None:
        self.end_gate()
        if self.quit_requested or self.abort_requested:
            return
        super().begin_gate(message, label, phase)
        self._gate_phase = phase
        self._gate_result = None
        self._gate_active = True
        self._start_reader(allowed_choices=("q",))

    def note_gate(
        self, message: str, label: str = "Start episode", phase: str = _GATE_READY
    ) -> None:
        # Preserve the phase of a pending decision, particularly a contact
        # abort followed by the collector restoring the idle HUD message.
        self.poll_gate()
        if self._gate_result is None:
            self._gate_phase = phase
        super().note_gate(message, label, phase)

    def poll_gate(self) -> str | None:
        if self._gate_active and self._gate_result is None:
            self._check_reader_error()
            choice = self._result.get("choice")
            if choice == "q" and self._gate_phase == _GATE_READY:
                self.quit_requested = True
                self._gate_result = "quit"
            elif choice is not None:
                self.quit_requested = False
                self.abort_requested = True
                self._gate_result = "abort"
        return self._gate_result

    def end_gate(self) -> None:
        if self._gate_active:
            super().end_episode()
            self.poll_gate()
            self._gate_active = False

    def begin_episode(self, on_subtask=None, num_subtasks: int = 0) -> None:
        self.end_gate()
        if self.quit_requested or self.abort_requested:
            raise RuntimeError("Cannot start DAgger after a terminal quit or abort")
        self._gate_result = None
        super().begin_episode(on_subtask, num_subtasks)

    def poll_choice(self) -> str | None:
        choice = super().poll_choice()
        if choice == "abort":
            self.abort_requested = True
        return choice

    def await_continue(
        self, message: str, label: str = "Start episode", phase: str = _GATE_READY
    ) -> bool:
        self.end_gate()
        if self.quit_requested or self.abort_requested:
            if phase != _GATE_READY:
                self.quit_requested = False
                self.abort_requested = True
            return False
        proceed = super().await_continue(message, label, phase)
        if not proceed and not self.quit_requested:
            self.abort_requested = True
        return proceed

    def close(self) -> None:
        self.end_gate()
        super().close()
