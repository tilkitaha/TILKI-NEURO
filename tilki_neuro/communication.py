"""Deterministic command state machine; no external messages are sent."""
from dataclasses import dataclass, field


@dataclass
class Board:
    phrases: tuple = ('Yes', 'No', 'Water', 'Please help', 'Thank you')
    index: int = 0
    pending: str | None = None
    history: list = field(default_factory=list)
    armed: bool = True

    def step(self, label):
        if label not in (-1, 0, 1, 2):
            raise ValueError('Unknown command')
        # A rejected signal cancels pending selection; only genuine idle rearms.
        if label == -1:
            self.pending = None
            self.armed = False
            return 'Signal rejected; return to idle'
        if label == 0:
            self.armed = True
            return 'Idle'
        if not self.armed:
            return 'Return to idle before another command'
        self.armed = False
        if label == 1:
            self.pending = None
            self.index = (self.index + 1) % len(self.phrases)
            return 'Moved selection'
        selected = self.phrases[self.index]
        if self.pending == selected:
            self.history.append(selected)
            self.pending = None
            return 'Confirmed: ' + selected
        self.pending = selected
        return 'Pending: ' + selected + ' — idle, then select again to confirm'

    def cancel(self):
        self.pending = None
        self.armed = False

    def undo(self):
        self.cancel()
        if self.history:
            self.history.pop()
