"""Cooperative whole-sample deadline; cannot interrupt blocked functions."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Deadline:
    expires_at: float
    clock: Callable[[], float]

    def remaining(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    def expired(self) -> bool:
        return self.remaining() <= 0
