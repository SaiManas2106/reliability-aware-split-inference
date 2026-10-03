from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

@dataclass
class BandwidthEstimator:
    window: int = 30

    def __post_init__(self):
        self.samples: Deque[Tuple[int, float]] = deque(maxlen=self.window)

    def add(self, nbytes: int, transfer_ms: float):
        if transfer_ms <= 0:
            return
        self.samples.append((nbytes, transfer_ms))

    def mean_bytes_per_ms(self) -> Optional[float]:
        if not self.samples:
            return None
        total_bytes = sum(b for b, _ in self.samples)
        total_ms = sum(ms for _, ms in self.samples)
        if total_ms <= 0:
            return None
        return total_bytes / total_ms

    def percentile_bytes_per_ms(self, p: float) -> Optional[float]:
        if not self.samples:
            return None
        vals = sorted((b / ms) for b, ms in self.samples if ms > 0)
        if not vals:
            return None
        k = int(round((p / 100.0) * (len(vals) - 1)))
        k = max(0, min(k, len(vals) - 1))
        return vals[k]
