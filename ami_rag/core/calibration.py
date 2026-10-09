"""Per-modality score calibration for multimodal reranking.

Runtime implementation is pure Python (no dependency on scikit-learn or numpy)
to keep container images lightweight and execution overhead minimal.

Supported methods:
- Platt scaling: P(relevant | score) = 1 / (1 + exp(-(a * score + b)))
- Isotonic regression: piecewise linear interpolation over step thresholds.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _logistic(z: float) -> float:
    """Numerically stable standard sigmoid."""
    if z >= 35.0:
        return 1.0
    if z <= -35.0:
        return 0.0
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _interpolate(score: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise linear interpolation with boundary clipping."""
    if not xs or not ys:
        return score
    if len(xs) == 1:
        return ys[0]
    if score <= xs[0]:
        return ys[0]
    if score >= xs[-1]:
        return ys[-1]

    # Binary search for interval
    low = 0
    high = len(xs) - 1
    while low <= high:
        mid = (low + high) // 2
        if xs[mid] <= score:
            if mid == len(xs) - 1 or score < xs[mid + 1]:
                x0, x1 = xs[mid], xs[mid + 1]
                y0, y1 = ys[mid], ys[mid + 1]
                if abs(x1 - x0) < 1e-12:
                    return y0
                return y0 + (score - x0) * (y1 - y0) / (x1 - x0)
            low = mid + 1
        else:
            high = mid - 1
    return ys[-1]


@dataclass
class ModalityCalibration:
    method: str = "platt"  # "platt" | "isotonic"
    a: float = 1.0
    b: float = 0.0
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    samples_pos: int = 0
    samples_neg: int = 0

    def predict(self, score: float) -> float:
        if self.method == "platt":
            return _logistic(self.a * score + self.b)
        if self.method == "isotonic":
            return _interpolate(score, self.x, self.y)
        return score


class RerankCalibrator:
    """Container mapping modalities to their respective calibration parameters."""

    def __init__(
        self,
        models: dict[str, ModalityCalibration] | None = None,
        version: str = "1.0",
        metadata: dict | None = None,
    ):
        self.models: dict[str, ModalityCalibration] = models or {}
        self.version = version
        self.metadata = metadata or {}

    def predict(self, score: float | None, modality: str | None = None) -> float:
        """Calculate calibrated probability P(relevant | score, modality)."""
        if score is None:
            return 0.0
        mod = (modality or "text").lower()
        model = self.models.get(mod) or self.models.get("default")
        if model is None:
            return float(score)
        return float(model.predict(score))

    def calibrate_and_rank(
        self,
        scored_chunks: list[tuple[dict, float | None]],
    ) -> list[tuple[dict, float, float | None]]:
        """Calibrate each chunk's rerank score and re-rank descending by calibrated probability.

        Returns list of tuples: (chunk, calibrated_score, raw_score).
        """
        calibrated: list[tuple[dict, float, float | None]] = []
        for chunk, raw_score in scored_chunks:
            modality = chunk.get("modality") or "text"
            p = self.predict(raw_score, modality)
            calibrated.append((chunk, p, raw_score))

        # Sort descending by calibrated probability P(relevant)
        calibrated.sort(key=lambda item: item[1], reverse=True)
        return calibrated

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "metadata": self.metadata,
            "models": {k: asdict(v) for k, v in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> RerankCalibrator:
        models: dict[str, ModalityCalibration] = {}
        for mod, mdata in (data.get("models") or {}).items():
            models[mod] = ModalityCalibration(
                method=mdata.get("method", "platt"),
                a=float(mdata.get("a", 1.0)),
                b=float(mdata.get("b", 0.0)),
                x=[float(v) for v in mdata.get("x", [])],
                y=[float(v) for v in mdata.get("y", [])],
                samples_pos=int(mdata.get("samples_pos", 0)),
                samples_neg=int(mdata.get("samples_neg", 0)),
            )
        return cls(
            models=models,
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RerankCalibrator | None:
        p = Path(path)
        if not p.is_file():
            logger.warning("Calibration file not found at %s", p)
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load calibration from %s: %s", p, exc)
            return None
