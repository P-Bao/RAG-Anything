import tempfile
from pathlib import Path

from ami_rag.core.calibration import (
    ModalityCalibration,
    RerankCalibrator,
    _interpolate,
    _logistic,
)


def test_logistic_math():
    assert _logistic(0.0) == 0.5
    assert _logistic(100.0) == 1.0
    assert _logistic(-100.0) == 0.0
    assert 0.0 < _logistic(1.0) < 1.0


def test_interpolate_isotonic():
    xs = [0.0, 0.5, 1.0]
    ys = [0.1, 0.6, 0.9]

    # Clip left
    assert _interpolate(-0.5, xs, ys) == 0.1
    # Clip right
    assert _interpolate(1.5, xs, ys) == 0.9
    # Exact points
    assert _interpolate(0.0, xs, ys) == 0.1
    assert _interpolate(0.5, xs, ys) == 0.6
    assert _interpolate(1.0, xs, ys) == 0.9
    # Linear between 0.0 and 0.5 (at 0.25 -> 0.1 + 0.25 * 0.5 / 0.5 = 0.35)
    assert abs(_interpolate(0.25, xs, ys) - 0.35) < 1e-6


def test_calibrator_modality_lookup_and_fallback():
    calibrator = RerankCalibrator(
        models={
            "image": ModalityCalibration(method="platt", a=50.0, b=-1.5),
            "text": ModalityCalibration(method="platt", a=10.0, b=-3.0),
            "default": ModalityCalibration(method="platt", a=5.0, b=-1.0),
        }
    )

    p_img = calibrator.predict(0.04, "image")
    p_txt = calibrator.predict(0.40, "text")
    p_unk = calibrator.predict(0.30, "unknown_modality")

    assert 0.0 <= p_img <= 1.0
    assert 0.0 <= p_txt <= 1.0
    assert 0.0 <= p_unk <= 1.0


def test_calibrator_reorders_multimodal_chunks():
    """Verify that calibration can elevate high-confidence image scores above mediocre text scores."""
    calibrator = RerankCalibrator(
        models={
            # Image score 0.04 is high for an image -> P ~ 0.88
            "image": ModalityCalibration(method="platt", a=80.0, b=-1.2),
            # Text score 0.15 is low/mediocre for text -> P ~ 0.27
            "text": ModalityCalibration(method="platt", a=10.0, b=-2.5),
        }
    )

    raw_scored = [
        ({"chunk_id": "txt-1", "modality": "text"}, 0.15),
        ({"chunk_id": "img-1", "modality": "image"}, 0.04),
    ]

    reordered = calibrator.calibrate_and_rank(raw_scored)

    # In raw score: txt-1 (0.15) > img-1 (0.04)
    # In calibrated P: img-1 (P ~ 0.88) > txt-1 (P ~ 0.27)
    assert reordered[0][0]["chunk_id"] == "img-1"
    assert reordered[0][1] > reordered[1][1]
    assert reordered[1][0]["chunk_id"] == "txt-1"


def test_calibrator_save_and_load_roundtrip():
    original = RerankCalibrator(
        models={
            "image": ModalityCalibration(method="platt", a=25.5, b=-0.8, samples_pos=10, samples_neg=50),
            "table": ModalityCalibration(method="isotonic", x=[0.0, 0.5, 1.0], y=[0.05, 0.55, 0.95]),
        },
        version="2.0",
        metadata={"trained_on": "PTIT-test-28"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "calib.json"
        original.save(path)

        loaded = RerankCalibrator.load(path)
        assert loaded is not None
        assert loaded.version == "2.0"
        assert loaded.metadata["trained_on"] == "PTIT-test-28"
        assert "image" in loaded.models
        assert "table" in loaded.models

        assert loaded.models["image"].a == 25.5
        assert loaded.models["image"].samples_pos == 10
        assert loaded.models["table"].method == "isotonic"
        assert loaded.models["table"].x == [0.0, 0.5, 1.0]

        # Prediction parity
        assert abs(original.predict(0.05, "image") - loaded.predict(0.05, "image")) < 1e-9
        assert abs(original.predict(0.25, "table") - loaded.predict(0.25, "table")) < 1e-9
