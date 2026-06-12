"""Property-based tests for robustness and file-size gate.

Feature: long-hair-gender-identification
Property 8: System is robust to unreadable inputs
Property 9: File size gate rejects all oversized files
Validates: Requirements 8.1, 8.2, 1.5
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

# Mock tensorflow and related modules before importing the inference engine,
# since TensorFlow may not be installed in the test environment.
_tf_mock = MagicMock()
_modules_to_mock = {
    "tensorflow": _tf_mock,
    "tensorflow.keras": _tf_mock.keras,
    "tensorflow.keras.layers": _tf_mock.keras.layers,
    "tensorflow.keras.applications": _tf_mock.keras.applications,
    "tensorflow.keras.losses": _tf_mock.keras.losses,
    "tensorflow.keras.applications.mobilenet_v2": _tf_mock.keras.applications.mobilenet_v2,
}

with patch.dict("sys.modules", _modules_to_mock):
    from src.inference.inference_engine import (
        InferenceEngine,
        FileSizeError,
        InvalidFileFormatError,
        CorruptImageError,
        InferenceError,
    )


# Feature: long-hair-gender-identification, Property 8: System is robust to unreadable inputs
@given(data=st.binary(min_size=1, max_size=1024))
@settings(max_examples=100)
def test_robust_to_unreadable_inputs(data):
    """Property 8: Corrupt/random bytes never cause an unhandled exception.

    **Validates: Requirements 8.1, 8.2**
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_dir = "models"
        engine._preprocessor = MagicMock()
        engine._face_cascade = MagicMock()
        engine._router = MagicMock()
        engine._age_estimator = MagicMock()
        engine._hair_classifier = MagicMock()
        engine._gender_predictor = MagicMock()

        try:
            engine.predict(temp_path)
        except (FileSizeError, InvalidFileFormatError, CorruptImageError, InferenceError) as e:
            # These are expected user-readable errors
            assert isinstance(str(e), str) and len(str(e)) > 0
        except Exception as e:
            # Any other exception is a bug
            raise AssertionError(f"Unhandled exception: {type(e).__name__}: {e}")
    finally:
        os.unlink(temp_path)


# Feature: long-hair-gender-identification, Property 9: File size gate rejects all oversized files
@given(size=st.integers(min_value=10 * 1024 * 1024 + 1, max_value=10 * 1024 * 1024 + 10**6))
@settings(max_examples=100)
def test_file_size_gate_rejects_oversized(size):
    """Property 9: Files exceeding 10 MB always raise FileSizeError.

    **Validates: Requirements 1.5**
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
        f.write(b"\x00")  # Write minimal content

    try:
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_dir = "models"

        with patch("os.path.getsize", return_value=size):
            with patch("os.path.isfile", return_value=True):
                try:
                    engine.predict(temp_path)
                    raise AssertionError("Should have raised FileSizeError")
                except FileSizeError:
                    pass  # Expected
    finally:
        os.unlink(temp_path)
