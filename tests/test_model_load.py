"""Property 12: Missing model weights always produce a specific error on launch.

**Validates: Requirements 6.5**

For any combination of absent model weight files (at least one missing),
InferenceEngine.load_models() always raises ModelLoadError with a message
instructing the user to run train.py.
"""

import itertools
import sys
from unittest.mock import MagicMock

import pytest

# Mock heavy dependencies that are not needed for testing file-existence checks
for mod_name in [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.applications",
    "tensorflow.keras.layers",
    "tensorflow.keras.models",
    "cv2",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from src.inference.inference_engine import InferenceEngine, ModelLoadError  # noqa: E402

MODEL_FILES = ["age_estimator.keras", "hair_classifier.keras", "gender_predictor.keras"]


def _get_missing_file_combinations():
    """Generate all combinations where at least one file is missing."""
    combos = []
    for r in range(1, len(MODEL_FILES) + 1):
        for combo in itertools.combinations(MODEL_FILES, r):
            combos.append(combo)
    return combos


@pytest.mark.parametrize("missing_files", _get_missing_file_combinations())
def test_missing_model_weights_raise_model_load_error(missing_files, tmp_path):
    """Property 12: Any combination of absent model files raises ModelLoadError."""
    # Create model directory with only non-missing files
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    for f in MODEL_FILES:
        if f not in missing_files:
            (model_dir / f).write_text("dummy content")

    engine = InferenceEngine(model_dir=str(model_dir))

    with pytest.raises(ModelLoadError) as exc_info:
        engine.load_models()

    # Verify the error message instructs running train.py
    error_message = str(exc_info.value).lower()
    assert "train.py" in error_message, (
        f"Expected error message to mention 'train.py', got: {exc_info.value}"
    )
