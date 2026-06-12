"""Property test: Prediction result display always includes all required fields.

**Validates: Requirements 2.5, 5.1, 5.3**

Constructs arbitrary PredictionResult objects and asserts the rendered result
widget text contains label, confidence, estimated_age, and age_group_display.
"""

import sys
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from hypothesis import given, settings
import hypothesis.strategies as st

from PyQt5.QtWidgets import QApplication

from src.inference.decision_router import PredictionResult, format_confidence
from src.gui.main_window import MainWindow

# Ensure QApplication exists
app = QApplication.instance() or QApplication(sys.argv)

# Strategy for generating arbitrary PredictionResult objects
prediction_results = st.builds(
    PredictionResult,
    label=st.sampled_from(["Male", "Female", "Undetermined"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    estimated_age=st.integers(min_value=1, max_value=100),
    age_group=st.sampled_from(["target", "outside"]),
    age_group_display=st.sampled_from([
        "Age 20\u201330 (biased prediction)",
        "Outside age range (standard prediction)",
    ]),
)


# Feature: long-hair-gender-identification, Property 7: Prediction result display always includes all required fields
@given(result=prediction_results)
@settings(max_examples=100)
def test_result_display_contains_all_fields(result):
    """Property 7: The rendered result area always contains all required fields."""
    window = MainWindow(engine=None)
    window._on_inference_complete(result)

    # Verify all fields are populated in the widgets
    assert window.prediction_label.text() == result.label
    assert window.confidence_label.text() == format_confidence(result.confidence)
    assert window.estimated_age_label.text() == str(result.estimated_age)
    assert window.age_group_label.text() == result.age_group_display
    assert not window.placeholder_label.isVisible()
