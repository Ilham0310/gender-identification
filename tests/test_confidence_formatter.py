"""Property-based test for format_confidence utility.

Feature: long-hair-gender-identification
Property 6: Confidence score formatting is always valid
Validates: Requirements 5.2
"""

import re

from hypothesis import given, settings
import hypothesis.strategies as st

from src.inference.decision_router import format_confidence


@given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_confidence_format_always_valid(confidence):
    """Property 6: format_confidence always returns a string matching \\d+\\.\\d{2}%.

    **Validates: Requirements 5.2**
    """
    formatted = format_confidence(confidence)
    assert re.match(r'^\d+\.\d{2}%$', formatted), f"Invalid format: {formatted} for input {confidence}"
