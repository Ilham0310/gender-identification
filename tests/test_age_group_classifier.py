"""Property-based test for age_group_classify.

Feature: long-hair-gender-identification
Property 1: Age-group classification is total and exclusive
Validates: Requirements 2.3, 2.4
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from src.inference.decision_router import age_group_classify


@given(age=st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_age_group_classification_total_exclusive(age):
    """Property 1: age_group_classify returns exactly one of {"target", "outside"}.

    **Validates: Requirements 2.3, 2.4**
    """
    result = age_group_classify(age)
    # Result must be one of the two valid age groups (total and exclusive)
    assert result in {"target", "outside"}
    # Verify the if/else mapping is correct
    if 20 <= age <= 30:
        assert result == "target"
    else:
        assert result == "outside"
