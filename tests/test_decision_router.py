"""Unit tests for DecisionRouter.route() method.

Tests verify:
- Target_Age_Group (20-30): hair_classifier is used, mapping long→Female, short→Male, Undetermined→Undetermined
- Outside_Age_Group (<20 or >30): gender_predictor is used only, hair label does not affect result
- PredictionResult is fully populated with correct fields
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference.decision_router import (
    DecisionRouter,
    PredictionResult,
    age_group_classify,
)


@pytest.fixture
def dummy_image():
    """A dummy preprocessed image of shape (224, 224, 3)."""
    return np.zeros((224, 224, 3), dtype=np.float32)


@pytest.fixture
def router():
    """A DecisionRouter instance."""
    return DecisionRouter()


def _make_age_estimator(age: int):
    """Create a mock AgeEstimator returning the given age."""
    mock = MagicMock()
    mock.predict.return_value = age
    return mock


def _make_hair_classifier(label: str, confidence: float):
    """Create a mock HairLengthClassifier returning the given label and confidence."""
    mock = MagicMock()
    mock.predict.return_value = (label, confidence)
    return mock


def _make_gender_predictor(label: str, confidence: float):
    """Create a mock GenderPredictor returning the given label and confidence."""
    mock = MagicMock()
    mock.predict.return_value = (label, confidence)
    return mock


class TestDecisionRouterTargetAgeGroup:
    """Tests for target age group (20-30) routing — biased prediction path."""

    def test_long_hair_maps_to_female(self, router, dummy_image):
        """Long hair in target age group should produce 'Female' label."""
        age_est = _make_age_estimator(25)
        hair_clf = _make_hair_classifier("long", 0.92)
        gender_pred = _make_gender_predictor("Male", 0.80)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.label == "Female"
        assert result.confidence == 0.92
        assert result.estimated_age == 25
        assert result.age_group == "target"
        assert "20" in result.age_group_display and "30" in result.age_group_display
        assert "biased" in result.age_group_display

    def test_short_hair_maps_to_male(self, router, dummy_image):
        """Short hair in target age group should produce 'Male' label."""
        age_est = _make_age_estimator(22)
        hair_clf = _make_hair_classifier("short", 0.88)
        gender_pred = _make_gender_predictor("Female", 0.95)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.label == "Male"
        assert result.confidence == 0.88
        assert result.estimated_age == 22
        assert result.age_group == "target"

    def test_undetermined_hair_maps_to_undetermined(self, router, dummy_image):
        """Undetermined hair label should produce 'Undetermined' prediction."""
        age_est = _make_age_estimator(28)
        hair_clf = _make_hair_classifier("Undetermined", 0.52)
        gender_pred = _make_gender_predictor("Male", 0.70)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.label == "Undetermined"
        assert result.confidence == 0.52
        assert result.estimated_age == 28
        assert result.age_group == "target"

    def test_gender_predictor_not_called_for_target_group(self, router, dummy_image):
        """Gender predictor should NOT be called for target age group."""
        age_est = _make_age_estimator(20)
        hair_clf = _make_hair_classifier("long", 0.90)
        gender_pred = _make_gender_predictor("Male", 0.80)

        router.route(dummy_image, age_est, hair_clf, gender_pred)

        gender_pred.predict.assert_not_called()

    def test_boundary_age_20(self, router, dummy_image):
        """Age 20 is the lower boundary of target group."""
        age_est = _make_age_estimator(20)
        hair_clf = _make_hair_classifier("short", 0.75)
        gender_pred = _make_gender_predictor("Female", 0.90)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.age_group == "target"
        assert result.label == "Male"

    def test_boundary_age_30(self, router, dummy_image):
        """Age 30 is the upper boundary of target group."""
        age_est = _make_age_estimator(30)
        hair_clf = _make_hair_classifier("long", 0.85)
        gender_pred = _make_gender_predictor("Male", 0.80)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.age_group == "target"
        assert result.label == "Female"


class TestDecisionRouterOutsideAgeGroup:
    """Tests for outside age group routing — standard prediction path."""

    def test_outside_uses_gender_predictor(self, router, dummy_image):
        """Outside age group should use gender predictor for prediction."""
        age_est = _make_age_estimator(35)
        hair_clf = _make_hair_classifier("long", 0.95)
        gender_pred = _make_gender_predictor("Male", 0.78)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.label == "Male"
        assert result.confidence == 0.78
        assert result.estimated_age == 35
        assert result.age_group == "outside"
        assert "Outside" in result.age_group_display
        assert "standard" in result.age_group_display

    def test_hair_classifier_not_called_for_outside_group(self, router, dummy_image):
        """Hair classifier should NOT be called for outside age group."""
        age_est = _make_age_estimator(15)
        hair_clf = _make_hair_classifier("long", 0.90)
        gender_pred = _make_gender_predictor("Female", 0.85)

        router.route(dummy_image, age_est, hair_clf, gender_pred)

        hair_clf.predict.assert_not_called()

    def test_boundary_age_19(self, router, dummy_image):
        """Age 19 is just outside the target group lower boundary."""
        age_est = _make_age_estimator(19)
        hair_clf = _make_hair_classifier("long", 0.90)
        gender_pred = _make_gender_predictor("Female", 0.82)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.age_group == "outside"
        assert result.label == "Female"

    def test_boundary_age_31(self, router, dummy_image):
        """Age 31 is just outside the target group upper boundary."""
        age_est = _make_age_estimator(31)
        hair_clf = _make_hair_classifier("short", 0.90)
        gender_pred = _make_gender_predictor("Male", 0.91)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert result.age_group == "outside"
        assert result.label == "Male"

    def test_hair_label_does_not_affect_outside_result(self, router, dummy_image):
        """For outside group, hair label must not influence the result."""
        age_est = _make_age_estimator(40)
        gender_pred = _make_gender_predictor("Female", 0.88)

        # Even with "short" hair label from classifier, gender predictor result should be used
        hair_clf_short = _make_hair_classifier("short", 0.95)
        result = router.route(dummy_image, age_est, hair_clf_short, gender_pred)

        assert result.label == "Female"
        assert result.confidence == 0.88


class TestPredictionResultStructure:
    """Tests that PredictionResult is always fully populated."""

    def test_target_result_has_all_fields(self, router, dummy_image):
        """Target group result has all required fields."""
        age_est = _make_age_estimator(25)
        hair_clf = _make_hair_classifier("long", 0.90)
        gender_pred = _make_gender_predictor("Male", 0.80)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert isinstance(result, PredictionResult)
        assert isinstance(result.label, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.estimated_age, int)
        assert isinstance(result.age_group, str)
        assert isinstance(result.age_group_display, str)

    def test_outside_result_has_all_fields(self, router, dummy_image):
        """Outside group result has all required fields."""
        age_est = _make_age_estimator(50)
        hair_clf = _make_hair_classifier("long", 0.90)
        gender_pred = _make_gender_predictor("Female", 0.77)

        result = router.route(dummy_image, age_est, hair_clf, gender_pred)

        assert isinstance(result, PredictionResult)
        assert isinstance(result.label, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.estimated_age, int)
        assert isinstance(result.age_group, str)
        assert isinstance(result.age_group_display, str)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
import hypothesis.strategies as st


# Feature: long-hair-gender-identification, Property 2: Biased prediction maps hair label to gender deterministically
# **Validates: Requirements 3.2, 3.3, 3.4**
@given(hair_label=st.sampled_from(["long", "short"]))
@settings(max_examples=100)
def test_biased_prediction_deterministic_property(hair_label):
    """Property 2: long→Female, short→Male for target age group.

    For any individual in the Target_Age_Group (age 20–30) with a definite
    hair label, the DecisionRouter assigns gender deterministically based
    solely on hair length.
    """
    router = DecisionRouter()
    image = np.zeros((224, 224, 3), dtype=np.float32)

    age_est = MagicMock()
    age_est.predict.return_value = 25  # target age group

    hair_clf = MagicMock()
    hair_clf.predict.return_value = (hair_label, 0.9)

    gender_pred = MagicMock()
    gender_pred.predict.return_value = ("Male", 0.8)

    result = router.route(image, age_est, hair_clf, gender_pred)

    expected = "Female" if hair_label == "long" else "Male"
    assert result.label == expected


# Feature: long-hair-gender-identification, Property 3: Outside_Age_Group predictions are hair-length-invariant
# **Validates: Requirements 3.6, 4.1, 4.2**
@given(
    age=st.integers(min_value=1, max_value=19) | st.integers(min_value=31, max_value=100),
    hair_label=st.sampled_from(["long", "short", "Undetermined"]),
)
@settings(max_examples=100)
def test_outside_group_hair_invariant_property(age, hair_label):
    """Property 3: For outside age group, result is independent of hair label.

    Given a fixed gender_predictor output, the prediction result must be
    identical regardless of what the hair_classifier would return, because
    the hair classifier is never consulted for outside-age-group individuals.
    """
    router = DecisionRouter()
    image = np.zeros((224, 224, 3), dtype=np.float32)

    age_est = MagicMock()
    age_est.predict.return_value = age

    hair_clf = MagicMock()
    hair_clf.predict.return_value = (hair_label, 0.9)

    gender_pred = MagicMock()
    gender_pred.predict.return_value = ("Female", 0.85)

    result = router.route(image, age_est, hair_clf, gender_pred)

    # Result should always be what gender_predictor returns, regardless of hair_label
    assert result.label == "Female"
    assert result.confidence == 0.85
    assert result.age_group == "outside"
