"""Decision router for gender prediction based on age group.

Contains the PredictionResult dataclass and helper functions for
age-group classification and routing logic.
"""

from dataclasses import dataclass

import numpy as np

from src.models.age_estimator import AgeEstimator
from src.models.hair_length_classifier import HairLengthClassifier
from src.models.gender_predictor import GenderPredictor


@dataclass
class PredictionResult:
    """Encapsulates the final prediction output.

    Attributes:
        label: Predicted gender — "Male", "Female", or "Undetermined".
        confidence: Model confidence score in [0.0, 1.0].
        estimated_age: Estimated age as an integer in [1, 100].
        age_group: Classification — "target" (age 20–30) or "outside".
        age_group_display: Human-readable age group description shown in the GUI.
    """

    label: str  # "Male" | "Female" | "Undetermined"
    confidence: float  # 0.0 – 1.0
    estimated_age: int  # 1 – 100
    age_group: str  # "target" | "outside"
    age_group_display: str  # e.g. "Age 20–30 (biased prediction)"


def age_group_classify(age: int) -> str:
    """Classify an estimated age into a target or outside group.

    Args:
        age: Estimated age as an integer.

    Returns:
        "target" if 20 <= age <= 30, otherwise "outside".
    """
    if 20 <= age <= 30:
        return "target"
    return "outside"


# Display strings for each age group
_TARGET_DISPLAY = "\u0041ge 20\u201330 (biased prediction)"
_OUTSIDE_DISPLAY = "Outside age range (standard prediction)"

# Hair label to gender mapping for the Target_Age_Group
_HAIR_TO_GENDER = {
    "long": "Female",
    "short": "Male",
    "Undetermined": "Undetermined",
}


def format_confidence(confidence: float) -> str:
    """Format a confidence score as a percentage string.

    Args:
        confidence: A float value in [0.0, 1.0].

    Returns:
        A string matching the pattern ``\\d+\\.\\d{2}%``, e.g. ``"87.45%"``.
    """
    return f"{confidence * 100:.2f}%"


class DecisionRouter:
    """Routes gender prediction through the appropriate path based on age group.

    For Target_Age_Group (age 20–30 inclusive):
        Uses Hair_Length_Classifier output to determine gender.
        "long" → "Female", "short" → "Male", "Undetermined" → "Undetermined".

    For Outside_Age_Group (age < 20 or age > 30):
        Uses Gender_Predictor directly; hair length does not affect the result.
    """

    def route(
        self,
        image: np.ndarray,
        age_estimator: AgeEstimator,
        hair_classifier: HairLengthClassifier,
        gender_predictor: GenderPredictor,
    ) -> PredictionResult:
        """Run the decision routing pipeline on a preprocessed image.

        Args:
            image: Preprocessed image array of shape (224, 224, 3).
            age_estimator: Loaded AgeEstimator instance.
            hair_classifier: Loaded HairLengthClassifier instance.
            gender_predictor: Loaded GenderPredictor instance.

        Returns:
            A fully populated PredictionResult with label, confidence,
            estimated age, age group, and age group display string.
        """
        # Step 1: Estimate age
        age = age_estimator.predict(image)

        # Step 2: Classify age group
        group = age_group_classify(age)

        # Step 3: Route based on age group
        if group == "target":
            # Biased prediction path — use hair length to determine gender
            hair_label, confidence = hair_classifier.predict(image)
            label = _HAIR_TO_GENDER[hair_label]
            return PredictionResult(
                label=label,
                confidence=confidence,
                estimated_age=age,
                age_group="target",
                age_group_display=_TARGET_DISPLAY,
            )
        else:
            # Standard prediction path — use gender predictor only
            label, confidence = gender_predictor.predict(image)
            return PredictionResult(
                label=label,
                confidence=confidence,
                estimated_age=age,
                age_group="outside",
                age_group_display=_OUTSIDE_DISPLAY,
            )

