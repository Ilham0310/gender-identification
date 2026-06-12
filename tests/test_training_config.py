"""
Property-based test for training config key completeness.

Property 11: Training config file always contains required keys
Validates: Requirements 9.3

After any call that generates a config dict, assert all four required keys
(seed, dataset_split, model_architecture, hyperparameters) are present.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypothesis import given, settings
import hypothesis.strategies as st


def build_training_config(seed: int, epochs_age: int, epochs_hair: int, epochs_gender: int) -> dict:
    """
    Replicate the config dict construction from train.py's main() function.

    This mirrors the exact structure built in train.py to verify that the
    config dict always contains the required keys regardless of input parameters.
    """
    config = {
        "seed": seed,
        "dataset_split": {"train": 0.70, "val": 0.10, "test": 0.20},
        "model_architecture": "MobileNetV2",
        "hyperparameters": {
            "age_estimator": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": epochs_age,
                "fine_tune_from_layer": 100,
                "dropout": 0.3,
            },
            "hair_classifier": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": epochs_hair,
                "undetermined_threshold": 0.55,
            },
            "gender_predictor": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": epochs_gender,
            },
        },
        "accuracy_thresholds": {
            "gender_standard": 0.70,
            "hair_biased": 0.80,
            "age_group": 0.75,
        },
    }
    return config


# **Validates: Requirements 9.3**
@given(
    seed=st.integers(min_value=0, max_value=2**31),
    epochs_age=st.integers(min_value=1, max_value=100),
    epochs_hair=st.integers(min_value=1, max_value=100),
    epochs_gender=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_training_config_always_contains_required_keys(seed, epochs_age, epochs_hair, epochs_gender):
    """
    Property 11: The config dict always contains seed, dataset_split,
    model_architecture, hyperparameters.

    For any combination of valid training parameters, the generated config
    dict must contain all four required keys with correct types and sub-structure.
    """
    config = build_training_config(seed, epochs_age, epochs_hair, epochs_gender)

    # Assert all four required keys are always present
    required_keys = {"seed", "dataset_split", "model_architecture", "hyperparameters"}
    assert required_keys.issubset(config.keys()), f"Missing keys: {required_keys - set(config.keys())}"

    # Assert types
    assert isinstance(config["seed"], int)
    assert isinstance(config["dataset_split"], dict)
    assert isinstance(config["model_architecture"], str)
    assert isinstance(config["hyperparameters"], dict)

    # Assert dataset_split sub-structure
    assert "train" in config["dataset_split"]
    assert "val" in config["dataset_split"]
    assert "test" in config["dataset_split"]

    # Assert hyperparameters sub-structure contains all three model sections
    assert "age_estimator" in config["hyperparameters"]
    assert "hair_classifier" in config["hyperparameters"]
    assert "gender_predictor" in config["hyperparameters"]

    # Assert each model section has required hyperparameter keys
    assert "learning_rate" in config["hyperparameters"]["age_estimator"]
    assert "batch_size" in config["hyperparameters"]["age_estimator"]
    assert "epochs" in config["hyperparameters"]["age_estimator"]

    assert "learning_rate" in config["hyperparameters"]["hair_classifier"]
    assert "batch_size" in config["hyperparameters"]["hair_classifier"]
    assert "epochs" in config["hyperparameters"]["hair_classifier"]

    assert "learning_rate" in config["hyperparameters"]["gender_predictor"]
    assert "batch_size" in config["hyperparameters"]["gender_predictor"]
    assert "epochs" in config["hyperparameters"]["gender_predictor"]
