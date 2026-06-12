"""
Property-based test for DatasetLoader split reproducibility.

Property 10: Training reproducibility with fixed seed
Validates: Requirements 9.1

For any random seed S, two DatasetLoader instances with the same seed S,
given the same input DataFrame, produce identical train/val/test splits.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st

from src.data.dataset_loader import DatasetLoader


def _create_sample_dataframe(n_samples: int = 100) -> pd.DataFrame:
    """
    Create a sample DataFrame mimicking UTKFace structure with sufficient
    samples per stratum for stratified splitting.
    """
    np.random.seed(0)  # Fixed seed for deterministic test data creation

    ages = []
    genders = []
    hair_labels = []
    paths = []

    # Ensure we have enough samples in each stratum (age_group x gender)
    # Strata: target_Male, target_Female, outside_Male, outside_Female
    for i in range(n_samples):
        if i % 4 == 0:
            age = np.random.randint(20, 31)  # target group
            gender = "Male"
        elif i % 4 == 1:
            age = np.random.randint(20, 31)  # target group
            gender = "Female"
        elif i % 4 == 2:
            age = np.random.choice(list(range(1, 20)) + list(range(31, 101)))  # outside
            gender = "Male"
        else:
            age = np.random.choice(list(range(1, 20)) + list(range(31, 101)))  # outside
            gender = "Female"

        ages.append(age)
        genders.append(gender)
        hair_labels.append(np.random.choice(["long", "short"]))
        paths.append(f"/fake/path/image_{i}.jpg")

    return pd.DataFrame({
        'path': paths,
        'age': ages,
        'gender': genders,
        'hair_label': hair_labels,
    })


# **Validates: Requirements 9.1**
@given(seed=st.integers(min_value=0, max_value=2**31))
@settings(max_examples=100)
def test_split_reproducibility_with_fixed_seed(seed):
    """
    Property 10: Training reproducibility with fixed seed.

    For any seed, two DatasetLoader instances with the same seed produce
    identical split indices when given the same input DataFrame.
    """
    # Create the same input DataFrame for both loaders
    df = _create_sample_dataframe(n_samples=100)

    # Use a temporary directory to satisfy DatasetLoader's __init__ requirement
    tmp_dir = tempfile.gettempdir()

    # Create two independent DatasetLoader instances with the same seed
    loader1 = DatasetLoader(zip_path="dummy.zip", extract_dir=tmp_dir, seed=seed)
    loader2 = DatasetLoader(zip_path="dummy.zip", extract_dir=tmp_dir, seed=seed)

    # Perform splits
    train1, val1, test1 = loader1.split(df)
    train2, val2, test2 = loader2.split(df)

    # Assert identical splits (same rows in same order)
    pd.testing.assert_frame_equal(
        train1.reset_index(drop=True),
        train2.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        val1.reset_index(drop=True),
        val2.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        test1.reset_index(drop=True),
        test2.reset_index(drop=True),
    )
