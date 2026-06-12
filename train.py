#!/usr/bin/env python3
"""
Long-Hair Gender Identification Training Pipeline

Main entry point for training the machine learning models:
- AgeEstimator: age regression (Huber loss)
- HairLengthClassifier: hair pseudo-label classification (binary crossentropy)
- GenderPredictor: standard gender classification (binary crossentropy)

After training, evaluates each model on the test split and compares against
accuracy thresholds, logging WARNINGs for any missed threshold.
"""

import sys
import os
import json
import argparse
import random
import logging

from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import tensorflow as tf

from data.dataset_loader import DatasetLoader
from data.preprocessor import Preprocessor
from models.age_estimator import AgeEstimator
from models.hair_length_classifier import HairLengthClassifier
from models.gender_predictor import GenderPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Accuracy thresholds (from requirements 7.1, 7.2, 7.3)
THRESHOLD_GENDER_STANDARD = 0.70
THRESHOLD_HAIR_BIASED = 0.80
THRESHOLD_AGE_GROUP = 0.75


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_age_dataset(df, preprocessor: Preprocessor, batch_size: int = 32, augment: bool = False):
    """
    Create a tf.data.Dataset for age estimation training.

    Returns batches of (image, age) pairs.
    """
    paths = df["path"].values
    ages = df["age"].values.astype(np.float32)

    def generator():
        for path, age in zip(paths, ages):
            try:
                image = preprocessor.preprocess(path)
                if augment:
                    image = preprocessor.augment(image)
                yield image, age
            except Exception:
                continue

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_hair_dataset(df, preprocessor: Preprocessor, batch_size: int = 32, augment: bool = False):
    """
    Create a tf.data.Dataset for hair length classification.

    Filters to target-age-group samples (age 20-30) with valid hair labels.
    Returns batches of (image, label) where label: 1=long, 0=short.
    """
    # Filter to target age group with valid hair labels
    target_df = df[(df["age"] >= 20) & (df["age"] <= 30)].copy()
    target_df = target_df[target_df["hair_label"].isin(["long", "short"])]

    if len(target_df) == 0:
        logger.warning("No target-age-group samples with valid hair labels found.")
        return None

    paths = target_df["path"].values
    labels = (target_df["hair_label"] == "long").astype(np.float32).values

    def generator():
        for path, label in zip(paths, labels):
            try:
                image = preprocessor.preprocess(path)
                if augment:
                    image = preprocessor.augment(image)
                yield image, label
            except Exception:
                continue

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_gender_dataset(df, preprocessor: Preprocessor, batch_size: int = 32, augment: bool = False):
    """
    Create a tf.data.Dataset for gender prediction.

    Filters to outside-age-group samples (age <20 or >30).
    Returns batches of (image, label) where label: 1=Female, 0=Male.
    """
    # Filter to outside age group
    outside_df = df[(df["age"] < 20) | (df["age"] > 30)].copy()

    if len(outside_df) == 0:
        logger.warning("No outside-age-group samples found.")
        return None

    paths = outside_df["path"].values
    labels = (outside_df["gender"] == "Female").astype(np.float32).values

    def generator():
        for path, label in zip(paths, labels):
            try:
                image = preprocessor.preprocess(path)
                if augment:
                    image = preprocessor.augment(image)
                yield image, label
            except Exception:
                continue

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_age_group_accuracy(age_estimator: AgeEstimator, test_df, preprocessor: Preprocessor) -> float:
    """
    Evaluate age-group classification accuracy on the test split.

    Returns the fraction of samples where the predicted age group matches the true age group.
    """
    correct = 0
    total = 0

    for _, row in test_df.iterrows():
        try:
            image = preprocessor.preprocess(row["path"])
            predicted_age = age_estimator.predict(image)
            true_age = row["age"]

            # Determine true and predicted age groups
            true_target = 20 <= true_age <= 30
            pred_target = 20 <= predicted_age <= 30

            if true_target == pred_target:
                correct += 1
            total += 1
        except Exception as e:
            logger.debug(f"Skipping sample during age eval: {e}")
            continue

    if total == 0:
        return 0.0
    return correct / total


def evaluate_hair_biased_accuracy(hair_classifier: HairLengthClassifier, test_df, preprocessor: Preprocessor) -> float:
    """
    Evaluate hair-length-based biased prediction accuracy on target-age-group test samples.

    Measures correct hair-length-to-gender mappings:
    - long hair → Female (correct if actual gender is Female)
    - short hair → Male (correct if actual gender is Male)
    """
    target_df = test_df[(test_df["age"] >= 20) & (test_df["age"] <= 30)].copy()
    target_df = target_df[target_df["hair_label"].isin(["long", "short"])]

    if len(target_df) == 0:
        logger.warning("No target-age-group test samples with valid hair labels for evaluation.")
        return 0.0

    correct = 0
    total = 0

    for _, row in target_df.iterrows():
        try:
            image = preprocessor.preprocess(row["path"])
            hair_label, confidence = hair_classifier.predict(image)

            if hair_label == "Undetermined":
                continue

            # Biased prediction: long → Female, short → Male
            predicted_gender = "Female" if hair_label == "long" else "Male"
            true_gender = row["gender"]

            if predicted_gender == true_gender:
                correct += 1
            total += 1
        except Exception as e:
            logger.debug(f"Skipping sample during hair eval: {e}")
            continue

    if total == 0:
        return 0.0
    return correct / total


def evaluate_gender_standard_accuracy(gender_predictor: GenderPredictor, test_df, preprocessor: Preprocessor) -> float:
    """
    Evaluate standard gender prediction accuracy on outside-age-group test samples.
    """
    outside_df = test_df[(test_df["age"] < 20) | (test_df["age"] > 30)]

    if len(outside_df) == 0:
        logger.warning("No outside-age-group test samples for evaluation.")
        return 0.0

    correct = 0
    total = 0

    for _, row in outside_df.iterrows():
        try:
            image = preprocessor.preprocess(row["path"])
            predicted_gender, confidence = gender_predictor.predict(image)
            true_gender = row["gender"]

            if predicted_gender == true_gender:
                correct += 1
            total += 1
        except Exception as e:
            logger.debug(f"Skipping sample during gender eval: {e}")
            continue

    if total == 0:
        return 0.0
    return correct / total


def main():
    """Main entry point for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train Long-Hair Gender Identification models")
    parser.add_argument(
        "--zip-path", type=str, default="archive.zip",
        help="Path to archive.zip containing UTKFace dataset (default: archive.zip)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (auto-generated if not provided)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Directory to save trained model weights (default: models)"
    )
    parser.add_argument(
        "--epochs-age", type=int, default=30,
        help="Number of training epochs for AgeEstimator (default: 30)"
    )
    parser.add_argument(
        "--epochs-hair", type=int, default=20,
        help="Number of training epochs for HairLengthClassifier (default: 20)"
    )
    parser.add_argument(
        "--epochs-gender", type=int, default=20,
        help="Number of training epochs for GenderPredictor (default: 20)"
    )

    args = parser.parse_args()

    # --- Seed handling ---
    if args.seed is None:
        seed = random.randint(0, 2**31)
        print(f"No seed provided. Auto-generated seed: {seed}")
        logger.info(f"Auto-generated seed: {seed}")
    else:
        seed = args.seed
        logger.info(f"Using provided seed: {seed}")

    set_seeds(seed)

    # --- Configuration ---
    config = {
        "seed": seed,
        "dataset_split": {"train": 0.70, "val": 0.10, "test": 0.20},
        "model_architecture": "MobileNetV2",
        "hyperparameters": {
            "age_estimator": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": args.epochs_age,
                "fine_tune_from_layer": 100,
                "dropout": 0.3,
            },
            "hair_classifier": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": args.epochs_hair,
                "undetermined_threshold": 0.55,
            },
            "gender_predictor": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": args.epochs_gender,
            },
        },
        "accuracy_thresholds": {
            "gender_standard": THRESHOLD_GENDER_STANDARD,
            "hair_biased": THRESHOLD_HAIR_BIASED,
            "age_group": THRESHOLD_AGE_GROUP,
        },
    }

    batch_size = 32

    # --- Data loading ---
    logger.info("Loading dataset...")
    loader = DatasetLoader(zip_path=args.zip_path, extract_dir="data", seed=seed)
    df = loader.load()
    logger.info(f"Dataset loaded: {len(df)} samples")

    # --- Data splitting ---
    logger.info("Splitting dataset into train/val/test...")
    train_df, val_df, test_df = loader.split(df)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- Preprocessor ---
    preprocessor = Preprocessor()

    # --- Create datasets ---
    logger.info("Creating training datasets...")

    # Age datasets (all samples)
    train_age_ds = create_age_dataset(train_df, preprocessor, batch_size=batch_size, augment=True)
    val_age_ds = create_age_dataset(val_df, preprocessor, batch_size=batch_size, augment=False)

    # Hair datasets (target age group only, with valid hair labels)
    train_hair_ds = create_hair_dataset(train_df, preprocessor, batch_size=batch_size, augment=True)
    val_hair_ds = create_hair_dataset(val_df, preprocessor, batch_size=batch_size, augment=False)

    # Gender datasets (outside age group only)
    train_gender_ds = create_gender_dataset(train_df, preprocessor, batch_size=batch_size, augment=True)
    val_gender_ds = create_gender_dataset(val_df, preprocessor, batch_size=batch_size, augment=False)

    # --- Train AgeEstimator ---
    logger.info("=" * 60)
    logger.info("Training AgeEstimator...")
    logger.info("=" * 60)
    age_estimator = AgeEstimator()
    age_estimator.train(train_age_ds, val_age_ds, config)
    logger.info("AgeEstimator training complete.")

    # --- Train HairLengthClassifier ---
    logger.info("=" * 60)
    logger.info("Training HairLengthClassifier...")
    logger.info("=" * 60)
    hair_classifier = HairLengthClassifier()
    if train_hair_ds is not None and val_hair_ds is not None:
        hair_classifier.train(train_hair_ds, val_hair_ds, config)
        logger.info("HairLengthClassifier training complete.")
    else:
        logger.warning("Skipping HairLengthClassifier training: no valid target-age-group samples.")

    # --- Train GenderPredictor ---
    logger.info("=" * 60)
    logger.info("Training GenderPredictor...")
    logger.info("=" * 60)
    gender_predictor = GenderPredictor()
    if train_gender_ds is not None and val_gender_ds is not None:
        gender_predictor.train(train_gender_ds, val_gender_ds, config)
        logger.info("GenderPredictor training complete.")
    else:
        logger.warning("Skipping GenderPredictor training: no valid outside-age-group samples.")

    # --- Evaluation on test split ---
    logger.info("=" * 60)
    logger.info("Evaluating models on test split...")
    logger.info("=" * 60)

    # Evaluate age-group classification accuracy
    age_group_acc = evaluate_age_group_accuracy(age_estimator, test_df, preprocessor)
    logger.info(f"Age-group classification accuracy: {age_group_acc:.4f} (threshold: {THRESHOLD_AGE_GROUP:.2f})")

    # Evaluate hair-biased prediction accuracy
    hair_biased_acc = evaluate_hair_biased_accuracy(hair_classifier, test_df, preprocessor)
    logger.info(f"Hair-biased prediction accuracy: {hair_biased_acc:.4f} (threshold: {THRESHOLD_HAIR_BIASED:.2f})")

    # Evaluate standard gender prediction accuracy
    gender_standard_acc = evaluate_gender_standard_accuracy(gender_predictor, test_df, preprocessor)
    logger.info(f"Gender standard prediction accuracy: {gender_standard_acc:.4f} (threshold: {THRESHOLD_GENDER_STANDARD:.2f})")

    # --- Threshold comparison (Requirement 7.4) ---
    if gender_standard_acc < THRESHOLD_GENDER_STANDARD:
        delta = THRESHOLD_GENDER_STANDARD - gender_standard_acc
        logger.warning(
            f"THRESHOLD MISSED: Gender standard accuracy {gender_standard_acc:.4f} "
            f"is below threshold {THRESHOLD_GENDER_STANDARD:.2f} by {delta:.4f}"
        )

    if hair_biased_acc < THRESHOLD_HAIR_BIASED:
        delta = THRESHOLD_HAIR_BIASED - hair_biased_acc
        logger.warning(
            f"THRESHOLD MISSED: Hair biased accuracy {hair_biased_acc:.4f} "
            f"is below threshold {THRESHOLD_HAIR_BIASED:.2f} by {delta:.4f}"
        )

    if age_group_acc < THRESHOLD_AGE_GROUP:
        delta = THRESHOLD_AGE_GROUP - age_group_acc
        logger.warning(
            f"THRESHOLD MISSED: Age-group accuracy {age_group_acc:.4f} "
            f"is below threshold {THRESHOLD_AGE_GROUP:.2f} by {delta:.4f}"
        )

    # --- Persist model weights (Requirement 6.4) ---
    logger.info("=" * 60)
    logger.info("Saving model weights and config...")
    logger.info("=" * 60)

    os.makedirs(args.model_dir, exist_ok=True)

    age_estimator.save(os.path.join(args.model_dir, "age_estimator.keras"))
    logger.info(f"Saved AgeEstimator to {args.model_dir}/age_estimator.keras")

    hair_classifier.save(os.path.join(args.model_dir, "hair_classifier.keras"))
    logger.info(f"Saved HairLengthClassifier to {args.model_dir}/hair_classifier.keras")

    gender_predictor.save(os.path.join(args.model_dir, "gender_predictor.keras"))
    logger.info(f"Saved GenderPredictor to {args.model_dir}/gender_predictor.keras")

    # --- Write config.json (Requirement 9.3) ---
    config["validation_accuracy_achieved"] = {
        "gender_standard": gender_standard_acc,
        "hair_biased": hair_biased_acc,
        "age_group": age_group_acc,
    }

    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {config_path}")

    logger.info("Training pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
