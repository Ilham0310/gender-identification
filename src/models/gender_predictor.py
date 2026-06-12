"""
GenderPredictor implementation for the Long-Hair Gender Identification system.

This module implements a MobileNetV2-based binary classifier that predicts gender
(Male vs Female) from facial images. Used for the Outside_Age_Group (below 20 or above 30)
to provide standard, unbiased gender prediction based on facial features.
"""

import os
from typing import Tuple, Dict, Any

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    _import_error = e
    # Create dummy references for when TensorFlow is not available
    class Model:
        pass
    class MobileNetV2:
        pass
    class GlobalAveragePooling2D:
        pass
    class Dense:
        pass
    class Dropout:
        pass
    class Adam:
        pass
    class EarlyStopping:
        pass
    class ModelCheckpoint:
        pass
    try:
        import numpy as np
    except ImportError:
        np = None
    tf = None


class ModelLoadError(Exception):
    """Raised when model weights cannot be loaded from file."""
    pass


class GenderPredictor:
    """
    A MobileNetV2-based binary classifier for gender prediction.

    Architecture:
    - Backbone: MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
    - Global Average Pooling
    - Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)

    Sigmoid output mapping:
    - 0 → "Male"
    - 1 → "Female"
    - Threshold at 0.5
    """

    INPUT_SHAPE = (224, 224, 3)

    def __init__(self):
        """Initialize the GenderPredictor with MobileNetV2 architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(f"TensorFlow is required but not available: {_import_error}")

        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the MobileNetV2-based architecture."""
        # MobileNetV2 backbone with ImageNet weights
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.INPUT_SHAPE
        )

        # Add custom head for gender classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='gender_dense_1')(x)
        x = Dropout(0.3, name='gender_dropout')(x)
        predictions = Dense(1, activation='sigmoid', name='gender_output')(x)

        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict gender from an input image.

        Args:
            image: Preprocessed image array of shape (224, 224, 3) with values in [-1, 1]

        Returns:
            Tuple of (gender_label, confidence) where:
            - gender_label ∈ {"Male", "Female"}
            - confidence is the model's confidence score (0.0-1.0)
            - Sigmoid output < 0.5 → "Male", >= 0.5 → "Female"

        Raises:
            ValueError: If model is not loaded or image has wrong shape
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        if image.shape != self.INPUT_SHAPE:
            raise ValueError(f"Expected image shape {self.INPUT_SHAPE}, got {image.shape}")

        # Ensure image is in batch format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Get model prediction
        prediction = self.model.predict(image, verbose=0)[0][0]  # Single scalar output

        # Determine gender label and confidence
        # Sigmoid output: 0 → Male, 1 → Female, threshold at 0.5
        if prediction >= 0.5:
            gender_label = "Female"
            confidence = float(prediction)
        else:
            gender_label = "Male"
            confidence = float(1.0 - prediction)

        return gender_label, confidence

    def train(self, train_ds, val_ds, config: Dict[str, Any]):
        """
        Train the gender predictor model.

        Args:
            train_ds: Training dataset (tf.data.Dataset)
            val_ds: Validation dataset (tf.data.Dataset)
            config: Training configuration dictionary containing hyperparameters

        Returns:
            Training history object

        Raises:
            ValueError: If model is not built or config is invalid
        """
        if self.model is None:
            raise ValueError("Model not built. Call _build_model() first.")

        # Extract hyperparameters from config
        gender_config = config.get('hyperparameters', {}).get('gender_predictor', {})
        learning_rate = gender_config.get('learning_rate', 0.0001)
        epochs = gender_config.get('epochs', 20)

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # Train the model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def load(self, weights_path: str) -> None:
        """
        Load model weights from file.

        Args:
            weights_path: Path to the saved model weights file (.keras format)

        Raises:
            ModelLoadError: If weights file is missing or corrupted
        """
        try:
            if not os.path.exists(weights_path):
                raise ModelLoadError(
                    f"Model weights file not found: {weights_path}. "
                    "Please run train.py to train the model first."
                )

            # Build model if not already built
            if self.model is None:
                self._build_model()

            # Load weights
            self.model.load_weights(weights_path)

        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            else:
                raise ModelLoadError(
                    f"Failed to load model weights from {weights_path}: {str(e)}. "
                    "The file may be corrupted. Please run train.py to retrain the model."
                )

    def save(self, weights_path: str) -> None:
        """
        Save model weights to file.

        Args:
            weights_path: Path where to save the model weights (.keras format)

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not built. Cannot save weights.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        # Save the model weights
        self.model.save_weights(weights_path)
