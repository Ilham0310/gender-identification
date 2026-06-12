"""
HairLengthClassifier implementation for the Long-Hair Gender Identification system.

This module implements a MobileNetV2-based binary classifier that predicts hair length
(long vs short) from facial images. Used exclusively for the Target_Age_Group (20-30 years)
to enable biased gender prediction based on hair length.
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
    # Create dummy classes/functions for when TensorFlow is not available
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


class HairLengthClassifier:
    """
    A MobileNetV2-based binary classifier for hair length prediction.
    
    Architecture:
    - Backbone: MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
    - Global Average Pooling
    - Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
    
    The classifier returns "Undetermined" when confidence is below 0.55 threshold.
    """
    
    UNDETERMINED_THRESHOLD = 0.55
    INPUT_SHAPE = (224, 224, 3)
    
    def __init__(self):
        """Initialize the HairLengthClassifier with MobileNetV2 architecture."""
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
        
        # Add custom head for hair length classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='hair_dense_1')(x)
        x = Dropout(0.3, name='hair_dropout')(x)
        predictions = Dense(1, activation='sigmoid', name='hair_output')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict hair length from an input image.
        
        Args:
            image: Preprocessed image array of shape (224, 224, 3) with values in [-1, 1]
        
        Returns:
            Tuple of (hair_label, confidence) where:
            - hair_label ∈ {"long", "short", "Undetermined"}
            - confidence is the model's confidence score (0.0-1.0)
            - Returns "Undetermined" if confidence < 0.55
        
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
        
        # Calculate confidence as the maximum of long/short probabilities
        long_prob = prediction
        short_prob = 1.0 - prediction
        confidence = max(long_prob, short_prob)
        
        # Apply undetermined threshold
        if confidence < self.UNDETERMINED_THRESHOLD:
            return "Undetermined", confidence
        
        # Determine hair label based on prediction
        hair_label = "long" if prediction >= 0.5 else "short"
        
        return hair_label, confidence
    
    def train(self, train_ds, val_ds, config: Dict[str, Any]):
        """
        Train the hair length classifier.
        
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
        hair_config = config.get('hyperparameters', {}).get('hair_classifier', {})
        learning_rate = hair_config.get('learning_rate', 0.0001)
        epochs = hair_config.get('epochs', 20)
        
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
        if not os.path.exists(weights_path):
            raise ModelLoadError(
                f"Model weights file not found: {weights_path}. "
                "Please run train.py to train the model first."
            )
        
        try:
            # Load the entire model (architecture + weights)
            self.model = tf.keras.models.load_model(
                weights_path,
                compile=False  # Don't compile, we may want different metrics
            )
        except Exception as e:
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
        
        # Save the entire model (architecture + weights)
        self.model.save(weights_path)