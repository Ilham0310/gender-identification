"""
Age Estimator Model

This module implements the AgeEstimator class for age regression using MobileNetV2 backbone.
The model estimates age as an integer in the range [1, 100].
"""

import os
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.losses import Huber
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    _import_error = e
    # Provide stub so class body compiles
    class _StubModule:
        callbacks = type('callbacks', (), {'History': object})()
    keras = _StubModule()
    tf = None


class ModelLoadError(Exception):
    """Raised when model weights cannot be loaded."""
    pass


class AgeEstimator:
    """
    Age estimation model using MobileNetV2 backbone.
    
    Architecture:
    - MobileNetV2 (ImageNet pretrained, no top, 224x224x3 input)
    - GlobalAveragePooling2D
    - Dense(256, activation='relu')  
    - Dropout(0.3)
    - Dense(1, activation='relu') for age output
    
    Age output is clamped to [1, 100] and returned as integer.
    """
    
    def __init__(self):
        """Initialize the AgeEstimator model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(f"TensorFlow is required but not available: {_import_error}")
        self.model: Optional[keras.Model] = None
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the MobileNetV2-based age estimation model."""
        # Define input
        input_tensor = layers.Input(shape=(224, 224, 3))
        
        # MobileNetV2 backbone - ImageNet pretrained, no top layers
        backbone = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False
        )
        
        # Add custom head for age regression
        x = backbone.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='age_dense_256')(x)
        x = layers.Dropout(0.3, name='age_dropout')(x)
        age_output = layers.Dense(1, activation='relu', name='age_output')(x)
        
        # Create the model
        self.model = keras.Model(inputs=input_tensor, outputs=age_output, name='age_estimator')
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict age from preprocessed image.
        
        Args:
            image: Preprocessed image array of shape (224, 224, 3), values in [-1, 1]
        
        Returns:
            Estimated age as integer in range [1, 100]
            
        Raises:
            ValueError: If model is not loaded or image has wrong shape
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        if image.shape != (224, 224, 3):
            raise ValueError(f"Expected image shape (224, 224, 3), got {image.shape}")
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Run forward pass
        raw_output = self.model.predict(image, verbose=0)[0, 0]
        
        # Clamp to [1, 100] and convert to int
        age = int(np.clip(raw_output, 1, 100))
        
        return age
    
    def train(self, train_ds, val_ds, config: dict):
        """
        Train the age estimator model.
        
        Args:
            train_ds: Training dataset (tf.data.Dataset)
            val_ds: Validation dataset (tf.data.Dataset)  
            config: Training configuration dictionary with hyperparameters
        
        Returns:
            Training history object
            
        Raises:
            ValueError: If model is not built
        """
        if self.model is None:
            raise ValueError("Model not built.")
        
        # Extract hyperparameters from config
        age_config = config.get('hyperparameters', {}).get('age_estimator', {})
        learning_rate = age_config.get('learning_rate', 0.0001)
        epochs = age_config.get('epochs', 30)
        fine_tune_from_layer = age_config.get('fine_tune_from_layer', 100)
        
        # Compile model with Huber loss for robust age regression
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=Huber(delta=1.0),  # Robust to outliers
            metrics=[
                'mae',  # Mean Absolute Error
                self._age_group_accuracy_metric()  # Custom age group accuracy
            ]
        )
        
        # Set up callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Fine-tune from specified layer
        if fine_tune_from_layer is not None:
            # Freeze backbone layers except the last few
            backbone_layer_count = 0
            for layer in self.model.layers:
                if hasattr(layer, 'layers'):  # This is the MobileNetV2 backbone
                    backbone_layer_count = len(layer.layers)
                    for i, backbone_layer in enumerate(layer.layers):
                        if i < fine_tune_from_layer:
                            backbone_layer.trainable = False
                        else:
                            backbone_layer.trainable = True
        
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
            weights_path: Path to the model weights file (.keras or .h5)
            
        Raises:
            ModelLoadError: If file is missing or corrupted
        """
        if not os.path.exists(weights_path):
            raise ModelLoadError(
                f"Model weights file not found: {weights_path}. "
                "Please run the training script (train.py) to generate model weights."
            )
        
        try:
            # Load the entire model (architecture + weights)
            self.model = keras.models.load_model(
                weights_path,
                custom_objects={'Huber': Huber},
                compile=False  # Don't compile, we might want different metrics
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model weights from {weights_path}. "
                f"The file may be corrupted. Error: {str(e)}. "
                "Please run the training script (train.py) to regenerate model weights."
            )
    
    def save(self, weights_path: str) -> None:
        """
        Save model weights to file.
        
        Args:
            weights_path: Path where to save the model weights
            
        Raises:
            ValueError: If model is not built
        """
        if self.model is None:
            raise ValueError("Model not built. Cannot save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        
        # Save the entire model (architecture + weights)
        self.model.save(weights_path)
    
    def _age_group_accuracy_metric(self):
        """
        Custom metric to compute age group classification accuracy.
        Target age group: 20-30, Outside age group: <20 or >30
        """
        def age_group_accuracy(y_true, y_pred):
            # Clamp predictions to [1, 100]
            y_pred_clamped = tf.clip_by_value(y_pred, 1, 100)
            y_true_clamped = tf.clip_by_value(y_true, 1, 100)
            
            # Determine age groups
            y_true_target = tf.logical_and(y_true_clamped >= 20, y_true_clamped <= 30)
            y_pred_target = tf.logical_and(y_pred_clamped >= 20, y_pred_clamped <= 30)
            
            # Compute accuracy
            correct = tf.equal(y_true_target, y_pred_target)
            return tf.reduce_mean(tf.cast(correct, tf.float32))
        
        return age_group_accuracy