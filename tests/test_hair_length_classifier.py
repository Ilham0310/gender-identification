"""
Tests for HairLengthClassifier implementation.

Property-based and unit tests to verify the hair length classification logic
according to the design specifications.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from hypothesis import given, settings
import hypothesis.strategies as st
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hair_length_classifier import HairLengthClassifier, ModelLoadError


class TestHairLengthClassifierLogic:
    """Test the core logic without requiring TensorFlow."""
    
    def test_undetermined_threshold_constant(self):
        """Test that the undetermined threshold is set correctly."""
        assert HairLengthClassifier.UNDETERMINED_THRESHOLD == 0.55
    
    def test_input_shape_constant(self):
        """Test that input shape is set correctly for MobileNetV2."""
        assert HairLengthClassifier.INPUT_SHAPE == (224, 224, 3)
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_predict_long_hair_high_confidence(self):
        """Test prediction logic for long hair with high confidence."""
        # Mock the model prediction
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.85]])  # High probability for long hair
        
        classifier = HairLengthClassifier()
        classifier.model = mock_model
        
        # Test input
        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        
        result = classifier.predict(test_image)
        
        # Should predict "long" with confidence 0.85
        hair_label, confidence = result
        assert hair_label == "long"
        assert confidence == 0.85
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_predict_short_hair_high_confidence(self):
        """Test prediction logic for short hair with high confidence."""
        # Mock the model prediction
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.15]])  # Low probability (short hair)
        
        classifier = HairLengthClassifier()
        classifier.model = mock_model
        
        # Test input
        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        
        result = classifier.predict(test_image)
        
        # Should predict "short" with confidence 0.85 (1.0 - 0.15)
        hair_label, confidence = result
        assert hair_label == "short"
        assert confidence == 0.85
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_predict_undetermined_low_confidence(self):
        """Test prediction logic when confidence is below threshold."""
        # Mock the model prediction with low confidence
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.52]])  # Close to 0.5, low confidence
        
        classifier = HairLengthClassifier()
        classifier.model = mock_model
        
        # Test input
        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        
        result = classifier.predict(test_image)
        
        # Should predict "Undetermined" because max(0.52, 0.48) = 0.52 < 0.55
        hair_label, confidence = result
        assert hair_label == "Undetermined"
        assert confidence == 0.52
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_predict_wrong_image_shape(self):
        """Test that wrong image shape raises ValueError."""
        classifier = HairLengthClassifier()
        classifier.model = Mock()
        
        # Wrong shape image
        wrong_shape_image = np.random.random((128, 128, 3)).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected image shape"):
            classifier.predict(wrong_shape_image)
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_predict_no_model_loaded(self):
        """Test that prediction fails when model is not loaded."""
        classifier = HairLengthClassifier()
        classifier.model = None  # Force no model
        
        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model not loaded"):
            classifier.predict(test_image)
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_load_missing_file(self):
        """Test that loading missing weights file raises ModelLoadError."""
        classifier = HairLengthClassifier()
        
        with pytest.raises(ModelLoadError, match="Model weights file not found"):
            classifier.load("/nonexistent/path/weights.keras")
    
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', False)
    def test_init_without_tensorflow(self):
        """Test that initialization fails gracefully without TensorFlow."""
        with pytest.raises(ImportError, match="TensorFlow is required"):
            HairLengthClassifier()


class TestHairLengthClassifierOutputDomain:
    """Property-based test for output domain validation (Property 5)."""

    @given(sigmoid_output=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=200)
    @patch('models.hair_length_classifier.TENSORFLOW_AVAILABLE', True)
    @patch.object(HairLengthClassifier, '_build_model', lambda self: None)
    def test_hair_classifier_output_domain_property(self, sigmoid_output):
        """
        Property 5: Hair_Length_Classifier output domain is bounded.

        For any raw sigmoid output in [0.0, 1.0], the returned hair_label
        is always a member of {"long", "short", "Undetermined"} and
        confidence is a float in [0.0, 1.0].

        **Validates: Requirements 3.1, 3.5**
        """
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[sigmoid_output]])

        classifier = HairLengthClassifier()
        classifier.model = mock_model

        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        hair_label, confidence = classifier.predict(test_image)

        assert hair_label in {"long", "short", "Undetermined"}
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])