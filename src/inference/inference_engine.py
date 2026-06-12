"""Inference engine for the Long-Hair Gender Identification system.

This module orchestrates the full prediction pipeline:
1. Validate the input image (format, size, readability)
2. Detect faces using OpenCV Haar cascade
3. Preprocess the face region
4. Route through the DecisionRouter for final prediction

Custom exceptions provide user-readable error messages for every
failure mode without exposing internal stack traces.
"""

import os
import logging
from typing import Optional

import cv2
import numpy as np

from src.models.age_estimator import AgeEstimator
from src.models.hair_length_classifier import HairLengthClassifier
from src.models.gender_predictor import GenderPredictor
from src.inference.decision_router import DecisionRouter, PredictionResult
from src.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class InferenceError(Exception):
    """Base exception for inference pipeline errors."""
    pass


class ModelLoadError(InferenceError):
    """Raised when model weight files are absent or corrupted."""
    pass


class InvalidFileFormatError(InferenceError):
    """Raised when the uploaded file is not a supported image format."""
    pass


class FileSizeError(InferenceError):
    """Raised when the uploaded file exceeds the 10 MB size limit."""
    pass


class CorruptImageError(InferenceError):
    """Raised when the uploaded image cannot be decoded."""
    pass


class NoFaceDetectedError(InferenceError):
    """Raised when no face is found in the uploaded image."""
    pass


class MultipleFacesError(InferenceError):
    """Raised when more than one face is detected in the uploaded image."""
    pass


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Orchestrates model loading, image validation, face detection, and prediction.

    Usage:
        engine = InferenceEngine(model_dir="models")
        engine.load_models()
        result = engine.predict("path/to/image.jpg")
    """

    # Expected model filenames
    _MODEL_FILES = {
        "age_estimator": "age_estimator.keras",
        "hair_classifier": "hair_classifier.keras",
        "gender_predictor": "gender_predictor.keras",
    }

    def __init__(self, model_dir: str = "models") -> None:
        """Initialise the engine with a model directory path.

        Args:
            model_dir: Directory containing the three .keras weight files.
        """
        self.model_dir = model_dir
        self._age_estimator: Optional[AgeEstimator] = None
        self._hair_classifier: Optional[HairLengthClassifier] = None
        self._gender_predictor: Optional[GenderPredictor] = None
        self._router = DecisionRouter()
        self._preprocessor = Preprocessor()
        self._face_cascade: Optional[cv2.CascadeClassifier] = None

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_models(self, model_dir: Optional[str] = None) -> None:
        """Load all three model weight files from disk.

        Args:
            model_dir: Optional override for the model directory. If None,
                       uses the directory provided at construction time.

        Raises:
            ModelLoadError: If any weight file is absent or cannot be loaded.
        """
        if model_dir is not None:
            self.model_dir = model_dir

        # Validate that all expected files exist before attempting to load
        missing_files = []
        for name, filename in self._MODEL_FILES.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.isfile(path):
                missing_files.append(filename)

        if missing_files:
            raise ModelLoadError(
                "Model files are missing or corrupted. "
                f"Missing: {', '.join(missing_files)}. "
                "Please run `train.py` before launching the application."
            )

        # Attempt to load each model
        try:
            self._age_estimator = AgeEstimator()
            self._age_estimator.load(
                os.path.join(self.model_dir, self._MODEL_FILES["age_estimator"])
            )
        except Exception as exc:
            raise ModelLoadError(
                "Model files are missing or corrupted. "
                f"Failed to load age_estimator.keras: {exc}. "
                "Please run `train.py` before launching the application."
            ) from exc

        try:
            self._hair_classifier = HairLengthClassifier()
            self._hair_classifier.load(
                os.path.join(self.model_dir, self._MODEL_FILES["hair_classifier"])
            )
        except Exception as exc:
            raise ModelLoadError(
                "Model files are missing or corrupted. "
                f"Failed to load hair_classifier.keras: {exc}. "
                "Please run `train.py` before launching the application."
            ) from exc

        try:
            self._gender_predictor = GenderPredictor()
            self._gender_predictor.load(
                os.path.join(self.model_dir, self._MODEL_FILES["gender_predictor"])
            )
        except Exception as exc:
            raise ModelLoadError(
                "Model files are missing or corrupted. "
                f"Failed to load gender_predictor.keras: {exc}. "
                "Please run `train.py` before launching the application."
            ) from exc

        # Load Haar cascade for face detection (bundled with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise ModelLoadError(
                "Failed to load the face detection model. "
                "Ensure opencv-python is properly installed."
            )

        logger.info("All models loaded successfully from '%s'.", self.model_dir)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> PredictionResult:
        """Run the full inference pipeline on an image.

        Steps:
        1. Validate file format and size.
        2. Load and validate the image data.
        3. Detect exactly one face.
        4. Preprocess the image.
        5. Route through DecisionRouter.

        Args:
            image_path: Filesystem path to the input image.

        Returns:
            A PredictionResult containing label, confidence, age, and group.

        Raises:
            InvalidFileFormatError: File extension not supported.
            FileSizeError: File exceeds 10 MB.
            CorruptImageError: Image cannot be decoded.
            NoFaceDetectedError: No face found in image.
            MultipleFacesError: More than one face detected.
            InferenceError: Unexpected failure during prediction.
        """
        # --- Step 1: Validate file format ---
        self._validate_file_format(image_path)

        # --- Step 2: Validate file size ---
        self._validate_file_size(image_path)

        # --- Step 3: Load and validate image ---
        image_bgr = self._load_image(image_path)

        # --- Step 4: Detect face(s) ---
        self._detect_faces(image_bgr)

        # --- Step 5: Preprocess ---
        try:
            preprocessed = self._preprocessor.preprocess(image_path)
        except Exception as exc:
            logger.error("Preprocessing failed: %s", exc)
            raise InferenceError(
                "An error occurred during prediction. "
                "Please try again with a different image."
            ) from exc

        # --- Step 6: Route through decision pipeline ---
        try:
            result = self._router.route(
                image=preprocessed,
                age_estimator=self._age_estimator,
                hair_classifier=self._hair_classifier,
                gender_predictor=self._gender_predictor,
            )
        except Exception as exc:
            logger.error("Inference routing failed: %s", exc)
            raise InferenceError(
                "An error occurred during prediction. "
                "Please try again with a different image."
            ) from exc

        logger.info(
            "Prediction complete — label=%s, confidence=%.4f, age=%d, group=%s",
            result.label,
            result.confidence,
            result.estimated_age,
            result.age_group,
        )
        return result

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------

    def _validate_file_format(self, image_path: str) -> None:
        """Check that the file has a supported image extension.

        Raises:
            InvalidFileFormatError: If the extension is not supported.
        """
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidFileFormatError(
                "The file format is not supported. "
                "Please upload a JPEG, PNG, or BMP image."
            )

    def _validate_file_size(self, image_path: str) -> None:
        """Ensure the file does not exceed the 10 MB limit.

        Raises:
            FileSizeError: If the file is larger than 10 MB.
        """
        if not os.path.isfile(image_path):
            raise CorruptImageError(
                "The selected file could not be opened. "
                "Please choose a valid image."
            )

        file_size = os.path.getsize(image_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise FileSizeError(
                "File exceeds the 10 MB size limit. "
                "Please choose a smaller image."
            )

    def _load_image(self, image_path: str) -> np.ndarray:
        """Attempt to load the image with OpenCV.

        Returns:
            The loaded image in BGR colour space.

        Raises:
            CorruptImageError: If OpenCV cannot decode the file.
        """
        try:
            image = cv2.imread(image_path)
        except Exception as exc:
            raise CorruptImageError(
                "The selected file could not be opened. "
                "Please choose a valid image."
            ) from exc

        if image is None:
            raise CorruptImageError(
                "The selected file could not be opened. "
                "Please choose a valid image."
            )
        return image

    def _detect_faces(self, image_bgr: np.ndarray) -> None:
        """Run Haar cascade face detection and enforce single-face constraint.

        Raises:
            NoFaceDetectedError: If zero faces are detected.
            MultipleFacesError: If more than one face is detected.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        num_faces = len(faces)
        if num_faces == 0:
            raise NoFaceDetectedError(
                "No face was detected in the image. "
                "Please upload an image with a clearly visible face."
            )
        if num_faces > 1:
            raise MultipleFacesError(
                "Multiple faces detected. "
                "Please upload an image containing a single face."
            )
