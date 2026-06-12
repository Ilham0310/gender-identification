"""Main window for the Long-Hair Gender Identification GUI.

Provides the primary application window with:
- Left panel: image preview (400×400 px)
- Right panel: prediction results display
- Buttons: "Upload Image" and "Run Prediction"
- Status bar showing application state
- Background inference via InferenceWorker (QThread) with 30 s timeout
"""

import os
import sys
import logging
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFormLayout,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from src.inference.decision_router import format_confidence

logger = logging.getLogger(__name__)

# Maximum allowed file size for uploaded images (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class InferenceTimeoutError(Exception):
    """Raised when inference exceeds the 30-second timeout."""
    pass


# ---------------------------------------------------------------------------
# Inference Worker (QThread)
# ---------------------------------------------------------------------------


class InferenceWorker(QThread):
    """Background thread that runs the inference engine on a given image.

    Signals:
        result_ready: Emitted with a PredictionResult on success.
        error_occurred: Emitted with a user-readable error string on failure.
    """

    result_ready = pyqtSignal(object)  # PredictionResult
    error_occurred = pyqtSignal(str)

    def __init__(self, engine, image_path: str, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.image_path = image_path

    def run(self):
        """Execute inference in the background thread."""
        try:
            result = self.engine.predict(self.image_path)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Main application window for gender prediction with hair-length bias."""

    INFERENCE_TIMEOUT_MS = 30_000  # 30 seconds

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Long-Hair Gender Identification")
        self.setMinimumSize(850, 550)

        # Store inference engine (may be None; app.py is responsible for loading models)
        self._engine = engine

        # Worker and timer references (active during inference)
        self._worker: Optional[InferenceWorker] = None
        self._timeout_timer: Optional[QTimer] = None

        # Current image path (set by upload logic)
        self._current_image_path: Optional[str] = None

        self._setup_ui()
        self._connect_signals()
        self.statusBar().showMessage("Ready")

    def _connect_signals(self):
        """Connect widget signals to slots."""
        self.upload_button.clicked.connect(self._on_upload_clicked)
        self.predict_button.clicked.connect(self._on_predict_clicked)

    # ------------------------------------------------------------------
    # Upload Logic
    # ------------------------------------------------------------------

    def _on_upload_clicked(self):
        """Handle the Upload Image button click.

        Opens a file dialog filtered to supported image formats,
        validates file size and readability, then displays a preview.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )

        if not file_path:
            return

        # Validate file size (must be ≤ 10 MB)
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            QMessageBox.warning(
                self,
                "File Too Large",
                "File exceeds the 10 MB size limit. Please choose a smaller image.",
            )
            return

        # Try to load the image
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "Invalid Image",
                "The selected file could not be opened. Please choose a valid image.",
            )
            return

        # Scale the image to fit 400×400 while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Store the path and enable prediction
        self._current_image_path = file_path
        self.predict_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Prediction / Threading Logic
    # ------------------------------------------------------------------

    def _on_predict_clicked(self):
        """Handle the Run Prediction button click.

        Launches an InferenceWorker in a background thread, shows a
        'Processing...' indicator in the status bar immediately, and
        starts a 30-second timeout timer.
        """
        if self._engine is None:
            QMessageBox.warning(
                self,
                "No Engine",
                "Inference engine is not available. "
                "Please run train.py before launching the application.",
            )
            return

        if self._current_image_path is None:
            return

        # Disable the predict button to prevent double-clicks
        self.predict_button.setEnabled(False)

        # Clear previous results before showing the loading indicator
        self._clear_results()

        # Show "Processing..." immediately (within 200 ms) via QTimer.singleShot(0, ...)
        QTimer.singleShot(0, self._show_processing_indicator)

        # Create and configure the worker
        self._worker = InferenceWorker(
            engine=self._engine,
            image_path=self._current_image_path,
            parent=self,
        )
        self._worker.result_ready.connect(self._on_inference_complete)
        self._worker.error_occurred.connect(self._on_inference_error)
        self._worker.finished.connect(self._on_worker_finished)

        # Start the timeout timer (30 seconds)
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.setInterval(self.INFERENCE_TIMEOUT_MS)
        self._timeout_timer.timeout.connect(self._on_inference_timeout)
        self._timeout_timer.start()

        # Start the worker thread
        self._worker.start()
        logger.info("Inference worker started for: %s", self._current_image_path)

    def _show_processing_indicator(self):
        """Display 'Processing...' text in the status bar."""
        self.statusBar().showMessage("Processing\u2026")

    def _clear_results(self):
        """Reset all result labels to their default placeholder state."""
        self.prediction_label.setText("\u2014")
        self.confidence_label.setText("\u2014")
        self.age_group_label.setText("\u2014")
        self.estimated_age_label.setText("\u2014")
        self.placeholder_label.setVisible(True)

    def _on_inference_complete(self, result):
        """Handle successful inference result.

        Args:
            result: A PredictionResult object from the inference engine.
        """
        self._stop_timeout_timer()
        logger.info("Inference completed successfully.")

        # Hide the placeholder and show populated results
        self.placeholder_label.setVisible(False)
        self.prediction_label.setText(result.label)
        self.confidence_label.setText(format_confidence(result.confidence))
        self.age_group_label.setText(result.age_group_display)
        self.estimated_age_label.setText(str(result.estimated_age))

        # Restore GUI to ready state
        self.statusBar().showMessage("Ready")
        self.predict_button.setEnabled(True)

    def _on_inference_error(self, error_message: str):
        """Handle inference error.

        Args:
            error_message: User-readable error description.
        """
        self._stop_timeout_timer()
        logger.error("Inference error: %s", error_message)

        # Reset results to default state
        self._clear_results()
        self.placeholder_label.setVisible(True)

        # Restore GUI to upload-ready state
        self.statusBar().showMessage("Ready")
        self.predict_button.setEnabled(True)

        QMessageBox.warning(self, "Prediction Error", error_message)

    def _on_inference_timeout(self):
        """Handle inference timeout (30 seconds exceeded)."""
        logger.warning("Inference timed out after %d ms.", self.INFERENCE_TIMEOUT_MS)

        # Terminate the worker thread
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)  # Wait up to 2 s for thread cleanup

        # Reset UI state
        self.statusBar().showMessage("Ready")
        self.predict_button.setEnabled(True)

        # Emit timeout error to the user
        QMessageBox.warning(
            self,
            "Timeout",
            "Prediction timed out after 30 seconds. Please try again.",
        )

    def _on_worker_finished(self):
        """Clean up after the worker thread finishes (success or failure)."""
        self._stop_timeout_timer()
        # Re-enable predict button (if not already done by error/timeout handler)
        if not self.predict_button.isEnabled():
            self.predict_button.setEnabled(True)
        if self.statusBar().currentMessage() == "Processing\u2026":
            self.statusBar().showMessage("Ready")

    def _stop_timeout_timer(self):
        """Stop and clean up the timeout timer if it's active."""
        if self._timeout_timer is not None:
            self._timeout_timer.stop()
            self._timeout_timer = None

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the main window layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # --- Left panel: image preview + buttons ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # Image preview label
        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel {"
            "  border: 2px dashed #aaa;"
            "  background-color: #f5f5f5;"
            "}"
        )
        self.image_label.setText("No image loaded")
        left_panel.addWidget(self.image_label)

        # Upload Image button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setMinimumHeight(36)
        left_panel.addWidget(self.upload_button)

        # Run Prediction button
        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.setMinimumHeight(36)
        self.predict_button.setEnabled(False)
        left_panel.addWidget(self.predict_button)

        left_panel.addStretch()
        main_layout.addLayout(left_panel)

        # --- Right panel: results ---
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(12)
        results_layout.setContentsMargins(16, 20, 16, 16)

        # Result fields using a form layout
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(10)
        form_layout.setHorizontalSpacing(12)

        label_font = QFont()
        label_font.setBold(True)

        # Prediction label
        prediction_header = QLabel("Prediction:")
        prediction_header.setFont(label_font)
        self.prediction_label = QLabel("\u2014")
        form_layout.addRow(prediction_header, self.prediction_label)

        # Confidence label
        confidence_header = QLabel("Confidence:")
        confidence_header.setFont(label_font)
        self.confidence_label = QLabel("\u2014")
        form_layout.addRow(confidence_header, self.confidence_label)

        # Age Group label
        age_group_header = QLabel("Age Group:")
        age_group_header.setFont(label_font)
        self.age_group_label = QLabel("\u2014")
        form_layout.addRow(age_group_header, self.age_group_label)

        # Estimated Age label
        est_age_header = QLabel("Estimated Age:")
        est_age_header.setFont(label_font)
        self.estimated_age_label = QLabel("\u2014")
        form_layout.addRow(est_age_header, self.estimated_age_label)

        results_layout.addLayout(form_layout)

        # Placeholder message
        self.placeholder_label = QLabel(
            "No prediction yet. Please upload an image."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet("color: #888; font-style: italic;")
        results_layout.addWidget(self.placeholder_label)

        results_layout.addStretch()
        main_layout.addWidget(results_group, stretch=1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
