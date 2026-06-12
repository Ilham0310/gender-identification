#!/usr/bin/env python3
"""
Long-Hair Gender Identification GUI Application

Main entry point for the PyQt5-based desktop application.
Loads pre-trained models and provides a user interface for gender prediction
with intentional hair-length-based bias for individuals aged 20-30.
"""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler

from PyQt5.QtWidgets import QApplication, QMessageBox

from src.inference.inference_engine import InferenceEngine, ModelLoadError
from src.gui.main_window import MainWindow


def setup_logging() -> None:
    """Configure rotating file log handler.

    Logs to logs/app.log with:
    - Max file size: 5 MB
    - Backup count: 3
    Creates the logs/ directory if it doesn't exist.
    """
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "app.log")

    handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)


def main() -> int:
    """Main entry point for the GUI application."""
    # Step 1: Set up logging
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Long-Hair Gender Identification application.")

    # Step 2: Create QApplication
    app = QApplication(sys.argv)

    # Step 3: Instantiate InferenceEngine
    engine = InferenceEngine(model_dir="models")

    # Step 4: Load models; on failure show error dialog and exit
    try:
        engine.load_models()
    except ModelLoadError as exc:
        logger.error("Model loading failed: %s", exc)
        QMessageBox.critical(
            None,
            "Model Load Error",
            "Model files are missing or corrupted. "
            "Please run `train.py` before launching the application.",
        )
        sys.exit(1)

    # Step 5: Create and show MainWindow
    window = MainWindow(engine=engine)
    window.show()

    logger.info("Application window shown. Entering event loop.")

    # Step 6: Enter the Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
