# 🧑 Long-Hair Gender Identification

A machine learning system that predicts gender from face images with an **intentional hair-length bias** for individuals aged 20–30. For this age group, the model uses hair length to determine gender (long hair → Female, short hair → Male). For all other ages, it uses standard facial feature analysis.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Keras](https://img.shields.io/badge/Keras-3.14-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B)

---

## 🎯 How It Works

```
Input Image → Face Detection → Age Estimation → Route Decision
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    │                                   │
                              Age 20-30                           Age <20 or >30
                                    │                                   │
                         Hair Length Classifier              Gender Predictor
                                    │                          (facial features)
                         ┌──────────┼──────────┐                    │
                    Long Hair    Short Hair   Undetermined      Male/Female
                         │          │              │
                      Female      Male        Undetermined
```

**Three MobileNetV2-based models work together:**

| Model | Purpose | Accuracy |
|-------|---------|----------|
| **AgeEstimator** | Predicts age (1-100), routes to correct path | 80.5% age-group accuracy |
| **HairLengthClassifier** | Classifies hair as long/short for ages 20-30 | 76.8% biased accuracy |
| **GenderPredictor** | Standard gender prediction for ages outside 20-30 | 87.9% accuracy |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Ilham0310/gender-identification.git
cd gender-identification

# Create virtual environment (Python 3.10-3.12 recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Models

**Option A: Train on Kaggle (Recommended, ~10 min with GPU)**

1. Upload `kaggle_train.ipynb` to [Kaggle](https://www.kaggle.com/code)
2. Add the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) 
3. Enable GPU accelerator in notebook settings
4. Run all cells
5. Download the trained models and place them in `models/`

**Option B: Train locally (CPU, ~2-3 hours)**

```bash
# Download UTKFace dataset as archive.zip first
python train.py --seed 42 --epochs-age 15 --epochs-hair 10 --epochs-gender 10
```

After training you should have:
```
models/
├── age_estimator.keras
├── hair_classifier.keras
├── gender_predictor.keras
└── config.json
```

### 3. Run the Web App

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser, upload a face photo, and get predictions!

---

## 📁 Project Structure

```
gender-identification/
├── src/
│   ├── data/
│   │   ├── dataset_loader.py      # UTKFace loader + hair pseudo-label generation
│   │   └── preprocessor.py        # Image resize, normalize, augment
│   ├── models/
│   │   ├── age_estimator.py       # MobileNetV2 age regression model
│   │   ├── hair_length_classifier.py  # MobileNetV2 hair binary classifier
│   │   └── gender_predictor.py    # MobileNetV2 gender binary classifier
│   ├── inference/
│   │   ├── decision_router.py     # Age-group routing + PredictionResult
│   │   └── inference_engine.py    # Full pipeline orchestrator
│   └── gui/
│       └── main_window.py         # PyQt5 desktop GUI (alternative to Streamlit)
├── tests/                          # Property-based + unit tests (pytest + hypothesis)
├── models/                         # Trained model weights (not in git, see training)
├── streamlit_app.py               # Web-based GUI (recommended)
├── app.py                         # PyQt5 desktop GUI entry point
├── train.py                       # Training pipeline entry point
├── kaggle_train.ipynb             # Kaggle notebook for GPU training
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🧪 Running Tests

The project includes 12 property-based tests (using Hypothesis) and unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run property-based tests only
pytest tests/ -v -k "property or test_age_group or test_confidence or test_decision"
```

**Properties tested:**
- Age-group classification is total and exclusive
- Biased prediction maps hair label to gender deterministically
- Outside-age-group predictions are hair-length-invariant
- Age estimator output always in [1, 100]
- Hair classifier output domain is bounded
- Confidence formatting is always valid
- File size gate rejects oversized files
- System is robust to unreadable inputs
- Missing model weights produce specific error
- Training config always contains required keys
- Training reproducibility with fixed seed

---

## ⚙️ Configuration

Training hyperparameters are stored in `models/config.json`:

```json
{
  "seed": 42,
  "model_architecture": "MobileNetV2",
  "dataset_split": {"train": 0.70, "val": 0.10, "test": 0.20},
  "hyperparameters": {
    "age_estimator": {"learning_rate": 0.0001, "epochs": 15, "dropout": 0.3},
    "hair_classifier": {"learning_rate": 0.0001, "epochs": 10, "undetermined_threshold": 0.55},
    "gender_predictor": {"learning_rate": 0.0001, "epochs": 10}
  }
}
```

---

## 📊 Dataset

This project uses the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) (~23K face images with age and gender labels).

- **Filename format:** `{age}_{gender}_{race}_{timestamp}.jpg.chip.jpg`
- **Gender:** 0 = Male, 1 = Female
- **Hair labels:** Derived via heuristic (no ground-truth hair annotations in UTKFace)

---

## 🔧 Requirements

- Python 3.10 – 3.12
- TensorFlow 2.x / Keras 3.x
- OpenCV (face detection)
- Streamlit (web UI)
- NumPy, Pandas, scikit-learn

See `requirements.txt` for full list.

---

## 📝 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML Framework | TensorFlow/Keras | Mature ecosystem, good Windows support |
| Base Model | MobileNetV2 (ImageNet) | Lightweight, fast on CPU, strong transfer learning |
| GUI | Streamlit | Zero-dependency web UI, works on any OS |
| Face Detection | OpenCV Haar Cascade | Bundled with OpenCV, no extra downloads |
| Hair Labels | Heuristic pseudo-labels | UTKFace has no hair annotations |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Push and create a Pull Request

---

## 📄 License

This project is for educational/research purposes.
