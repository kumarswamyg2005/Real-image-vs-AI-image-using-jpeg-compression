# BitstreamGuard: AI-Generated Image Detection with JPEG Forensics

BitstreamGuard detects whether an image is AI-generated or camera-captured by analyzing JPEG compression traces instead of visual content. It uses quantized DCT statistics, quantization artifacts, Benford-law deviation, double-compression periodicity, spectral energy, color, wavelet, texture, edge, and chroma-subsampling signals, then classifies the image with a five-model LightGBM soft-voting ensemble.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-LightGBM%20Ensemble-green)
![Accuracy](https://img.shields.io/badge/Accuracy-94.17%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC--ROC-0.988-orange)

---

## Overview

Pixel-domain AI detectors often learn generator-specific image artifacts that can break after JPEG recompression, resizing, or generator updates. BitstreamGuard instead examines the compression history of an image: camera photos and AI-generated images leave different statistical traces in the JPEG bitstream.

The detector extracts a 104-dimensional forensic descriptor from twelve feature stages. During training, 91 pairwise interaction features are added, producing a 195-dimensional model input. The final decision is made by a five-seed LightGBM ensemble using soft voting.

### Key Results

| Metric | Value |
| --- | --- |
| Overall Accuracy | 94.17% |
| Balanced Accuracy | 94.17% |
| Real Recall / TNR | 95.9% |
| AI Recall / TPR | 92.5% |
| False Positive Rate | 4.1% |
| False Negative Rate | 7.5% |
| AUC-ROC | 0.988 |
| MCC | 0.884 |
| Test Images | 187,827 |
| CPU Inference | < 50 ms per JPEG image |

Evaluation uses a clean three-way split: 657,391 training images, 93,914 validation images, and 187,827 held-out test images. The balanced dataset contains 939,132 images, split evenly between real and AI-generated samples.

---

## What It Analyzes

| Stage | Feature Group | Dimensions | Signal |
| --- | ---: | ---: | --- |
| A | DCT coefficient statistics | 10 | AC coefficient distribution shape |
| B | Quantization gradients | 4 | 8x8 boundary discontinuities |
| C | Blocking artifact strength | 3 | JPEG block-edge energy |
| D | Benford's Law analysis | 10 | First-digit DCT coefficient deviation |
| E | Double-compression FFT | 4 | Periodicity in DCT histograms |
| F | Frequency-band energy | 9 | Low/mid/high spectral ratios |
| G | Normalized DCT histogram | 30 | Full AC coefficient distribution |
| H | RGB/HSV color statistics | 9 | Channel spread and saturation bias |
| I | Haar wavelet residuals | 9 | Subband noise structure |
| J | LBP texture histogram | 10 | Local surface regularity |
| K | Edge and gradient features | 5 | Sharpness and edge density |
| L | Chroma subsampling flag | 1 | JPEG header provenance |
|  | Base descriptor | 104 |  |
|  | Training interactions | +91 | DCT/quantization feature pairs |
|  | Model input | 195 |  |

The strongest feature groups in ablation are the normalized DCT histogram, Benford-law features, DCT statistics, and double-compression features. The single highest-gain feature is the double-compression maximum FFT peak, which captures re-quantization artifacts in the DCT histogram.

---

## Model Architecture

1. Extract the 104-feature JPEG forensic descriptor.
2. Add 91 interaction features from the most discriminative DCT and quantization features.
3. Standardize features with the saved `StandardScaler`.
4. Run five LightGBM models trained with independent seeds: `42`, `123`, `777`, `2024`, and `9999`.
5. Average the five AI probabilities.
6. Classify as AI-generated when the final probability is `>= 0.50`.

The SPN/noise-residual module and camera-signature checks are available as investigative context, but they are not fused into the final score. The operational decision is the LightGBM bitstream ensemble only.

---

## Dataset

The balanced training set uses 469,566 real images and 469,566 AI images.

| Split | Real | AI | Total |
| --- | ---: | ---: | ---: |
| Train | 328,695 | 328,696 | 657,391 |
| Validation | 46,957 | 46,957 | 93,914 |
| Test | 93,913 | 93,914 | 187,827 |

The AI side spans 30+ generator families, including GAN, VAE, and diffusion model families such as StyleGAN, BigGAN, DDPM, Stable Diffusion, GLIDE, and related generators. Training includes faces, landscapes, objects, artwork, and other heterogeneous image categories.

To improve robustness to social-media style recompression, 20% of real-image feature extraction applies JPEG re-save augmentation at quality 70-85.

---

## Project Structure

```text
├── web_app.py              # Flask web interface and upload/predict routes
├── api.py                  # REST API, Swagger docs, JWT auth, user/history routes
├── auth.py                 # Authentication helpers and role-based access
├── models.py               # SQLAlchemy models for users and analysis history
├── ensemble_detector.py    # LightGBM bitstream ensemble + optional context modules
├── bitstream_features.py   # 104-feature JPEG forensic extractor
├── spn_fingerprint.py      # Noise residual / PRNU-style investigative heuristic
├── ai_model_attribution.py # Qualitative generator-family attribution helper
├── ai_region_heatmap.py    # Experimental region visualization helper
├── detect.py               # Command-line detector
├── train_improved.py       # Model training pipeline
├── kaggle_retrain.py       # Kaggle-oriented retraining workflow
├── figures/                # Evaluation and architecture figures
├── templates/
│   └── index.html          # Web UI
└── requirements.txt        # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/kumarswamyg2005/Real-image-vs-AI-image-using-jpeg-compression.git
cd Real-image-vs-AI-image-using-jpeg-compression

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install lightgbm
```

The detector expects a trained `.pth` checkpoint that contains the five LightGBM models, the scaler, interaction-feature metadata, and model metadata.

---

## Usage

### Web Application

```bash
python web_app.py
```

The app starts on `http://localhost:5001` by default and selects the next free port if needed.

### Command Line

```bash
python detect.py path/to/image.jpg
```

### REST API

Swagger documentation is available at:

```text
http://localhost:5001/api/docs
```

Predict endpoint:

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5001/predict
```

Example response:

```json
{
  "is_ai": true,
  "label": "AI-Generated",
  "confidence": 94.2,
  "ai_probability": 0.942,
  "real_probability": 0.058,
  "model": "Bitstream Forensics (100%) - SPN available as investigative heuristic only",
  "method": "Trained bitstream ensemble (alpha_SPN=0.00)",
  "model_breakdown": {
    "bitstream": {
      "ai_probability": 0.942,
      "ensemble_seeds": [42, 123, 777, 2024, 9999],
      "weight": 1.0
    },
    "spn": {
      "weight": 0.0
    },
    "camera_signature": {
      "weight": 0,
      "note": "Informational only - not used in score"
    }
  }
}
```

---

## Input Scope

BitstreamGuard is designed and evaluated for JPEG-compression forensics. The core features are defined around JPEG DCT coefficients, quantization behavior, block artifacts, and JPEG header metadata.

Some application code can load formats such as PNG, WebP, HEIC, TIFF, and BMP by converting them before analysis, but the reported 94.17% result is for the JPEG forensic setting. Treat non-JPEG predictions as convenience outputs, not validated performance claims.

---

## Authentication And Roles

The `/api` routes support JWT authentication and role-based access.

| Role | Access |
| --- | --- |
| admin | User management, stats, and all analysis routes |
| analyst | Image analysis, heatmap helper, and history |
| user | Basic image analysis |

Default local admin credentials are configured for development. Change them before any deployment.

---

## Limitations

- The model produces an image-level AI probability; region-level localization is experimental and not part of the validated classifier.
- JPEG quality-100 AI exports can be harder because flat quantization tables resemble high-quality camera output.
- Cross-scene evaluation on completely separate real-image sources has not been fully validated.
- A determined adversary could try to manipulate DCT statistics to imitate natural JPEG traces.
- The SPN/noise-residual module is a heuristic aid, not true camera fingerprint matching with a reference camera database.
- Outputs should be used as screening signals that require human review, not as the sole basis for high-stakes decisions.

---

## Tech Stack

- **Language:** Python 3.10
- **Model:** Five-seed LightGBM GBDT ensemble
- **Features:** JPEG DCT statistics, Benford's Law, double-compression FFT, wavelets, LBP, color, edge, chroma metadata
- **ML tooling:** scikit-learn, LightGBM, PyWavelets, PyTorch checkpoint serialization
- **Backend:** Flask, flask-restx, JWT auth, SQLAlchemy
- **Frontend:** HTML/CSS/JavaScript

---

## License

This project is for educational and portfolio use.
