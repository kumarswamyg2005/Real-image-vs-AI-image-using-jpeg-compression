# Real Image vs AI Image Detection Using JPEG Compression Forensics

Detect whether an image is **AI-generated** or **real** using JPEG bitstream forensics, sensor pattern noise analysis, and AI model attribution.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![Accuracy](https://img.shields.io/badge/Accuracy-86.5%25-brightgreen)
![Formats](https://img.shields.io/badge/Formats-JPEG%20%7C%20PNG%20%7C%20WebP%20%7C%20HEIC%20%7C%20TIFF%20%7C%20BMP-orange)

---

## Overview

AI-generated images (from tools like DALL·E 3, Midjourney, Stable Diffusion, Adobe Firefly) leave subtle forensic fingerprints that differ from real camera photos. This project exploits those differences by analyzing:

1. **JPEG Bitstream Forensics** — 70 features from DCT coefficients, quantization tables, blocking artifacts, and Benford's Law statistics
2. **SPN Sensor Pattern Noise** — Detects camera PRNU (Photo Response Non-Uniformity); AI images have no physical sensor noise
3. **AI Model Attribution** — Identifies *which* AI model generated the image (DALL-E 3, Midjourney, Stable Diffusion, Firefly)
4. **Camera Signature Analysis** — Checks aspect ratios and megapixel counts typical of real cameras
5. **Region Heatmap** — Block-level overlay showing *where* in an image AI patterns are detected

### Key Results

| Metric               | Value    |
| -------------------- | -------- |
| Overall Accuracy     | 86.5%    |
| Real Image Detection | 84.3%    |
| AI Image Detection   | 88.5%    |
| False Positive Rate  | 15.7%    |
| Training Images      | 541,000+ |

---

## Project Structure

```
├── web_app.py              # Flask web server with heatmap and predict endpoints
├── api.py                  # Swagger REST API (/api/docs) with JWT auth
├── auth.py                 # JWT authentication + role-based access control
├── models.py               # SQLAlchemy DB models (User, AnalysisHistory)
├── ensemble_detector.py    # Ensemble AI detector (bitstream + SPN + attribution)
├── bitstream_features.py   # JPEG forensic feature extraction (70 features)
├── spn_fingerprint.py      # SPN/PRNU sensor pattern noise analysis
├── ai_model_attribution.py # Identifies DALL-E / Midjourney / SD / Firefly
├── ai_region_heatmap.py    # Block-level heatmap overlay
├── detect.py               # CLI detector (single image)
├── templates/
│   └── index.html          # Web UI
├── requirements.txt        # Python dependencies
└── bitstream_detector_local.pth  # Trained model weights
```

---

## Installation

```bash
git clone https://github.com/kumarswamyg2005/Real-image-vs-AI-image-using-jpeg-compression.git
cd Real-image-vs-AI-image-using-jpeg-compression

pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch, TensorFlow
- OpenCV, Pillow, pillow-heif
- Flask, flask-restx, flask-jwt-extended, flask-sqlalchemy
- NumPy, SciPy, scikit-learn, XGBoost
- bcrypt

---

## Usage

### Web Application

```bash
python web_app.py
```

Opens at `http://localhost:5001` (auto-selects next available port if busy). Upload any image to get an instant classification with confidence score, model breakdown, and AI model attribution.

### Command-Line Interface

```bash
python detect.py path/to/image.jpg
```

Supports JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP.

### REST API

Full Swagger documentation available at `http://localhost:5001/api/docs`.

**Predict endpoint:**

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5001/predict
```

**Response:**

```json
{
  "is_ai": true,
  "label": "AI-Generated",
  "confidence": 87.2,
  "ai_score": 87.2,
  "real_score": 12.8,
  "model": "Bitstream Forensics + SPN + Attribution",
  "method": "JPEG Compression Forensics",
  "image_format": "jpeg",
  "breakdown": {
    "bitstream": { "ai_prob": 90.1, "weight": 95.0 },
    "spn": { "ai_prob": 72.0, "is_camera_noise": false },
    "camera": { "is_camera": false, "confidence": 0.1 }
  },
  "attribution": {
    "top_model": "Stable Diffusion",
    "top_confidence": 0.83,
    "is_ai_generated": true,
    "scores": { "stable_diffusion": 0.83, "dall_e": 0.12, "midjourney": 0.05 }
  }
}
```

**Heatmap endpoint:**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filepath": "/uploads/<filename>"}' \
  http://localhost:5001/heatmap
```

Returns a base64-encoded RGBA PNG overlay highlighting AI-pattern regions (blue/green = real, yellow/red = AI).

---

## Authentication & Roles

The `/api` endpoints are protected by JWT. Default credentials:

| Role    | Username | Password   | Access                                  |
| ------- | -------- | ---------- | --------------------------------------- |
| admin   | admin    | admin123   | Full access + user management           |
| analyst | —        | —          | Image analysis + heatmap + history      |
| user    | —        | —          | Basic image analysis only               |

> **Change the default admin password in production.**

---

## How JPEG Forensics Detects AI Images

### DCT Coefficient Analysis
Real camera photos produce characteristic DCT coefficient distributions. AI images show different statistical patterns in their frequency-domain coefficients even after JPEG compression.

### Quantization Table Fingerprinting
Real cameras embed firmware-specific quantization tables. AI images use generic or software-defined tables detectable through pattern analysis.

### Blocking Artifact Detection
JPEG 8×8 block boundary artifacts differ between real photos (captured and compressed once) and AI images (generated then compressed).

### Benford's Law
The first-digit distribution of DCT coefficients in natural images follows Benford's Law. AI images often deviate from this pattern.

### SPN Sensor Pattern Noise
Every real camera sensor has unique PRNU imperfections present in every photo. AI images completely lack this physical noise signature, making it a strong discriminator.

---

## AI Model Attribution

The attribution module identifies which generative model produced an image by analyzing:

- **DALL-E 3** — Smooth low-frequency bias and JPEG post-processing signature
- **Midjourney** — Upsampling artifacts at 2× grid boundaries and high-frequency sharpening
- **Stable Diffusion** — Latent-space grid artifacts at 64-pixel periodicity, VAE spectral peaks
- **Adobe Firefly** — Luminance-channel smoothing and artificial color saturation pattern

---

## Tech Stack

- **Backend:** Flask, PyTorch, TensorFlow, OpenCV
- **Forensics:** DCT analysis, quantization fingerprinting, Benford's Law, SPN/PRNU
- **Model:** ResNet50 backbone + custom forensic feature layer + LightGBM ensemble
- **API:** flask-restx (Swagger UI), JWT authentication, SQLite + SQLAlchemy
- **Frontend:** HTML/CSS/JavaScript

---

## License

This project is for educational and portfolio use.
