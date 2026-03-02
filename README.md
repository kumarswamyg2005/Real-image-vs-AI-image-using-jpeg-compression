# Real Image vs AI Image Detection Using JPEG Compression Forensics

Detect whether an image is **AI-generated** or **real** using JPEG bitstream forensics — DCT coefficient analysis, quantization table patterns, blocking artifacts, and Benford's Law statistics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97.2%25-brightgreen)

---

## Overview

AI-generated images (from tools like DALL·E, Midjourney, Stable Diffusion) leave subtle compression fingerprints that differ from real camera photos. This project exploits those differences by analyzing the JPEG bitstream — the low-level compression data — to classify images with **97.2% accuracy**.

### How It Works

1. **JPEG Compression Forensics** — Extracts 70 features from DCT coefficients, quantization tables, blocking artifacts, and first-digit distributions (Benford's Law).
2. **Camera Signature Analysis** — Checks whether the image matches common camera aspect ratios (4:3, 3:2, 16:9) and megapixel counts (12MP, 48MP, 24MP, etc.).
3. **Ensemble Classification** — Combines forensic features with a trained neural network to produce the final AI probability score.

### Key Results

| Metric               | Value    |
| -------------------- | -------- |
| Overall Accuracy     | 97.2%    |
| Real Image Detection | 97.4%    |
| AI Image Detection   | 97.0%    |
| False Positive Rate  | 2.6%     |
| Training Images      | 348,000+ |

---

## Project Structure

```
├── web_app.py                  # Flask web server and API
├── ensemble_detector.py        # Main ensemble AI detector
├── bitstream_features.py       # JPEG forensic feature extraction (70 features)
├── smart_camera_detector.py    # Camera signature analysis
├── jpg.py                      # Core JPEG codec (RGB→YCbCr, DCT, quantization, Huffman)
├── config.py                   # Configuration settings
├── train_local_bitstream.py    # Local model training script
├── test_local_model.py         # Model testing and evaluation
├── bitstream_detector_local.pth # Trained model weights
├── kaggle_bitstream_training.ipynb # Kaggle training notebook
├── templates/
│   └── index.html              # Web UI for uploading and classifying images
├── images/                     # Sample test images
└── requirements.txt            # Python dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/kumarswamyg2005/Real-image-vs-AI-image-using-jpeg-compression.git
cd Real-image-vs-AI-image-using-jpeg-compression

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- NumPy, SciPy, scikit-learn

---

## Usage

### Web Application

```bash
python web_app.py
```

Open `http://localhost:5000` in your browser. Upload any image to get an instant AI vs Real classification with a confidence score.

### API Endpoint

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/predict
```

**Response:**

```json
{
  "is_ai": true,
  "label": "🤖 AI-Generated",
  "confidence": 87.2,
  "ai_score": 87.2,
  "real_score": 12.8,
  "model": "Bitstream Forensics + Camera Analysis",
  "method": "JPEG Compression Forensics"
}
```

### JPEG Compression Analysis (standalone)

```bash
python jpg.py
```

Demonstrates the full JPEG pipeline: color transform, DCT, quantization, zigzag scanning, and Huffman coding.

---

## How JPEG Forensics Detects AI Images

### 1. DCT Coefficient Analysis

Real camera photos produce characteristic DCT coefficient distributions after JPEG compression. AI-generated images, even when saved as JPEG, show different statistical patterns in their frequency-domain coefficients.

### 2. Quantization Table Fingerprinting

Real cameras embed specific quantization tables from their firmware. AI images use generic or software-defined tables, which can be detected through pattern analysis.

### 3. Blocking Artifact Detection

JPEG divides images into 8×8 blocks. The boundary artifacts between blocks differ between real photos (captured and compressed once) and AI images (generated then compressed).

### 4. Benford's Law

The first-digit distribution of DCT coefficients in natural images follows Benford's Law. AI-generated images often deviate from this statistical pattern.

---

## Training

The model was trained on **348,000+ images** from multiple datasets. To retrain locally:

```bash
python train_local_bitstream.py
```

For GPU-accelerated training on Kaggle, use the included notebook: `kaggle_bitstream_training.ipynb`.

See [BITSTREAM_TRAINING_GUIDE.md](BITSTREAM_TRAINING_GUIDE.md) for detailed training instructions.

---

## Tech Stack

- **Backend:** Flask, PyTorch, OpenCV
- **Forensics:** DCT analysis, quantization forensics, Benford's Law
- **Model:** ResNet50 backbone + custom forensic feature layer
- **Frontend:** HTML/CSS/JavaScript

---

## License

This project is for educational and research purposes.
