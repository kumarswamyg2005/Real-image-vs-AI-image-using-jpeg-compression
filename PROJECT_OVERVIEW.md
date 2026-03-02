# AI vs Real Image Detector — Project Overview

This project serves a small Flask web app that classifies an uploaded image as **AI-generated** or **real**. The core logic combines a deep-learning model with a **camera signature analysis** (aspect ratio + megapixels) to refine the final decision.

---

## ✅ What happens end-to-end

1. The browser uploads an image to the Flask app.
2. The app saves the file in `uploads/` and calls the detector.
3. The detector:
   - Loads a model (ResNet50 by default).
   - Preprocesses the image.
   - Predicts AI probability.
   - Adjusts that probability using camera signature cues.
4. The API returns a JSON response with labels and confidence.

---

## 📁 Key files (and what they do)

- `web_app.py` — Flask server and API routes.
- `smart_camera_detector.py` — model loading + inference + camera signature logic.
- `ai_detector_resnet50.pth` — default model weights used at runtime.
- `models/` — additional model files (not used by default).
- `templates/index.html` — browser UI.

---

## 🧠 Which model is used (and where)

### **Default model (ResNet50)**

The detector uses **ResNet50** unless the checkpoint contains CLIP-specific keys.

- **Default path:** `ai_detector_resnet50.pth`
- **Loaded in:** `SmartCameraDetector.__init__()`

```python
# smart_camera_detector.py
class SmartCameraDetector:
    def __init__(self, model_path='ai_detector_resnet50.pth'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                   'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = self._detect_model_type(model_path)
        self.model = self._load_model(model_path)
```

### **Optional CLIP-based model**

If the checkpoint includes keys containing `clip_model`, the detector switches to a CLIP-based classifier:

```python
# smart_camera_detector.py
if any('clip_model' in key for key in state_dict.keys()):
    return 'CLIP'
else:
    return 'ResNet50'
```

---

## 🧩 How prediction actually works

### 1) Camera signature analysis

The detector checks if the image **looks like a real camera photo** based on:

- common aspect ratios (4:3, 3:2, 16:9, 1:1)
- common megapixel ranges (12MP, 48MP, 24MP, etc.)

```python
# smart_camera_detector.py
is_camera, camera_conf, reasons = self._analyze_camera_signature(width, height)
```

### 2) Deep learning prediction

The image is resized and normalized, then passed through the model:

```python
# smart_camera_detector.py
image_tensor = self.transform(image).unsqueeze(0).to(self.device)
with torch.no_grad():
    model_ai_prob = self.model(image_tensor).item()
```

### 3) Probability adjustment

If the photo strongly looks like a camera image, the AI probability is reduced.

```python
# smart_camera_detector.py
if is_camera and camera_conf > 50:
    reduction = (camera_conf / 100) * 0.7
    final_ai_prob = model_ai_prob * (1 - reduction)
```

---

## 🌐 Web API — where the model is called

The Flask app creates **one detector instance** and reuses it for each request:

```python
# web_app.py
print("Initializing Smart Camera Detector...")
detector = SmartCameraDetector()
```

The `/predict` endpoint saves the image and calls `detector.predict()`:

```python
# web_app.py
result = detector.predict(filepath, return_details=True)
```

It returns a structured JSON response:

```json
{
  "is_ai": true,
  "label": "🤖 AI-Generated",
  "confidence": 87.2,
  "ai_score": 87.2,
  "real_score": 12.8,
  "model": "ResNet50 + Camera Analysis",
  "method": "Deep Learning + Camera Signature",
  "image_url": "/uploads/your_file.jpg"
}
```

---

## 🔍 Summary in one line

The system uses **ResNet50 (or CLIP)** for AI detection and **camera metadata-like heuristics** to refine the final decision before returning results through a Flask web API.
