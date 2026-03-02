# Gamma Presentation Prompt

## Title

Detecting AI Images Through JPEG Compression Forensics
Subtitle: A Machine Learning Approach
Stats: 97.2% Accuracy | 348,385 Training Images | February 2026

---

## Slide 1: What Is This Project?

Problem: AI generators like Stable Diffusion, DALL-E, and Midjourney can now create images that look completely real to the human eye.

Challenge: How do we tell real photos from AI-generated images when they look identical?

Our Solution: Build a detector using JPEG compression forensics to analyze HOW images were created, not just WHAT they look like.

Goal: Achieve 98%+ accuracy with minimal false positives

---

## Slide 2: How We Use JPEG Compression

When a real camera takes a photo:

1. Light hits the camera sensor (physical process)
2. Camera converts to pixels
3. JPEG compresses pixels into 8×8 blocks using DCT (Discrete Cosine Transform)
4. Creates unique compression "fingerprints"

We Extract 70 Forensic Features:

- DCT coefficients - compression math patterns
- Blocking artifacts - 8×8 grid boundaries
- Benford's Law - natural statistical distribution
- Camera dimension signatures - 12MP, 4:3 aspect ratio patterns

Key Insight: AI-generated images have different compression patterns because they weren't created by a physical camera sensor!

---

## Slide 3: What About PNG Images?

JPEG (Lossy Compression):

- Loses some data to reduce file size
- Creates compression artifacts we can detect
- 97.2% accuracy with our model ✓
- Most real cameras save as JPEG

PNG (Lossless Compression):

- Perfect quality, no data loss
- No JPEG compression artifacts to analyze
- Lower accuracy (model not trained on PNG)
- Relies on dimension analysis instead

Our Model: Works on both formats, but best with JPEG (trained on 348k JPEG images)

---

## Slide 4: Our Approach - Forensic Analysis

Traditional Method (FAILED):

- Deep Learning (ResNet50): "Does this LOOK like AI?"
- Problem: 100% false positives ❌
- All real photos were wrongly flagged as AI

Our Method (SUCCESS):

- Forensics: "How was this created? Check the compression!"
- Machine Learning (Random Forest) with 70 forensic features
- Result: 97.2% accuracy, only 2.6% false positives ✓

Key Insight: AI can make images that LOOK real to our eyes, but it can't fake the mathematical patterns from real camera JPEG compression. Physics is harder to fake than appearances!

---

## Slide 5: Training Process

Training Data:

- 348,385 images total (half real, half AI-generated)
- 120,000 from CIFAKE dataset (objects, scenes, landscapes)
- 140,000 from faces dataset (human portraits)

Training Method:

- Extract 70 forensic features from each image
- Train Random Forest classifier (300 decision trees)
- Offline training on Mac (6-8 hours, CPU-only)
- Test on separate validation dataset

Technology: Python, OpenCV, NumPy, SciPy, scikit-learn

---

## Slide 6: Results

Overall Accuracy: 97.2%

Breakdown:

- 📷 Real Photo Detection: 97.4%
- 🤖 AI Image Detection: 97.0%
- ❌ False Positives: 2.6% (down from 100%!)
- 📊 Training Images: 348,385 images
- 💻 Hardware: CPU-only (no GPU needed)

The system is practical, deployable, and works completely offline!

---

## Slide 7: Web Application Demo

Built a Flask Web Interface

How It Works:

1. User uploads an image (drag and drop)
2. Model analyzes JPEG compression patterns
3. Checks camera dimension signatures
4. Shows result: Real Photo 📷 or AI-Generated 🤖
5. Displays confidence score (0-100%)

Performance:

- Runs locally at http://localhost:5001
- Processes images in under 1 second
- Fast, accurate, and practical

---

## Slide 8: Future Improvements

1. Improve PNG Detection
   - Train specifically on PNG images
   - Develop PNG-specific forensic features

2. Detect Specific AI Generators
   - Identify which AI created it (Stable Diffusion, DALL-E, Midjourney)
   - Each AI has unique fingerprints

3. Mobile App
   - iOS/Android apps for on-the-go detection
   - Check images before sharing on social media

4. Higher Accuracy
   - Target: 99%+ accuracy with <1% false positives
   - Add more forensic features (currently 70)

---

## Slide 9: Summary

What We Built:

- AI detector using JPEG compression forensics
- Achieved 97.2% accuracy, fixed false positive problem
- Works on 348k images, CPU-only, offline-capable
- Created working web demo

Key Takeaway:
Forensic analysis beats deep learning for this task. Instead of looking at pixels and guessing, we check the compression math - and that's much harder to fake!

The Principle: Trust physics and mathematics, not appearances.

---

## Design Notes for Gamma:

Theme: Professional blue and white color scheme
Style: Clean, modern, technical but accessible
Use: Charts, icons, and visual elements where appropriate
Tone: Confident, scientific, practical
Emphasis: High-accuracy results, practical implementation, forensic approach over pixel-based deep learning

Key Visual Elements:

- Large "97.2%" accuracy number prominently displayed
- Before/After comparison (100% false positives → 2.6%)
- Icons for real photos 📷 and AI images 🤖
- Comparison tables for JPEG vs PNG
- Flowcharts showing the compression process
