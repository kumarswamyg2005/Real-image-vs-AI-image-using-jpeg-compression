"""
Ensemble AI Detector - Bitstream Forensics + AI Attribution
============================================================
High-accuracy detection using:
  • JPEG/PNG/WebP/HEIC compression forensics (DCT, quantization, Benford's Law)
  • AI Model Attribution (DALL-E, Midjourney, Stable Diffusion, Firefly)
  • Camera Signature Analysis

Model Accuracy: 94.17% (trained on 939k images, three-way split, 4.1% FPR)
Supported formats: JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP

Note: SPN Sensor Pattern Noise analysis is available as an optional
investigative heuristic (see SPNFingerprint) but is NOT used in the
classification pipeline (SPN fusion weight alpha=0.00).
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from bitstream_features import BitstreamFeatureExtractor, get_image_format, SUPPORTED_EXTENSIONS
from spn_fingerprint import SPNFingerprint
from ai_model_attribution import AIModelAttributor


class LGBMWrapper:
    """Wraps a LightGBM Booster to expose sklearn-compatible predict_proba/predict."""
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model
        self.verbose = 0

    def predict_proba(self, X):
        probs = self.lgb_model.predict(X)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.lgb_model.predict(X) >= 0.5).astype(int)


class EnsembleAIDetector:
    """
    Bitstream forensics-based AI detector using:
    - 104 image forensics features (DCT, quantization, Benford's Law, color, wavelet, LBP, edge, chroma)
    - SPN Sensor Pattern Noise analysis (camera fingerprinting)
    - AI Model Attribution (DALL-E, Midjourney, Stable Diffusion, Firefly)
    - Camera signature analysis

    Model Accuracy: 94.17% (three-way 70/10/20 split, 187,827-image test set)
    - Real detection (TNR): 95.9%
    - AI detection (TPR):   92.5%
    - False positive rate:   4.1%

    Supported formats: JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP

    Prediction weights:
    - Bitstream model (trained): 100% — sole decision maker
    - SPN sensor noise:            0% — optional investigative heuristic only
    - Camera signature:            0% — informational only in breakdown
    """

    # Camera common aspect ratios
    CAMERA_ASPECT_RATIOS = [
        (4, 3),    # 4:3 - Most common (iPhone, many cameras)
        (3, 2),    # 3:2 - DSLR standard
        (16, 9),   # 16:9 - Video/modern phones
        (1, 1),    # 1:1 - Square format cameras
    ]
    
    # Common camera megapixel counts
    CAMERA_MEGAPIXELS = [
        (12, 0.5),   # 12MP ± 0.5 (iPhone 15, 14, 13, 12)
        (48, 2),     # 48MP ± 2 (High-end phones)
        (24, 1),     # 24MP ± 1 (Full-frame cameras)
        (20, 1),     # 20MP ± 1 (APS-C cameras)
        (16, 1),     # 16MP ± 1 (Older phones)
        (8, 0.5),    # 8MP ± 0.5 (Older phones)
    ]
    
    def __init__(self,
                 model_path='bitstream_detector_v2.pth',
                 use_clip=False):
        """
        Initialize the ensemble detector.

        Args:
            model_path: Path to bitstream detector (.pth) model file.
            use_clip: Ignored (kept for backward compatibility).
        """
        print(f"✓ Using CPU for bitstream analysis")

        import warnings
        warnings.filterwarnings('ignore')

        # Load bitstream model
        # Patch: older .pth files have LGBMWrapper pickled as __main__.LGBMWrapper.
        # Redirect that lookup to this module so loading works from any entry point.
        import sys
        sys.modules['__main__'].LGBMWrapper = LGBMWrapper  # noqa: F821
        print("Loading Bitstream Forensics Model (94.17% accuracy, 4.1% FPR)...")
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        self.scaler = model_data['scaler']

        # Load ensemble of models (list) or fall back to single model for older .pth files
        if 'models' in model_data:
            self.models = model_data['models']
            for m in self.models:
                m.verbose = 0
            n = len(self.models)
            seeds = model_data.get('ensemble_seeds', list(range(n)))
            self.ensemble_seeds = seeds
            print(f"✓ Loaded ensemble of {n} LightGBM models (seeds: {seeds})")
        else:
            self.models = [model_data['model']]
            self.models[0].verbose = 0
            self.ensemble_seeds = [42]
            print("✓ Loaded single LightGBM model")
        self.model = self.models[0]  # kept for backward compatibility

        # Set up polynomial interaction features if the model was trained with them.
        # Training adds 91 interaction features (degree-2 pairs) from the first 14 features,
        # expanding 104 base features → 195 total. We must replicate this at predict time.
        self.use_interaction_features = model_data.get('interaction_features', False)
        if self.use_interaction_features:
            from sklearn.preprocessing import PolynomialFeatures
            self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self._poly.fit(np.zeros((1, 14)))  # fit on dummy to initialise output shape
            print(f"✓ Polynomial interaction features enabled ({self.scaler.n_features_in_} total features)")

        # Initialize feature extractor (now supports JPEG, PNG, WebP, HEIC, TIFF, BMP)
        self.extractor = BitstreamFeatureExtractor()
        print("✓ Bitstream model loaded")

        # SPN Sensor Pattern Noise analyser
        self.spn = SPNFingerprint()
        print("✓ SPN Sensor Pattern Noise analyser ready")

        # AI Model Attributor
        self.attributor = AIModelAttributor()
        print("✓ AI Model Attributor ready (DALL-E / Midjourney / SD / Firefly)")

        print("\n" + "="*80)
        print("🎯 BITSTREAM DETECTOR READY")
        print("="*80)
        n_models = len(self.models)
        print(f"Model: LightGBM ensemble ({n_models}× soft vote, 939k images, 104 features)")
        print(f"Accuracy: 94.17% overall (three-way split, n=187,827)")
        print(f"Real detection (TNR): 95.9% | AI detection (TPR): 92.5%")
        print(f"False positive rate: 4.1%")
        print(f"Formats supported: JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP")
        print("="*80 + "\n")
    
    def _gcd(self, a, b):
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def _analyze_camera_signature(self, width, height):
        """
        Analyze if image has camera signature
        Returns: (is_likely_camera, confidence, reasoning)
        """
        megapixels = (width * height) / 1_000_000
        
        # Calculate aspect ratio
        gcd = self._gcd(width, height)
        aspect_w = width // gcd
        aspect_h = height // gcd
        
        # Simplify aspect ratio if numbers are large
        if aspect_w > 20 or aspect_h > 20:
            ratio = width / height
            for cam_w, cam_h in self.CAMERA_ASPECT_RATIOS:
                cam_ratio = cam_w / cam_h
                if abs(ratio - cam_ratio) < 0.05:
                    aspect_w, aspect_h = cam_w, cam_h
                    break
        
        reasons = []
        confidence = 0
        
        # Check aspect ratio match
        aspect_match = False
        for cam_w, cam_h in self.CAMERA_ASPECT_RATIOS:
            if (aspect_w == cam_w and aspect_h == cam_h) or (aspect_w == cam_h and aspect_h == cam_w):
                aspect_match = True
                reasons.append(f"{aspect_w}:{aspect_h} aspect ratio (camera standard)")
                confidence += 30
                break
        
        if not aspect_match:
            reasons.append(f"{aspect_w}:{aspect_h} aspect ratio (unusual for cameras)")
        
        # Check megapixel count
        mp_match = False
        for cam_mp, tolerance in self.CAMERA_MEGAPIXELS:
            if abs(megapixels - cam_mp) <= tolerance:
                mp_match = True
                reasons.append(f"{megapixels:.1f}MP matches common camera ({cam_mp}MP)")
                confidence += 35
                break
        
        if not mp_match and megapixels > 8:
            reasons.append(f"{megapixels:.1f}MP (high resolution)")
            confidence += 15
        elif not mp_match:
            reasons.append(f"{megapixels:.1f}MP (uncommon for cameras)")
        
        # Check if resolution is very high (professional gear)
        if width >= 6000 or height >= 6000:
            reasons.append(f"Very high resolution ({width}×{height}) - professional camera")
            confidence += 20
        elif width >= 4000 or height >= 3000:
            reasons.append(f"High resolution ({width}×{height}) - typical smartphone/DSLR")
            confidence += 15
        
        # AI generators typically produce square sizes
        ai_typical = False
        if width == height:
            ai_size_match = width in [512, 768, 1024, 2048]
            if ai_size_match:
                reasons.append(f"Square {width}×{width} (typical AI generator size)")
                confidence -= 40
                ai_typical = True
        
        is_likely_camera = confidence > 40
        
        return is_likely_camera, max(0, min(100, confidence)), reasons
    
    def predict(self, image_path, return_details=False):
        """
        Predict using bitstream forensics + SPN noise analysis + camera signature.

        Args:
            image_path: Path to image (JPEG, PNG, WebP, HEIC, TIFF, BMP)
            return_details: Return detailed breakdown dict

        Returns:
            float ai_probability  OR  detailed results dict (when return_details=True)
        """
        import os
        import warnings as _warnings
        try:
            from bitstream_features import load_image_universal
            bgr, pil_img = load_image_universal(image_path)
            pil_rgb = pil_img.convert('RGB')
        except Exception:
            pil_rgb = Image.open(image_path).convert('RGB')
            bgr = None

        width, height = pil_rgb.size
        megapixels = (width * height) / 1_000_000

        # Detect image format for reporting
        img_format = __import__('bitstream_features').get_image_format(image_path)

        # ============================================
        # MODEL 1: Bitstream Forensics
        # ============================================
        try:
            features = self.extractor.extract_features(image_path)
            if features is None:
                bitstream_ai_prob = 0.5
            else:
                feat_arr = np.array(features, dtype=np.float32).reshape(1, -1)

                # Apply the same polynomial interaction features used during training:
                # degree-2 pairs of the first 14 features → 91 extra columns appended.
                if self.use_interaction_features:
                    key = feat_arr[:, :14]
                    X_poly = self._poly.transform(key)[:, 14:]  # drop originals, keep pairs
                    feat_arr = np.hstack([feat_arr, X_poly])

                expected = self.scaler.n_features_in_
                if feat_arr.shape[1] != expected:
                    raise ValueError(f"Feature count mismatch: got {feat_arr.shape[1]}, expected {expected}")
                features_scaled = self.scaler.transform(feat_arr)
                # Soft voting: average AI probability across all ensemble models
                all_probs = [float(m.predict_proba(features_scaled)[0][1])
                             for m in self.models]
                bitstream_ai_prob = float(np.mean(all_probs))
                self._last_individual_probs = all_probs
        except Exception as e:
            _warnings.warn(f"Bitstream feature error: {e}")
            bitstream_ai_prob = 0.5

        # ============================================
        # MODEL 2: SPN Sensor Pattern Noise
        # ============================================
        try:
            spn_result = self.spn.extract_features(image_path)
            spn_ai_prob = float(spn_result['ai_probability'])
            spn_prnu_score = float(spn_result['prnu_score'])
            spn_is_camera_noise = bool(spn_result['is_camera_noise'])
        except Exception as e:
            _warnings.warn(f"SPN analysis error: {e}")
            spn_ai_prob = 0.5
            spn_prnu_score = 0.0
            spn_is_camera_noise = False

        # ============================================
        # MODEL 3: Camera Signature Analysis
        # ============================================
        is_camera, camera_conf, camera_reasons = self._analyze_camera_signature(width, height)
        camera_real_prob = camera_conf / 100.0

        # ============================================
        # AI MODEL ATTRIBUTION
        # ============================================
        try:
            attribution_result = self.attributor.attribute(image_path)
            attribution = {
                'top_model': attribution_result.top_model,
                'top_confidence': attribution_result.top_confidence,
                'scores': attribution_result.scores,
                'reasoning': attribution_result.reasoning,
                'is_ai_generated': attribution_result.is_ai_generated,
            }
        except Exception as e:
            _warnings.warn(f"Attribution error: {e}")
            attribution = {
                'top_model': 'Unknown',
                'top_confidence': 0.0,
                'scores': {},
                'reasoning': [f'Attribution unavailable: {e}'],
                'is_ai_generated': False,
            }

        # ============================================
        # ENSEMBLE FUSION
        # Bitstream model: 100% (sole decision maker)
        # SPN:               0% (optional investigative heuristic — not fused)
        # Camera signature:  0% (informational only)
        # ============================================
        weights = {'bitstream': 1.00, 'spn': 0.00, 'camera': 0.00}

        ensemble_ai_prob = bitstream_ai_prob

        ensemble_ai_prob = float(max(0.0, min(1.0, ensemble_ai_prob)))

        # Final classification
        AI_THRESHOLD = 0.50
        is_ai = ensemble_ai_prob >= AI_THRESHOLD
        confidence_pct = ensemble_ai_prob * 100 if is_ai else (1 - ensemble_ai_prob) * 100
        label = "🤖 AI-Generated" if is_ai else "📷 Real Photo"

        if return_details:
            result = {
                'probability': ensemble_ai_prob,
                'is_ai': is_ai,
                'confidence': confidence_pct,
                'label': label,
                'real_probability': 1 - ensemble_ai_prob,
                'ai_probability': ensemble_ai_prob,
                'image_size': (width, height),
                'megapixels': megapixels,
                'image_format': img_format,
                'model': 'Bitstream Forensics (100%) — SPN available as investigative heuristic only',
                'method': 'Trained bitstream ensemble (alpha_SPN=0.00)',

                # Attribution results
                'attribution': attribution,

                # Individual model results
                'model_breakdown': {
                    'bitstream': {
                        'ai_probability': bitstream_ai_prob,
                        'individual_probs': getattr(self, '_last_individual_probs', []),
                        'ensemble_seeds': self.ensemble_seeds,
                        'weight': weights['bitstream'],
                        'contribution': weights['bitstream'] * bitstream_ai_prob
                    },
                    'spn': {
                        'ai_probability': spn_ai_prob,
                        'prnu_score': spn_prnu_score,
                        'is_camera_noise': spn_is_camera_noise,
                        'weight': weights['spn'],
                        'contribution': weights['spn'] * spn_ai_prob
                    },
                    'camera_signature': {
                        'is_camera_likely': is_camera,
                        'confidence': camera_conf,
                        'real_probability': camera_real_prob,
                        'weight': 0,
                        'contribution': 0,
                        'reasons': camera_reasons,
                        'note': 'Informational only — not used in score'
                    }
                },

                'ensemble_method': 'Bitstream ensemble only (SPN weight alpha=0.00)'
            }

            return result
        else:
            return ensemble_ai_prob


def test_ensemble_detector():
    """Test the ensemble detector"""
    print("="*80)
    print("TESTING ENSEMBLE AI DETECTOR")
    print("Bitstream + SPN + Camera Signature + AI Attribution")
    print("="*80)
    print()

    detector = EnsembleAIDetector()

    test_cases = [
        ('test_sample/a1.JPG', 'AI', '1712×1698 AI image'),
        ('test_sample/re7.JPG', 'Real', '4032×3024 iPhone 15 Plus photo'),
    ]

    print("\n📋 Testing Images:")
    print("-"*80)

    for image_path, expected, description in test_cases:
        print(f"\n{'='*80}")
        print(f"🖼️  Image: {image_path}")
        print(f"   Description: {description}")
        print(f"   Expected: {expected}")
        print("="*80)

        result = detector.predict(image_path, return_details=True)

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue

        breakdown = result['model_breakdown']

        print(f"\n📊 Individual Model Results:")
        print("-"*80)

        print(f"1️⃣  Bitstream Forensics (JPEG/PNG/WebP Analysis):")
        print(f"    AI Probability: {breakdown['bitstream']['ai_probability']:.4f}")
        print(f"    Weight: {breakdown['bitstream']['weight']:.2f}")
        print(f"    Contribution: {breakdown['bitstream']['contribution']:.4f}")

        print(f"\n2️⃣  SPN Sensor Pattern Noise:")
        print(f"    AI Probability: {breakdown['spn']['ai_probability']:.4f}")
        print(f"    PRNU Score (camera-ness): {breakdown['spn']['prnu_score']:.3f}")
        print(f"    Camera Noise Detected: {'✓ YES' if breakdown['spn']['is_camera_noise'] else '✗ NO'}")
        print(f"    Weight: {breakdown['spn']['weight']:.2f}")
        print(f"    Contribution: {breakdown['spn']['contribution']:.4f}")

        print(f"\n3️⃣  Camera Signature Analysis:")
        print(f"    Camera Likely: {'✓ YES' if breakdown['camera_signature']['is_camera_likely'] else '✗ NO'}")
        print(f"    Confidence: {breakdown['camera_signature']['confidence']:.1f}%")
        print(f"    Weight: {breakdown['camera_signature']['weight']:.2f}")
        print(f"    Reasons:")
        for reason in breakdown['camera_signature']['reasons']:
            print(f"      - {reason}")

        print(f"\n4️⃣  AI Model Attribution:")
        attr = result['attribution']
        print(f"    Top model: {attr['top_model']} ({attr['top_confidence']:.1f}%)")
        print(f"    Score breakdown:")
        for model_name, score in sorted(attr['scores'].items(), key=lambda x: -x[1]):
            print(f"      {model_name:<22} {score:.1f}%")
        print(f"    Clues:")
        for clue in attr['reasoning']:
            print(f"      • {clue}")

        print(f"\n🎯 ENSEMBLE RESULT:")
        print("="*80)
        print(f"Final AI Probability  : {result['ai_probability']:.4f}")
        print(f"Final Real Probability: {result['real_probability']:.4f}")
        print(f"Image Format          : {result.get('image_format', 'unknown')}")
        print(f"\nClassification: {result['label']}")
        print(f"Confidence: {result['confidence']:.1f}%")

        is_correct = (result['is_ai'] and expected == 'AI') or (not result['is_ai'] and expected == 'Real')
        print(f"\nCorrect: {'✅ YES' if is_correct else '❌ NO'}")

    print("\n" + "="*80)
    print("Ensemble testing complete!")
    print("="*80)


if __name__ == '__main__':
    test_ensemble_detector()
