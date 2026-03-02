"""
Ensemble AI Detector - Uses Bitstream Forensics Model
High-accuracy detection using JPEG compression forensics (DCT, quantization, Benford's Law)

Model Accuracy: 97.2% (trained on 348k images)
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from bitstream_features import BitstreamFeatureExtractor


class EnsembleAIDetector:
    """
    Bitstream forensics-based AI detector using:
    - 70 JPEG compression features (DCT, quantization, Benford's Law)
    - Camera signature analysis
    
    Model Accuracy: 97.2%
    - Real detection: 97.4%
    - AI detection: 97.0%
    - False positive rate: 2.6%
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
                 model_path='bitstream_detector_local.pth',
                 use_clip=False):
        """
        Initialize bitstream detector
        
        Args:
            model_path: Path to bitstream detector model
            use_clip: Ignored (kept for backward compatibility)
        """
        print(f"✓ Using CPU for bitstream analysis")
        
        # Load bitstream model
        print("Loading Bitstream Forensics Model (97.2% accuracy)...")
        import warnings
        warnings.filterwarnings('ignore')
        
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model.verbose = 0  # Suppress verbose output
        
        # Initialize feature extractor
        self.extractor = BitstreamFeatureExtractor()
        print("✓ Bitstream model loaded")
        
        print("\n" + "="*80)
        print("🎯 BITSTREAM DETECTOR READY")
        print("="*80)
        print(f"Model: Random Forest (trained on 348k images)")
        print(f"Accuracy: 97.2% overall")
        print(f"Real detection: 97.4% | AI detection: 97.0%")
        print(f"False positive rate: 2.6%")
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
        Predict using bitstream forensics + camera signature analysis
        
        Args:
            image_path: Path to image
            return_details: Return detailed breakdown
            
        Returns:
            Probability or detailed results dict
        """
        try:
            # Load image for camera analysis
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            megapixels = (width * height) / 1_000_000
            
            # ============================================
            # MODEL: Bitstream Forensics (97.2%)
            # ============================================
            features = self.extractor.extract_features(image_path)
            if features is None:
                # Fallback if feature extraction fails
                bitstream_ai_prob = 0.5
            else:
                features_scaled = self.scaler.transform([features])
                prediction = self.model.predict(features_scaled)[0]
                confidence = self.model.predict_proba(features_scaled)[0]
                bitstream_ai_prob = confidence[1]  # Probability of AI
            
            # ============================================
            # Camera Signature Analysis
            # ============================================
            is_camera, camera_conf, camera_reasons = self._analyze_camera_signature(width, height)
            camera_real_prob = camera_conf / 100.0
            
            # ============================================
            # ENSEMBLE FUSION
            # ============================================
            camera_override = False
            if is_camera and camera_conf > 50:
                camera_override = True
            
            if camera_override:
                # Give high weight to camera signature for real photos
                weights = {
                    'bitstream': 0.30,
                    'camera': 0.70
                }
            else:
                # Normal weights
                weights = {
                    'bitstream': 0.80,
                    'camera': 0.20
                }
            
            ensemble_ai_prob = (
                weights['bitstream'] * bitstream_ai_prob +
                weights['camera'] * (1 - camera_real_prob)
            )
            
            # Smart adjustment for camera override
            if camera_override and ensemble_ai_prob > 0.5:
                reduction = min(0.45, camera_conf / 100 * 0.5)
                ensemble_ai_prob = ensemble_ai_prob - reduction
                ensemble_ai_prob = max(0.0, min(1.0, ensemble_ai_prob))
            
            # Final classification
            is_ai = ensemble_ai_prob >= 0.5
            confidence_pct = ensemble_ai_prob * 100 if is_ai else (1 - ensemble_ai_prob) * 100
            label = "🤖 AI-Generated" if is_ai else "📷 Real Photo"
            
            if return_details:
                # Determine if camera signature detected
                computational_photography = False
                if camera_override and bitstream_ai_prob > 0.6:
                    computational_photography = True
                
                result = {
                    'probability': ensemble_ai_prob,
                    'is_ai': is_ai,
                    'confidence': confidence_pct,
                    'label': label,
                    'real_probability': 1 - ensemble_ai_prob,
                    'ai_probability': ensemble_ai_prob,
                    'image_size': (width, height),
                    'megapixels': megapixels,
                    'model': 'Bitstream Forensics + Camera Signature (97.2%)',
                    'method': 'JPEG compression forensics with camera detection',
                    
                    # Special notes
                    'camera_override_applied': camera_override,
                    'computational_photography_detected': computational_photography,
                    'note': 'Smartphone photo (AI-enhanced but real photo)' if computational_photography else None,
                    
                    # Individual model results
                    'model_breakdown': {
                        'bitstream': {
                            'ai_probability': bitstream_ai_prob,
                            'weight': weights['bitstream'],
                            'contribution': weights['bitstream'] * bitstream_ai_prob
                        },
                        'camera_signature': {
                            'is_camera_likely': is_camera,
                            'confidence': camera_conf,
                            'real_probability': camera_real_prob,
                            'weight': weights['camera'],
                            'contribution': weights['camera'] * (1 - camera_real_prob),
                            'reasons': camera_reasons
                        }
                    },
                    
                    'ensemble_method': 'Bitstream forensics with camera signature weighting'
                }
                
                return result
            else:
                return ensemble_ai_prob
                
        except Exception as e:
            print(f"❌ Ensemble prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            if return_details:
                return {
                    'probability': 0.5,
                    'is_ai': False,
                    'confidence': 50.0,
                    'label': '❓ Error',
                    'error': str(e)
                }
            return 0.5


def test_ensemble_detector():
    """Test the ensemble detector"""
    print("="*80)
    print("TESTING ENSEMBLE AI DETECTOR")
    print("Multiple models working together for maximum accuracy")
    print("="*80)
    print()
    
    # Try to load with CLIP, fall back if not available
    try:
        detector = EnsembleAIDetector(use_clip=True)
    except:
        print("⚠️  CLIP not available, using ResNet50 + Forensics")
        detector = EnsembleAIDetector(use_clip=False)
    
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
        
        # Show individual model results
        print(f"\n📊 Individual Model Results:")
        print("-"*80)
        
        breakdown = result['model_breakdown']
        
        print(f"1️⃣  ResNet50 (98.12% accuracy):")
        print(f"    AI Probability: {breakdown['resnet50']['ai_probability']:.4f}")
        print(f"    Weight: {breakdown['resnet50']['weight']:.2f}")
        print(f"    Contribution: {breakdown['resnet50']['contribution']:.4f}")
        
        if breakdown.get('clip') and breakdown['clip']['enabled']:
            print(f"\n2️⃣  CLIP (Semantic Analysis):")
            print(f"    AI Probability: {breakdown['clip']['ai_probability']:.4f}")
            print(f"    Weight: {breakdown['clip']['weight']:.2f}")
            print(f"    Contribution: {breakdown['clip']['contribution']:.4f}")
        
        print(f"\n3️⃣  JPEG Forensics:")
        print(f"    Real Score: {breakdown['forensics']['real_score']:.4f}")
        print(f"    AI Score: {breakdown['forensics']['ai_score']:.4f}")
        print(f"    Weight: {breakdown['forensics']['weight']:.2f}")
        print(f"    Details:")
        for detail in breakdown['forensics']['details']:
            print(f"      - {detail}")
        
        print(f"\n4️⃣  Camera Signature:")
        print(f"    Camera Likely: {'✓ YES' if breakdown['camera_signature']['is_camera_likely'] else '✗ NO'}")
        print(f"    Confidence: {breakdown['camera_signature']['confidence']:.1f}%")
        print(f"    Weight: {breakdown['camera_signature']['weight']:.2f}")
        print(f"    Reasons:")
        for reason in breakdown['camera_signature']['reasons']:
            print(f"      - {reason}")
        
        # Show ensemble result
        print(f"\n🎯 ENSEMBLE RESULT:")
        print("="*80)
        print(f"Final AI Probability: {result['ai_probability']:.4f}")
        print(f"Final Real Probability: {result['real_probability']:.4f}")
        print(f"Model Agreement: {result['model_agreement']:.2f} (higher = more confident)")
        print(f"\nClassification: {result['label']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        
        is_correct = (result['is_ai'] and expected == 'AI') or (not result['is_ai'] and expected == 'Real')
        print(f"\nCorrect: {'✅ YES' if is_correct else '❌ NO'}")
    
    print("\n" + "="*80)
    print("Ensemble testing complete!")
    print("="*80)


if __name__ == '__main__':
    test_ensemble_detector()
