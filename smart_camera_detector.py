"""
Smart AI Detector with Camera Signature Detection
Uses aspect ratio and common camera resolutions to improve accuracy
Supports both ResNet50 and CLIP-based models
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

class SmartCameraDetector:
    """
    Smart AI detector that recognizes camera signatures
    - Analyzes aspect ratios (cameras have standard ratios)
    - Recognizes common camera resolutions
    - Combines with deep learning model
    """
    
    # Camera used common aspect ratios
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
    
    def __init__(self, model_path='ai_detector_resnet50.pth'):
        """Initialize smart detector"""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                   'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")
        
        # Detect model type and load
        self.model_type = self._detect_model_type(model_path)
        print(f"Loading {self.model_type} model...")
        self.model = self._load_model(model_path)
        
        # Image preprocessing (depends on model type)
        if self.model_type == 'CLIP':
            # CLIP preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            # ResNet50 preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        print("✓ Smart Camera Detector ready!\n")
    
    def _detect_model_type(self, model_path):
        """Detect if model is ResNet50 or CLIP-based"""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check keys to determine model type
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Look for CLIP-specific keys
        if any('clip_model' in key for key in state_dict.keys()):
            return 'CLIP'
        else:
            return 'ResNet50'
    
    def _load_model(self, model_path):
        """Load model based on detected type"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if self.model_type == 'CLIP':
            # Import CLIP only when needed (lazy loading)
            import open_clip
            
            # Define CLIP-based detector class
            class CLIPBasedDetector(nn.Module):
                """CLIP-based AI detector with classifier head"""
                def __init__(self, load_pretrained=False):
                    super().__init__()
                    if load_pretrained:
                        self.clip_model, _, _ = open_clip.create_model_and_transforms(
                            'ViT-B-32', pretrained='openai'
                        )
                    else:
                        # Create CLIP model without loading pretrained weights
                        self.clip_model, _, _ = open_clip.create_model_and_transforms(
                            'ViT-B-32', pretrained=None
                        )
                    self.classifier = nn.Linear(512, 2)  # 2 classes: Real, AI
                
                def forward(self, x):
                    features = self.clip_model.encode_image(x)
                    features = features.float()  # Ensure float32
                    logits = self.classifier(features)
                    probs = torch.softmax(logits, dim=1)
                    return probs[:, 1]  # Return AI probability
            
            # Load CLIP-based model (without downloading pretrained weights)
            model = CLIPBasedDetector(load_pretrained=False)
            
            # Handle both direct state_dict and checkpoint format
            if isinstance(checkpoint, dict) and not any('clip_model' in k for k in checkpoint.keys()):
                # Checkpoint format
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                # Direct state_dict
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            print("✓ CLIP model loaded")
            return model
        else:
            # Load ResNet50 model
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
            
            # Handle both direct state_dict and checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            print("✓ ResNet50 loaded")
            return model
    
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
            # Check against common ratios
            for cam_w, cam_h in self.CAMERA_ASPECT_RATIOS:
                cam_ratio = cam_w / cam_h
                if abs(ratio - cam_ratio) < 0.05:  # 5% tolerance
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
        
        # AI generators typically produce:
        # - Square or near-square (1024×1024, 512×512)
        # - Specific sizes (1024, 2048)
        # - Odd aspect ratios
        ai_typical = False
        if width == height:  # Perfect square
            ai_size_match = width in [512, 768, 1024, 2048]
            if ai_size_match:
                reasons.append(f"Square {width}×{width} (typical AI generator size)")
                confidence -= 40
                ai_typical = True
        
        is_likely_camera = confidence > 40
        
        return is_likely_camera, max(0, min(100, confidence)), reasons
    
    def predict(self, image_path, return_details=False):
        """
        Predict if image is AI-generated or real
        
        SMART LOGIC:
        1. Analyze camera signature
        2. Get model prediction
        3. Adjust based on camera confidence
        
        Args:
            image_path: Path to image file
            return_details: If True, return detailed results
        
        Returns:
            float: Probability of being AI (0-1)
            or dict: Detailed results if return_details=True
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            megapixels = (width * height) / 1_000_000
            
            # Analyze camera signature
            is_camera, camera_conf, reasons = self._analyze_camera_signature(width, height)
            
            # Preprocess
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                model_ai_prob = self.model(image_tensor).item()
            
            # Combine model + camera analysis
            final_ai_prob = model_ai_prob
            adjustment_applied = False
            
            if is_camera and camera_conf > 50:
                # Strong camera signature - reduce AI probability
                reduction = (camera_conf / 100) * 0.7  # Up to 70% reduction
                final_ai_prob = model_ai_prob * (1 - reduction)
                adjustment_applied = True
            elif camera_conf < 20 and model_ai_prob > 0.8:
                # Weak camera signature and model confident it's AI
                # Slight increase in AI probability
                final_ai_prob = min(1.0, model_ai_prob * 1.1)
                adjustment_applied = True
            
            # Classification
            is_ai = final_ai_prob > 0.5
            confidence = final_ai_prob if is_ai else (1 - final_ai_prob)
            confidence_pct = confidence * 100
            
            label = "🤖 AI-Generated" if is_ai else "📷 Real Photo"
            
            model_name = f"{self.model_type} + Camera Analysis"
            
            if return_details:
                return {
                    'probability': final_ai_prob,
                    'is_ai': is_ai,
                    'confidence': confidence_pct,
                    'label': label,
                    'real_probability': 1 - final_ai_prob,
                    'ai_probability': final_ai_prob,
                    'image_size': (width, height),
                    'megapixels': megapixels,
                    'model': model_name,
                    'method': 'Deep Learning + Camera Signature',
                    'model_prediction': model_ai_prob,
                    'camera_likely': is_camera,
                    'camera_confidence': camera_conf,
                    'camera_reasons': reasons,
                    'adjustment_applied': adjustment_applied
                }
            else:
                return final_ai_prob
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
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


def test_smart_camera_detector():
    """Test the smart camera detector"""
    print("="*80)
    print("TESTING SMART CAMERA DETECTOR")
    print("Uses aspect ratio + megapixels + model prediction")
    print("="*80)
    print()
    
    detector = SmartCameraDetector()
    
    test_cases = [
        ('test_sample/a1.JPG', 'AI', '1712×1698 AI image'),
        ('test_sample/re7.JPG', 'Real', '4032×3024 iPhone 15 Plus photo'),
    ]
    
    print("📋 Testing Images:")
    print("-"*80)
    
    results_summary = []
    
    for image_path, expected, description in test_cases:
        print(f"\n🖼️  Image: {image_path}")
        print(f"   Description: {description}")
        print(f"   Expected: {expected}")
        
        result = detector.predict(image_path, return_details=True)
        
        print(f"\n   🔍 Camera Signature Analysis:")
        print(f"      Image Size: {result['image_size'][0]}×{result['image_size'][1]} ({result['megapixels']:.1f}MP)")
        print(f"      Camera Likely: {' YES' if result['camera_likely'] else '❌ NO'}")
        print(f"      Camera Confidence: {result['camera_confidence']:.1f}%")
        print(f"      Reasons:")
        for reason in result['camera_reasons']:
            print(f"        - {reason}")
        
        print(f"\n   🤖 Model Prediction:")
        print(f"      ResNet50 AI Probability: {result['model_prediction']:.4f}")
        print(f"      Adjustment Applied: {'✅ YES' if result['adjustment_applied'] else '❌ NO'}")
        
        print(f"\n   📊 Final Result:")
        print(f"      Final AI Probability: {result['ai_probability']:.4f}")
        print(f"      Final Real Probability: {result['real_probability']:.4f}")
        print(f"      Classification: {result['label']}")
        print(f"      Confidence: {result['confidence']:.1f}%")
        
        if 'error' not in result:
            is_correct = (result['is_ai'] and expected == 'AI') or (not result['is_ai'] and expected == 'Real')
            print(f"      Correct: {'✅ YES' if is_correct else '❌ NO'}")
            results_summary.append(is_correct)
        else:
            print(f"      Error: {result['error']}")
            results_summary.append(False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    correct = sum(results_summary)
    total = len(results_summary)
    print(f"✅ Correct: {correct}/{total}")
    print(f"❌ Incorrect: {total - correct}/{total}")
    print(f"📊 Accuracy: {correct/total*100:.1f}%")
    print("="*80)


if __name__ == '__main__':
    test_smart_camera_detector()
