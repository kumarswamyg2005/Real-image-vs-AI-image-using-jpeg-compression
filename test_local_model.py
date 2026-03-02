#!/usr/bin/env python3
"""Test the locally trained bitstream detector model"""

import torch
import glob
import os
from bitstream_features import BitstreamFeatureExtractor

print("="*60)
print("🧪 TESTING LOCAL BITSTREAM MODEL")
print("="*60)

# Load model
print("\n📦 Loading model...")
model_data = torch.load('bitstream_detector_local.pth', map_location='cpu', weights_only=False)
model = model_data['model']
scaler = model_data['scaler']
extractor = BitstreamFeatureExtractor()

print(f"✅ Model loaded!")
print(f"   Type: {model_data['model_type']}")
print(f"   Training accuracy: {model_data['accuracy']*100:.2f}%")
print(f"   Trained on: {model_data['training_samples']:,} images")

# Test on real images
print("\n" + "="*60)
print("🖼️  TESTING ON REAL IMAGES")
print("="*60)

real_folders = ['data/test/real', 'data/val/real']
real_images = []
for folder in real_folders:
    if os.path.exists(folder):
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
            real_images.extend(glob.glob(os.path.join(folder, ext)))

# Test up to 500 real images
test_real = real_images[:500]
correct_real = 0
total_real = 0

print(f"\nTesting {len(test_real)} real images...")

for img_path in test_real:
    features = extractor.extract_features(img_path)
    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        if prediction == 0:  # Correctly identified as REAL
            correct_real += 1
        total_real += 1
        
        if total_real % 100 == 0:
            print(f"   Processed {total_real}/{len(test_real)}...")

real_accuracy = (correct_real / total_real * 100) if total_real > 0 else 0
false_positive_rate = ((total_real - correct_real) / total_real * 100) if total_real > 0 else 0

print(f"\n✅ REAL Images Results:")
print(f"   Correct: {correct_real}/{total_real} ({real_accuracy:.1f}%)")
print(f"   False Positives: {total_real - correct_real} ({false_positive_rate:.1f}%)")

# Test on AI images
print("\n" + "="*60)
print("🤖 TESTING ON AI IMAGES")
print("="*60)

ai_folders = ['data/test/ai', 'data/test/fake']
ai_images = []
for folder in ai_folders:
    if os.path.exists(folder):
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
            ai_images.extend(glob.glob(os.path.join(folder, ext)))

# Test up to 500 AI images
test_ai = ai_images[:500]
correct_ai = 0
total_ai = 0

print(f"\nTesting {len(test_ai)} AI images...")

for img_path in test_ai:
    features = extractor.extract_features(img_path)
    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        if prediction == 1:  # Correctly identified as AI
            correct_ai += 1
        total_ai += 1
        
        if total_ai % 100 == 0:
            print(f"   Processed {total_ai}/{len(test_ai)}...")

ai_accuracy = (correct_ai / total_ai * 100) if total_ai > 0 else 0
false_negative_rate = ((total_ai - correct_ai) / total_ai * 100) if total_ai > 0 else 0

print(f"\n✅ AI Images Results:")
print(f"   Correct: {correct_ai}/{total_ai} ({ai_accuracy:.1f}%)")
print(f"   False Negatives: {total_ai - correct_ai} ({false_negative_rate:.1f}%)")

# Overall summary
print("\n" + "="*60)
print("📊 OVERALL PERFORMANCE")
print("="*60)

total_tested = total_real + total_ai
total_correct = correct_real + correct_ai
overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0

print(f"\n🎯 Total Accuracy: {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
print(f"\n📊 Breakdown:")
print(f"   Real Detection: {real_accuracy:.1f}% ({'✅ Excellent' if real_accuracy >= 95 else '⚠️ Needs Improvement' if real_accuracy >= 85 else '❌ Poor'})")
print(f"   AI Detection: {ai_accuracy:.1f}% ({'✅ Excellent' if ai_accuracy >= 95 else '⚠️ Good' if ai_accuracy >= 85 else '❌ Poor'})")
print(f"   False Positive Rate: {false_positive_rate:.1f}% ({'✅ Excellent' if false_positive_rate <= 2 else '⚠️ Acceptable' if false_positive_rate <= 10 else '❌ Too High'})")

print("\n" + "="*60)
print("🏁 TEST COMPLETE")
print("="*60)

# Compare with previous model
print("\n📈 COMPARISON WITH PREVIOUS KAGGLE MODEL:")
print("   Previous: 78.72% training accuracy, 85% real detection, 15% false positives")
print(f"   Current:  {model_data['accuracy']*100:.2f}% training accuracy, {real_accuracy:.1f}% real detection, {false_positive_rate:.1f}% false positives")

if overall_accuracy > 87:
    print("\n✨ Improvement detected!")
else:
    print("\n⚠️ Performance similar to previous model")
