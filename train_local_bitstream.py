#!/usr/bin/env python3
"""
Local Bitstream Training Script - OFFLINE VERSION
Trains AI detector using local datasets in datasets/1 folder
"""

import numpy as np
import glob
import os
import cv2
from scipy.fftpack import dct
from scipy.stats import kurtosis, skew
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports complete!")

# BitstreamFeatureExtractor class
class BitstreamFeatureExtractor:
    """Extract forensic features directly from JPEG compression data"""
    
    def extract_features(self, image_path):
        """Extract 70 bitstream features from JPEG"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = []
            features.extend(self._extract_dct_features(gray))
            features.extend(self._extract_quantization_features(gray))
            features.extend(self._extract_blocking_artifacts(gray))
            features.extend(self._extract_benford_features(gray))
            features.extend(self._extract_double_compression_features(gray))
            features.extend(self._extract_frequency_features(gray))
            features.extend(self._extract_dct_histogram(gray))
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            return None
    
    def _extract_dct_features(self, gray_img):
        """DCT coefficient statistics (10 features)"""
        features = []
        h, w = gray_img.shape
        block_size = 8
        dct_coeffs = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten())
        
        dct_coeffs = np.array(dct_coeffs)
        
        features.append(np.mean(dct_coeffs))
        features.append(np.std(dct_coeffs))
        features.append(np.median(dct_coeffs))
        features.append(np.percentile(dct_coeffs, 25))
        features.append(np.percentile(dct_coeffs, 75))
        features.append(np.min(dct_coeffs))
        features.append(np.max(dct_coeffs))
        features.append(np.sum(np.abs(dct_coeffs)))
        features.append(kurtosis(dct_coeffs))
        features.append(skew(dct_coeffs))
        
        return features
    
    def _extract_quantization_features(self, gray_img):
        """Quantization patterns (4 features)"""
        features = []
        h, w = gray_img.shape
        block_size = 8
        
        vertical_gradients = []
        horizontal_gradients = []
        
        for i in range(block_size, h - block_size, block_size):
            grad = np.abs(gray_img[i, :].astype(float) - gray_img[i-1, :].astype(float))
            vertical_gradients.append(np.mean(grad))
        
        for j in range(block_size, w - block_size, block_size):
            grad = np.abs(gray_img[:, j].astype(float) - gray_img[:, j-1].astype(float))
            horizontal_gradients.append(np.mean(grad))
        
        features.append(np.mean(vertical_gradients) if vertical_gradients else 0)
        features.append(np.std(vertical_gradients) if vertical_gradients else 0)
        features.append(np.mean(horizontal_gradients) if horizontal_gradients else 0)
        features.append(np.std(horizontal_gradients) if horizontal_gradients else 0)
        
        return features
    
    def _extract_blocking_artifacts(self, gray_img):
        """8x8 blocking detection (3 features)"""
        features = []
        h, w = gray_img.shape
        block_size = 8
        boundary_strength = []
        
        for i in range(0, h, block_size):
            if i > 0 and i < h - 1:
                diff = np.abs(gray_img[i, :].astype(float) - gray_img[i-1, :].astype(float))
                boundary_strength.append(np.mean(diff))
        
        for j in range(0, w, block_size):
            if j > 0 and j < w - 1:
                diff = np.abs(gray_img[:, j].astype(float) - gray_img[:, j-1].astype(float))
                boundary_strength.append(np.mean(diff))
        
        if boundary_strength:
            features.append(np.mean(boundary_strength))
            features.append(np.std(boundary_strength))
            features.append(np.max(boundary_strength))
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_benford_features(self, gray_img):
        """Benford's Law compliance (10 features)"""
        features = []
        h, w = gray_img.shape
        block_size = 8
        first_digits = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                for coeff in dct_block.flatten():
                    if abs(coeff) >= 1:
                        first_digit = int(str(abs(int(coeff)))[0])
                        if first_digit > 0:
                            first_digits.append(first_digit)
        
        benford_expected = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
        
        if len(first_digits) > 100:
            digit_counts = Counter(first_digits)
            actual_dist = [digit_counts.get(i, 0) / len(first_digits) for i in range(1, 10)]
            chi_square = sum((actual - expected)**2 / expected 
                           for actual, expected in zip(actual_dist, benford_expected))
            features.append(chi_square)
            features.extend(actual_dist)
        else:
            features.extend([0] * 10)
        
        return features
    
    def _extract_double_compression_features(self, gray_img):
        """Double compression detection (4 features)"""
        features = []
        h, w = gray_img.shape
        block_size = 8
        dct_values = []
        
        for i in range(0, h - block_size + 1, block_size * 2):
            for j in range(0, w - block_size + 1, block_size * 2):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_values.extend(dct_block.flatten())
        
        dct_values = np.array(dct_values)
        hist, _ = np.histogram(dct_values, bins=50)
        fft = np.fft.fft(hist)
        power_spectrum = np.abs(fft[:25])
        
        features.append(np.mean(power_spectrum))
        features.append(np.std(power_spectrum))
        features.append(np.max(power_spectrum))
        peaks = (power_spectrum > np.mean(power_spectrum) + np.std(power_spectrum)).sum()
        features.append(float(peaks))
        
        return features
    
    def _extract_frequency_features(self, gray_img):
        """Frequency domain analysis (9 features)"""
        features = []
        dct_img = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')
        h, w = dct_img.shape
        
        low_freq = dct_img[:h//4, :w//4]
        mid_freq = dct_img[h//4:h//2, w//4:w//2]
        high_freq = dct_img[h//2:, w//2:]
        
        features.append(np.mean(np.abs(low_freq)))
        features.append(np.std(np.abs(low_freq)))
        features.append(np.mean(np.abs(mid_freq)))
        features.append(np.std(np.abs(mid_freq)))
        features.append(np.mean(np.abs(high_freq)))
        features.append(np.std(np.abs(high_freq)))
        
        total_energy = np.sum(np.abs(dct_img))
        if total_energy > 0:
            features.append(np.sum(np.abs(low_freq)) / total_energy)
            features.append(np.sum(np.abs(mid_freq)) / total_energy)
            features.append(np.sum(np.abs(high_freq)) / total_energy)
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_dct_histogram(self, gray_img):
        """DCT histogram (30 features)"""
        h, w = gray_img.shape
        block_size = 8
        dct_coeffs = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten())
        
        hist, _ = np.histogram(dct_coeffs, bins=30)
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-10)
        
        return hist.tolist()

print("✅ BitstreamFeatureExtractor defined")

# Local dataset paths
print("\n🔍 Scanning local datasets folder...")
base_path = "datasets/1"

# Define all real and fake paths
real_paths = [
    "datasets/1/Dataset/Train/Real",
    "datasets/1/Dataset/Test/Real",
    "datasets/1/Dataset/Validation/Real",
    "datasets/1/archive (3)/train/REAL",
    "datasets/1/archive (3)/test/REAL",
    "datasets/1/real_vs_fake/real-vs-fake/train/real",
    "datasets/1/real_vs_fake/real-vs-fake/test/real",
    "datasets/1/real_vs_fake/real-vs-fake/valid/real",
    "datasets/1/archive (1)/real",
]

fake_paths = [
    "datasets/1/Dataset/Train/Fake",
    "datasets/1/Dataset/Test/Fake",
    "datasets/1/Dataset/Validation/Fake",
    "datasets/1/archive (3)/train/FAKE",
    "datasets/1/archive (3)/test/FAKE",
    "datasets/1/real_vs_fake/real-vs-fake/train/fake",
    "datasets/1/real_vs_fake/real-vs-fake/test/fake",
    "datasets/1/real_vs_fake/real-vs-fake/valid/fake",
    "datasets/1/archive (1)/fake",
]

# Verify paths and count images
print("\n📁 Found REAL image folders:")
verified_real_paths = []
for p in real_paths:
    if os.path.exists(p):
        img_count = sum(len(glob.glob(os.path.join(p, f'*.{ext}'))) 
                       for ext in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'])
        if img_count > 0:
            print(f"   {p} ({img_count:,} images)")
            verified_real_paths.append(p)

print("\n📁 Found AI/FAKE image folders:")
verified_fake_paths = []
for p in fake_paths:
    if os.path.exists(p):
        img_count = sum(len(glob.glob(os.path.join(p, f'*.{ext}'))) 
                       for ext in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'])
        if img_count > 0:
            print(f"   {p} ({img_count:,} images)")
            verified_fake_paths.append(p)

if len(verified_real_paths) == 0 or len(verified_fake_paths) == 0:
    print("\n❌ ERROR: Could not find datasets!")
    exit(1)

# Extract features
extractor = BitstreamFeatureExtractor()
X = []
y = []

image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']

print("\n🔍 Extracting bitstream features from REAL images...")
print("⏰ This will take 2-3 hours for ~300k real images...")
real_count = 0
for folder in tqdm(verified_real_paths, desc="Real folders"):
    for ext in image_extensions:
        # Limit to 100k images per folder
        for img_file in glob.glob(os.path.join(folder, ext))[:100000]:
            features = extractor.extract_features(img_file)
            if features is not None:
                X.append(features)
                y.append(0)  # REAL
                real_count += 1
                if real_count % 10000 == 0:
                    print(f"   Processed {real_count:,} real images...")

print(f"✅ Extracted features from {real_count:,} real images")

print("\n🔍 Extracting bitstream features from AI/FAKE images...")
print("⏰ This will take 2-3 hours for ~300k fake images...")
fake_count = 0
for folder in tqdm(verified_fake_paths, desc="Fake folders"):
    for ext in image_extensions:
        # Limit to 100k images per folder
        for img_file in glob.glob(os.path.join(folder, ext))[:100000]:
            features = extractor.extract_features(img_file)
            if features is not None:
                X.append(features)
                y.append(1)  # AI
                fake_count += 1
                if fake_count % 10000 == 0:
                    print(f"   Processed {fake_count:,} fake images...")

print(f"✅ Extracted features from {fake_count:,} AI images")

X = np.array(X)
y = np.array(y)

print(f"\n📊 Dataset Summary:")
print(f"   Total samples: {len(X):,}")
print(f"   REAL images: {np.sum(y == 0):,}")
print(f"   AI images: {np.sum(y == 1):,}")
print(f"   Features: {X.shape[1]}")

if len(X) == 0:
    print("\n❌ ERROR: No training data extracted!")
    exit(1)

# Split data
print("\n📊 Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")

# Normalize features
print("🔄 Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data prepared for training")

# Train models
models = {}
results = {}

# Random Forest - OPTIMIZED
print("\n" + "="*60)
print("🌲 Training Random Forest (300 trees)")
print("⏰ Estimated time: 2-3 hours")
print("="*60)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    random_state=42, 
    n_jobs=-1,
    verbose=1
)
rf.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
models['random_forest'] = rf
results['random_forest'] = rf_acc
print(f"✅ Random Forest Accuracy: {rf_acc*100:.2f}%")

# Gradient Boosting - OPTIMIZED
print("\n" + "="*60)
print("🎯 Training Gradient Boosting (200 trees)")
print("⏰ Estimated time: 1-2 hours")
print("="*60)
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1, 
    max_depth=7,
    random_state=42,
    verbose=1
)
gb.fit(X_train_scaled, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test_scaled))
models['gradient_boosting'] = gb
results['gradient_boosting'] = gb_acc
print(f"✅ Gradient Boosting Accuracy: {gb_acc*100:.2f}%")

# Neural Network - OPTIMIZED
print("\n" + "="*60)
print("🧠 Training Neural Network (1000 iterations)")
print("⏰ Estimated time: 1-2 hours")
print("="*60)
nn = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=1000,
    early_stopping=False,
    random_state=42, 
    learning_rate_init=0.001,
    verbose=True
)
nn.fit(X_train_scaled, y_train)
nn_acc = accuracy_score(y_test, nn.predict(X_test_scaled))
results['neural_network'] = nn_acc
models['neural_network'] = nn
print(f"✅ Neural Network Accuracy: {nn_acc*100:.2f}%")

# Pick best model
print("\n" + "="*60)
print("🏆 MODEL COMPARISON")
print("="*60)
for name, acc in results.items():
    print(f"   {name}: {acc*100:.2f}%")

best_name = max(results, key=results.get)
best_model = models[best_name]
best_acc = results[best_name]

print(f"\n🏆 Best Model: {best_name.upper()}")
print(f"   Accuracy: {best_acc*100:.2f}%")

# Detailed evaluation
y_pred = best_model.predict(X_test_scaled)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['REAL', 'AI']))

print("\n📈 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_name}')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], ['REAL', 'AI'])
plt.yticks([0, 1], ['REAL', 'AI'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("✅ Saved confusion matrix to confusion_matrix.png")

# Save best model
print("\n💾 Saving model...")
model_data = {
    'model': best_model,
    'scaler': scaler,
    'model_type': best_name,
    'accuracy': best_acc,
    'n_features': 70,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'feature_names': [
        'dct_mean', 'dct_std', 'dct_median', 'dct_q25', 'dct_q75', 
        'dct_min', 'dct_max', 'dct_sum_abs', 'dct_kurtosis', 'dct_skew',
        'quant_v_mean', 'quant_v_std', 'quant_h_mean', 'quant_h_std',
        'block_mean', 'block_std', 'block_max',
        'benford_chi2', 'benford_1', 'benford_2', 'benford_3', 'benford_4',
        'benford_5', 'benford_6', 'benford_7', 'benford_8', 'benford_9',
        'double_mean', 'double_std', 'double_max', 'double_peaks',
        'freq_low_mean', 'freq_low_std', 'freq_mid_mean', 'freq_mid_std',
        'freq_high_mean', 'freq_high_std', 'freq_low_energy', 'freq_mid_energy', 'freq_high_energy'
    ] + [f'dct_hist_{i}' for i in range(30)]
}

torch.save(model_data, 'bitstream_detector_local.pth')
joblib.dump(best_model, 'bitstream_detector_model_local.pkl')
joblib.dump(scaler, 'bitstream_scaler_local.pkl')

# Save metadata
with open('model_info_local.txt', 'w') as f:
    f.write(f"🎯 LOCAL BITSTREAM TRAINING RESULTS\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"Model Type: {best_name}\n")
    f.write(f"Accuracy: {best_acc*100:.2f}%\n")
    f.write(f"Features: 70 (Pure Bitstream)\n")
    f.write(f"Training Method: DCT, Quantization, Benford's Law\n")
    f.write(f"Training Samples: {len(X_train):,}\n")
    f.write(f"Test Samples: {len(X_test):,}\n")
    f.write(f"Real Images: {real_count:,}\n")
    f.write(f"Fake Images: {fake_count:,}\n")
    f.write(f"Total Images: {real_count + fake_count:,}\n")
    f.write(f"\nDatasets Used:\n")
    f.write(f"  - CIFAKE Dataset\n")
    f.write(f"  - 140k Real and Fake Faces\n")
    f.write(f"  - Additional deepfake datasets\n")
    f.write(f"\nAll Model Accuracies:\n")
    for name, acc in results.items():
        f.write(f"  - {name}: {acc*100:.2f}%\n")

print("\n✅ MODEL SAVED!")
print("\n📥 Generated files:")
print("   - bitstream_detector_local.pth (PyTorch format - MAIN FILE)")
print("   - bitstream_detector_model_local.pkl (model only)")
print("   - bitstream_scaler_local.pkl (scaler only)")
print("   - model_info_local.txt (training details)")
print("   - confusion_matrix.png (visualization)")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"✨ Final Accuracy: {best_acc*100:.2f}%")
print(f"📊 Trained on: {len(X_train):,} images")
print(f"🧪 Tested on: {len(X_test):,} images")
print("\n🚀 Ready to use! Test with:")
print("   python3 test_new_model.py")
print("="*60)
