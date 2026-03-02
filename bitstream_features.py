"""
Bitstream Feature Extraction for AI Image Detection
Extracts DCT coefficients, quantization tables, and compression artifacts
from JPEG files for forensic analysis.
"""

import numpy as np
from PIL import Image
import cv2
from scipy.fftpack import dct
from collections import Counter
import warnings

class BitstreamFeatureExtractor:
    """Extract forensic features directly from JPEG compression data"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, image_path):
        """
        Extract comprehensive bitstream features from JPEG file
        Returns: 1D numpy array of features
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Convert to grayscale for DCT analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract all feature groups
            features = []
            
            # 1. DCT Coefficient Statistics (most important!)
            dct_features = self._extract_dct_features(gray)
            features.extend(dct_features)
            
            # 2. Quantization Pattern Analysis
            quant_features = self._extract_quantization_features(gray)
            features.extend(quant_features)
            
            # 3. Blocking Artifacts (8x8 grid boundaries)
            blocking_features = self._extract_blocking_artifacts(gray)
            features.extend(blocking_features)
            
            # 4. Benford's Law (first digit distribution)
            benford_features = self._extract_benford_features(gray)
            features.extend(benford_features)
            
            # 5. Double Compression Detection
            double_comp_features = self._extract_double_compression_features(gray)
            features.extend(double_comp_features)
            
            # 6. Frequency Domain Statistics
            freq_features = self._extract_frequency_features(gray)
            features.extend(freq_features)
            
            # 7. DCT Histogram (distribution of coefficients)
            histogram_features = self._extract_dct_histogram(gray)
            features.extend(histogram_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            warnings.warn(f"Error extracting features from {image_path}: {e}")
            # Return zeros if extraction fails
            return np.zeros(self._get_feature_size(), dtype=np.float32)
    
    def _extract_dct_features(self, gray_img):
        """Extract DCT coefficient statistics"""
        features = []
        h, w = gray_img.shape
        
        # Process image in 8x8 blocks (JPEG standard)
        block_size = 8
        dct_coeffs = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # Compute DCT for this block
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten())
        
        dct_coeffs = np.array(dct_coeffs)
        
        # Statistical features of DCT coefficients
        features.append(np.mean(dct_coeffs))           # Mean
        features.append(np.std(dct_coeffs))            # Std deviation
        features.append(np.median(dct_coeffs))         # Median
        features.append(np.percentile(dct_coeffs, 25)) # 25th percentile
        features.append(np.percentile(dct_coeffs, 75)) # 75th percentile
        features.append(np.min(dct_coeffs))            # Min
        features.append(np.max(dct_coeffs))            # Max
        features.append(np.sum(np.abs(dct_coeffs)))    # L1 norm
        
        # Kurtosis and Skewness (shape of distribution)
        from scipy.stats import kurtosis, skew
        features.append(kurtosis(dct_coeffs))
        features.append(skew(dct_coeffs))
        
        return features  # 10 features
    
    def _extract_quantization_features(self, gray_img):
        """Analyze quantization patterns (compression quality indicators)"""
        features = []
        h, w = gray_img.shape
        block_size = 8
        
        # Measure gradient discontinuities at block boundaries
        # (stronger in heavily compressed images)
        vertical_gradients = []
        horizontal_gradients = []
        
        for i in range(block_size, h - block_size, block_size):
            # Vertical block boundaries
            grad = np.abs(gray_img[i, :].astype(float) - gray_img[i-1, :].astype(float))
            vertical_gradients.append(np.mean(grad))
        
        for j in range(block_size, w - block_size, block_size):
            # Horizontal block boundaries
            grad = np.abs(gray_img[:, j].astype(float) - gray_img[:, j-1].astype(float))
            horizontal_gradients.append(np.mean(grad))
        
        features.append(np.mean(vertical_gradients) if vertical_gradients else 0)
        features.append(np.std(vertical_gradients) if vertical_gradients else 0)
        features.append(np.mean(horizontal_gradients) if horizontal_gradients else 0)
        features.append(np.std(horizontal_gradients) if horizontal_gradients else 0)
        
        return features  # 4 features
    
    def _extract_blocking_artifacts(self, gray_img):
        """Detect 8x8 JPEG blocking artifacts"""
        features = []
        h, w = gray_img.shape
        
        # Compute difference between adjacent blocks
        block_size = 8
        boundary_strength = []
        
        # Check vertical boundaries
        for i in range(0, h, block_size):
            if i > 0 and i < h - 1:
                diff = np.abs(gray_img[i, :].astype(float) - gray_img[i-1, :].astype(float))
                boundary_strength.append(np.mean(diff))
        
        # Check horizontal boundaries
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
        
        return features  # 3 features
    
    def _extract_benford_features(self, gray_img):
        """
        Check Benford's Law compliance (AI images often violate this)
        Natural images follow Benford's Law in their DCT coefficients
        """
        features = []
        h, w = gray_img.shape
        block_size = 8
        
        # Collect first significant digits from DCT coefficients
        first_digits = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Get first significant digit of non-zero coefficients
                for coeff in dct_block.flatten():
                    if abs(coeff) >= 1:
                        first_digit = int(str(abs(int(coeff)))[0])
                        if first_digit > 0:
                            first_digits.append(first_digit)
        
        # Benford's Law expected distribution
        benford_expected = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
        
        if len(first_digits) > 100:
            # Compute actual distribution
            digit_counts = Counter(first_digits)
            actual_dist = [digit_counts.get(i, 0) / len(first_digits) for i in range(1, 10)]
            
            # Chi-square distance from Benford's Law
            chi_square = sum((actual - expected)**2 / expected 
                           for actual, expected in zip(actual_dist, benford_expected))
            features.append(chi_square)
            
            # Also include the actual distribution (9 features)
            features.extend(actual_dist)
        else:
            features.extend([0] * 10)  # Chi-square + 9 distribution values
        
        return features  # 10 features
    
    def _extract_double_compression_features(self, gray_img):
        """
        Detect double JPEG compression (AI images often show this)
        Real photos: compressed once by camera
        AI images: often saved multiple times or post-processed
        """
        features = []
        h, w = gray_img.shape
        block_size = 8
        
        # Analyze DCT coefficient periodicity (sign of double compression)
        dct_values = []
        
        for i in range(0, h - block_size + 1, block_size * 2):  # Sample every other block
            for j in range(0, w - block_size + 1, block_size * 2):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_values.extend(dct_block.flatten())
        
        dct_values = np.array(dct_values)
        
        # Look for periodic patterns in DCT histogram (double compression signature)
        hist, _ = np.histogram(dct_values, bins=50)
        
        # Compute FFT of histogram to detect periodicity
        fft = np.fft.fft(hist)
        power_spectrum = np.abs(fft[:25])  # First half of spectrum
        
        features.append(np.mean(power_spectrum))
        features.append(np.std(power_spectrum))
        features.append(np.max(power_spectrum))
        
        # Peak detection (multiple peaks suggest double compression)
        peaks = (power_spectrum > np.mean(power_spectrum) + np.std(power_spectrum)).sum()
        features.append(float(peaks))
        
        return features  # 4 features
    
    def _extract_frequency_features(self, gray_img):
        """Extract frequency domain statistics"""
        features = []
        
        # Full image DCT
        dct_img = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')
        
        # Divide into frequency bands (low, mid, high)
        h, w = dct_img.shape
        
        # Low frequency (top-left corner)
        low_freq = dct_img[:h//4, :w//4]
        features.append(np.mean(np.abs(low_freq)))
        features.append(np.std(np.abs(low_freq)))
        
        # Mid frequency
        mid_freq = dct_img[h//4:h//2, w//4:w//2]
        features.append(np.mean(np.abs(mid_freq)))
        features.append(np.std(np.abs(mid_freq)))
        
        # High frequency (bottom-right corner)
        high_freq = dct_img[h//2:, w//2:]
        features.append(np.mean(np.abs(high_freq)))
        features.append(np.std(np.abs(high_freq)))
        
        # Ratio between frequency bands
        total_energy = np.sum(np.abs(dct_img))
        if total_energy > 0:
            features.append(np.sum(np.abs(low_freq)) / total_energy)
            features.append(np.sum(np.abs(mid_freq)) / total_energy)
            features.append(np.sum(np.abs(high_freq)) / total_energy)
        else:
            features.extend([0, 0, 0])
        
        return features  # 9 features
    
    def _extract_dct_histogram(self, gray_img):
        """Extract histogram of DCT coefficients"""
        h, w = gray_img.shape
        block_size = 8
        dct_coeffs = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten())
        
        # Create histogram (30 bins)
        hist, _ = np.histogram(dct_coeffs, bins=30)
        
        # Normalize
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-10)
        
        return hist.tolist()  # 30 features
    
    def _get_feature_size(self):
        """Total number of features extracted"""
        return 10 + 4 + 3 + 10 + 4 + 9 + 30  # = 70 features


def test_extraction():
    """Test feature extraction on sample image"""
    extractor = BitstreamFeatureExtractor()
    
    import glob
    test_images = glob.glob('images/re*.jpg')[:2] + glob.glob('images/a*.jpg')[:2]
    
    print("Testing Bitstream Feature Extraction:\n")
    
    for img_path in test_images:
        features = extractor.extract_features(img_path)
        print(f"{img_path}")
        print(f"  Features extracted: {len(features)}")
        print(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"  Sample features: {features[:5]}...")
        print()


if __name__ == "__main__":
    test_extraction()
