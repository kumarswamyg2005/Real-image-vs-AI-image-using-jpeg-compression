"""
Bitstream Feature Extraction for AI Image Detection
Extracts DCT coefficients, quantization tables, and compression artifacts
from image files for forensic analysis.

Supported formats
-----------------
  JPEG / JPG  – full DCT + quantisation forensics (original)
  PNG         – PNG filter-type analysis + lossless DCT statistics
  WebP        – decoded as RGB, spatial + frequency forensics
  HEIF / HEIC – decoded via pillow-heif (install separately) or Pillow ≥ 10
  TIFF        – multi-page aware; first frame analysed
  BMP         – lossless; no compression artefacts by definition
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.fftpack import dct
from collections import Counter
import warnings
import struct

# ---------------------------------------------------------------------------
# Optional HEIF / HEIC support
# ---------------------------------------------------------------------------
try:
    import pillow_heif  # pip install pillow-heif
    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Supported format registry
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    # Lossy
    ".jpg", ".jpeg",
    # Lossless
    ".png",
    # Modern web
    ".webp",
    # Apple / mobile lossless
    ".heic", ".heif",
    # Legacy
    ".tiff", ".tif",
    ".bmp",
}


def get_image_format(image_path: str) -> str:
    """Return a normalised format string: 'jpeg', 'png', 'webp', 'heic', 'tiff', 'bmp'."""
    ext = os.path.splitext(image_path)[1].lower()
    mapping = {
        ".jpg": "jpeg", ".jpeg": "jpeg",
        ".png": "png",
        ".webp": "webp",
        ".heic": "heic", ".heif": "heic",
        ".tiff": "tiff", ".tif": "tiff",
        ".bmp": "bmp",
    }
    return mapping.get(ext, "unknown")


def load_image_universal(image_path: str):
    """
    Load any supported image format and return (bgr_array, pil_image).
    Handles JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP.
    Raises ValueError for unsupported / unreadable files.
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in (".heic", ".heif"):
        if not _HEIF_AVAILABLE:
            raise ValueError(
                "HEIC/HEIF support requires 'pillow-heif'. "
                "Install it with: pip install pillow-heif"
            )
        pil = Image.open(image_path).convert("RGB")
    else:
        # Try OpenCV first (handles JPEG, PNG, TIFF, BMP, WebP)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is not None:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return bgr, pil
        # Fallback to Pillow (handles more edge cases)
        try:
            pil = Image.open(image_path)
            if pil.mode not in ("RGB", "RGBA", "L"):
                pil = pil.convert("RGB")
            elif pil.mode == "RGBA":
                # Flatten alpha on white background
                bg = Image.new("RGB", pil.size, (255, 255, 255))
                bg.paste(pil, mask=pil.split()[3])
                pil = bg
            else:
                pil = pil.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot load image '{image_path}': {e}")

    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr, pil


class BitstreamFeatureExtractor:
    """Extract forensic features from JPEG, PNG, WebP, HEIC, TIFF, and BMP images."""
    
    def __init__(self):
        self.feature_names = []
        
    # Crop size for large images — we take a center crop instead of downscaling.
    # Cropping preserves native JPEG compression artifacts (DCT blocks, quantization).
    # Downscaling destroys these artifacts, making real photos look like AI images.
    CROP_SIZE = 256

    def _to_jpeg_gray(self, image_path):
        """
        Load any supported image format and convert to a JPEG-compressed grayscale array.
        - Small images (< CROP_SIZE): used as-is
        - Large images (>= CROP_SIZE): center-cropped to CROP_SIZE×CROP_SIZE
          Cropping preserves original JPEG compression artifacts at native resolution,
          which is critical for DCT forensics. Downscaling destroys these artifacts.
        """
        import io
        pil = Image.open(image_path).convert("RGB")
        w, h = pil.size
        # Center crop large images — preserves JPEG DCT artifacts
        if w >= self.CROP_SIZE and h >= self.CROP_SIZE:
            left = (w - self.CROP_SIZE) // 2
            top  = (h - self.CROP_SIZE) // 2
            pil  = pil.crop((left, top, left + self.CROP_SIZE, top + self.CROP_SIZE))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        pil_jpg = Image.open(buf).convert("L")
        return np.array(pil_jpg, dtype=np.uint8)

    def extract_features(self, image_path, _recompress_buf=None):
        """
        Extract 104 forensic features from any supported image format.

        Non-JPEG images are converted to JPEG in memory first so that all
        DCT / quantisation / Benford features are computed on a consistent
        JPEG-compressed representation.

        Parameters
        ----------
        image_path : str
            Path to the source image. Always used for chroma-subsampling
            (reads raw JPEG header bytes from the original file).
        _recompress_buf : BytesIO or None
            If provided, pixel data is loaded from this buffer instead of
            image_path. Used by the training script to simulate social-media
            re-compression augmentation on real images.

        Returns
        -------
        1D numpy array of length 104, or None on failure.
        """
        try:
            import io

            # ── Load & centre-crop PIL RGB ─────────────────────────────────
            src = _recompress_buf if _recompress_buf is not None else image_path
            pil_rgb = Image.open(src).convert("RGB")
            w, h = pil_rgb.size
            if w >= self.CROP_SIZE and h >= self.CROP_SIZE:
                left = (w - self.CROP_SIZE) // 2
                top  = (h - self.CROP_SIZE) // 2
                pil_rgb = pil_rgb.crop((left, top,
                                        left + self.CROP_SIZE,
                                        top  + self.CROP_SIZE))

            # ── Re-encode to JPEG Q95 → grayscale for compression forensics ─
            buf = io.BytesIO()
            pil_rgb.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            gray = np.array(Image.open(buf).convert("L"), dtype=np.uint8)

            features = []

            # ── Original 70 features ───────────────────────────────────────
            features.extend(self._extract_dct_features(gray))                 # 10
            features.extend(self._extract_quantization_features(gray))        # 4
            features.extend(self._extract_blocking_artifacts(gray))           # 3
            features.extend(self._extract_benford_features(gray))             # 10
            features.extend(self._extract_double_compression_features(gray))  # 4
            features.extend(self._extract_frequency_features(gray))           # 9
            features.extend(self._extract_dct_histogram(gray))                # 30

            # ── New 34 features ────────────────────────────────────────────
            features.extend(self._extract_color_features(pil_rgb))            # 9
            features.extend(self._extract_wavelet_features(gray))             # 9
            features.extend(self._extract_lbp_features(gray))                 # 10
            features.extend(self._extract_edge_features(gray))                # 5
            features.append(self._extract_chroma_subsampling(image_path))     # 1

            result = np.array(features, dtype=np.float32)

            # Pad or truncate to expected feature size
            expected = self._get_feature_size()
            if len(result) < expected:
                result = np.pad(result, (0, expected - len(result)))
            elif len(result) > expected:
                result = result[:expected]

            return result

        except Exception as e:
            warnings.warn(f"Error extracting features from {image_path}: {e}")
            return None  # Caller must check for None — do NOT return zeros (they corrupt training)
    
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
        features.append(np.mean(np.abs(dct_coeffs) < 0.5))  # Fraction near-zero (sparsity)
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
        
        # Skip DC component (index 0) — it equals the total histogram count and is
        # always constant for same-size images, making it a useless max feature.
        ac_spectrum = power_spectrum[1:]
        features.append(np.mean(ac_spectrum))
        features.append(np.std(ac_spectrum))
        features.append(np.max(ac_spectrum))

        # Peak detection (multiple peaks suggest double compression)
        peaks = (ac_spectrum > np.mean(ac_spectrum) + np.std(ac_spectrum)).sum()
        features.append(float(peaks))
        
        return features  # 4 features
    
    def _extract_frequency_features(self, gray_img):
        """Extract frequency domain statistics.

        Bands are defined by normalised Euclidean distance from the DC corner
        (0, 0) so that all three regions are mutually exclusive AND together
        cover 100 % of the DCT matrix.

        Old code used three rectangular patches that covered only ~37.5 % of
        the matrix — the top-right and bottom-left quadrants were silently
        ignored, making the energy-ratio features inaccurate.
        """
        features = []

        # Full-image 2-D DCT (same as before)
        dct_img = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')

        h, w = dct_img.shape

        # Normalised Euclidean distance from DC corner → range [0, 1]
        I, J = np.mgrid[0:h, 0:w]
        freq_norm = np.sqrt((I / h) ** 2 + (J / w) ** 2) / np.sqrt(2)

        # Three mutually exclusive bands covering the full matrix
        low_mask  = freq_norm < 0.25                          # ~10 % of matrix
        high_mask = freq_norm >= 0.55                         # ~48 % of matrix
        mid_mask  = ~low_mask & ~high_mask                    # ~42 % of matrix

        low_freq  = dct_img[low_mask]
        mid_freq  = dct_img[mid_mask]
        high_freq = dct_img[high_mask]

        for band in (low_freq, mid_freq, high_freq):
            features.append(float(np.mean(np.abs(band))))
            features.append(float(np.std(np.abs(band))))

        total_energy = float(np.sum(np.abs(dct_img))) + 1e-10
        features.append(float(np.sum(np.abs(low_freq))  / total_energy))
        features.append(float(np.sum(np.abs(mid_freq))  / total_energy))
        features.append(float(np.sum(np.abs(high_freq)) / total_energy))

        return features  # 9 features — same count, corrected values
    
    def _extract_dct_histogram(self, gray_img):
        """Extract histogram of DCT coefficients"""
        h, w = gray_img.shape
        block_size = 8
        dct_coeffs = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                # Skip DC coefficient (position [0,0]) — it encodes brightness,
                # not compression forensics. Only AC coefficients are forensically
                # meaningful for AI vs real detection.
                ac = dct_block.flatten()[1:]
                dct_coeffs.extend(ac)

        # Range (-10, 10) covers p5–p95 of AC coefficients where the
        # forensic signal lives. Wider ranges leave extreme bins empty
        # (zero-variance dead features). AC-only excludes DC brightness.
        hist, _ = np.histogram(dct_coeffs, bins=30, range=(-10, 10))

        # Normalize
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-10)

        return hist.tolist()  # 30 features
    
    # ------------------------------------------------------------------
    # New feature extractors (v2) — 34 additional features
    # ------------------------------------------------------------------

    def _extract_color_features(self, pil_rgb):
        """RGB + HSV statistics — 9 features.

        AI generators (Midjourney, DALL-E) have unnaturally saturated colours
        and smooth gradients; real camera sensors inject channel-specific noise.
        """
        rgb = np.array(pil_rgb, dtype=np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        features = [
            float(np.mean(r)), float(np.std(r)),   # rgb_r_mean, rgb_r_std
            float(np.mean(g)), float(np.std(g)),   # rgb_g_mean, rgb_g_std
            float(np.mean(b)), float(np.std(b)),   # rgb_b_mean, rgb_b_std
        ]

        # HSV via OpenCV
        bgr = cv2.cvtColor(np.array(pil_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        features.append(float(np.std(hsv[:, :, 0])))    # hsv_h_std
        features.append(float(np.mean(hsv[:, :, 1])))   # hsv_s_mean

        # R-G channel correlation
        r_c = r.flatten() - np.mean(r)
        g_c = g.flatten() - np.mean(g)
        denom = np.linalg.norm(r_c) * np.linalg.norm(g_c) + 1e-10
        features.append(float(np.dot(r_c, g_c) / denom))  # rgb_corr_rg

        return features  # 9 features

    def _extract_wavelet_features(self, gray):
        """Haar wavelet noise residuals — 9 features.

        Camera PRNU produces structured random noise in high-frequency subbands.
        AI synthesis produces smooth or repetitive patterns there.
        """
        try:
            import pywt
            from scipy.stats import kurtosis
            _, (lh, hl, hh) = pywt.dwt2(gray.astype(np.float32), 'haar')
            features = []
            for band in (lh, hl, hh):
                flat = band.flatten()
                features.append(float(np.mean(flat)))
                features.append(float(np.std(flat)))
                features.append(float(kurtosis(flat)))
            return features  # 9 features
        except Exception:
            return [0.0] * 9

    def _extract_lbp_features(self, gray):
        """Uniform Local Binary Pattern histogram — 10 features.

        AI images have unrealistically smooth or repetitively tiled micro-textures
        that show up as distinctive LBP distributions.
        """
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            # For P=8, uniform patterns: values in [0, P+1] = [0, 9] → 10 bins
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-10)
            return hist.tolist()  # 10 features
        except Exception:
            return [0.0] * 10

    def _extract_edge_features(self, gray):
        """Sobel / Laplacian edge statistics — 5 features.

        Midjourney over-sharpens edges relative to real camera optics.
        """
        gray_f = gray.astype(np.float32)
        sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        threshold = float(np.mean(mag) + np.std(mag))
        return [
            float(np.mean(mag)),             # sobel_mean
            float(np.std(mag)),              # sobel_std
            float(np.var(lap)),              # laplacian_var
            float(np.mean(np.abs(lap))),     # laplacian_mean_abs
            float(np.mean(mag > threshold)), # edge_density
        ]  # 5 features

    def _extract_chroma_subsampling(self, image_path):
        """Read chroma subsampling from raw JPEG SOF header — 1 feature.

        Returns
        -------
        0.0 → 4:2:0 (typical camera JPEG)
        1.0 → 4:4:4 (some AI tool outputs)
        0.5 → non-JPEG or parse failure
        """
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ('.jpg', '.jpeg'):
            return 0.5
        try:
            with open(image_path, 'rb') as f:
                data = f.read(4096)   # header is always in first 4 KB
            i = 2  # skip SOI marker (0xFF 0xD8)
            while i < len(data) - 4:
                if data[i] != 0xFF:
                    break
                marker = data[i + 1]
                if marker in (0xC0, 0xC2):  # SOF0 or SOF2
                    nf = data[i + 9]        # number of components
                    if nf >= 3 and i + 14 < len(data):
                        y_samp  = data[i + 11]  # luma  H:V nibbles
                        cb_samp = data[i + 14]  # chroma H:V nibbles
                        return 1.0 if y_samp == cb_samp else 0.0
                    break
                seg_len = struct.unpack('>H', data[i + 2: i + 4])[0]
                i += 2 + seg_len
        except Exception:
            pass
        return 0.5

    # ------------------------------------------------------------------

    def _get_feature_size(self):
        """Total number of features extracted (v2)."""
        # Original 70 + new 34 = 104
        return 10 + 4 + 3 + 10 + 4 + 9 + 30 + 9 + 9 + 10 + 5 + 1  # = 104


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
