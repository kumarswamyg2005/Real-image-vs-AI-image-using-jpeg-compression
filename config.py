"""
Configuration File
Central configuration for all project parameters
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
VIZ_DIR = PROJECT_ROOT / 'visualizations'
CACHE_DIR = PROJECT_ROOT / 'cache'

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, VIZ_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==================== IMAGE STANDARDIZATION ====================
IMAGE_SIZE = (256, 256)  # Target image size (width, height)
JPEG_QUALITY = 90        # JPEG compression quality (1-100)

# ==================== DATASET SPLITTING ====================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Class mapping
CLASS_NAMES = {
    0: 'Real',
    1: 'AI'
}

# ==================== FEATURE EXTRACTION ====================
# Number of features extracted (automatically determined by extractor)
# Approximately 100+ features across different categories

# Feature categories enabled
ENABLE_FREQUENCY_FEATURES = True
ENABLE_STATISTICAL_FEATURES = True
ENABLE_ARTIFACT_FEATURES = True
ENABLE_CHANNEL_FEATURES = True
ENABLE_ZERO_RUN_FEATURES = True

# ==================== MODEL HYPERPARAMETERS ====================

# SVM
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 10.0,
    'gamma': 'scale',
}

# Random Forest
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'random_state': RANDOM_SEED,
}

# XGBoost
XGBOOST_CONFIG = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': RANDOM_SEED,
}

# Logistic Regression
LR_CONFIG = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': RANDOM_SEED,
}

# MLP (Deep Learning)
MLP_CONFIG = {
    'hidden_layers': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
}

# CNN (Deep Learning)
CNN_CONFIG = {
    'input_shape': (1000, 8, 8),  # (n_blocks, 8, 8)
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 16,
}

# ==================== TRAINING SETTINGS ====================
USE_FEATURE_NORMALIZATION = True
USE_CACHE = True  # Cache extracted features for faster loading

# Feature selection
ENABLE_FEATURE_SELECTION = False
N_TOP_FEATURES = 50  # Number of top features to select

# ==================== EVALUATION SETTINGS ====================
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'confusion_matrix',
]

# Visualization settings
PLOT_DPI = 300
PLOT_FORMAT = 'png'
FIGURE_SIZE = (10, 8)

# ==================== ROBUSTNESS TESTING ====================
ROBUSTNESS_CONFIG = {
    'jpeg_qualities': [95, 90, 80, 70, 60, 50],
    'crop_ratios': [0.9, 0.8, 0.7],
    'resize_scales': [0.75, 0.5, 0.25],
    'noise_levels': [5, 10, 15],
    'blur_kernels': [3, 5, 7],
}

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
VERBOSE = True

# ==================== COMPUTATIONAL SETTINGS ====================
N_JOBS = -1  # Number of parallel jobs (-1 = use all cores)
GPU_ENABLED = True  # Use GPU for deep learning if available

# ==================== FILE FORMATS ====================
SUPPORTED_IMAGE_FORMATS = [
    '.jpg', '.jpeg', '.png', '.bmp', '.webp',
    '.tiff', '.tif', '.gif', '.heic', '.heif'
]

# ==================== HELPER FUNCTIONS ====================

def get_model_path(model_name):
    """Get path for saving/loading a model."""
    return MODELS_DIR / f"{model_name}.pkl"

def get_results_path(filename):
    """Get path for saving results."""
    return RESULTS_DIR / filename

def get_viz_path(filename):
    """Get path for saving visualizations."""
    return VIZ_DIR / filename

def print_config():
    """Print current configuration."""
    print("="*70)
    print("PROJECT CONFIGURATION")
    print("="*70)
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"JPEG Quality: {JPEG_QUALITY}")
    print(f"Dataset Split: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Feature Normalization: {USE_FEATURE_NORMALIZATION}")
    print(f"Use Cache: {USE_CACHE}")
    print(f"Parallel Jobs: {N_JOBS}")
    print("="*70)


# ==================== VALIDATION ====================
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Dataset ratios must sum to 1.0"
assert 1 <= JPEG_QUALITY <= 100, "JPEG quality must be between 1 and 100"
assert IMAGE_SIZE[0] > 0 and IMAGE_SIZE[1] > 0, "Image size must be positive"
