"""
Configuration for Genomic Foundation Model
"""
import os
from pathlib import Path
import torch

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed_data"   # filtered 92-class dataset
MODEL_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for dir_path in [PROCESSED_DIR, MODEL_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
DATA_FILE = DATA_DIR / "metadata.csv"  # We'll use metadata

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
# These will be auto-detected from your CSV, but you can specify here
# Leave empty [] to auto-detect
CATEGORICAL_FEATURES = []  # Will auto-detect
NUMERICAL_FEATURES = []    # Will auto-detect

# Target column (what you want to predict)
TARGET_COLUMN = 'disease'  # Change if your column name is different

# Features to drop (identifiers, not useful for prediction)
FEATURES_TO_DROP = ['sample_id', 'sampleID', 'subject_id', 'subjectID', 'PMID']

# Missing data
MISSING_THRESHOLD = 0.95  # Drop features with >95% missing

# Data splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Default embedding dimensions
DEFAULT_EMBEDDING_DIM = 8

# Transformer architecture
HIDDEN_DIM = 256
NUM_ENCODER_LAYERS = 6
NUM_ATTENTION_HEADS = 8
FEEDFORWARD_DIM = 1024
DROPOUT = 0.1
USE_POSITIONAL_ENCODING = True

# ============================================================================
# FEATURE CROSS-ATTENTION (NEW)
# ============================================================================
# Whether to use feature cross-attention before the main transformer
USE_FEATURE_CROSS_ATTENTION = True

# Number of cross-attention layers (1–3 recommended)
NUM_CROSS_ATTENTION_LAYERS = 2

# Number of attention heads inside the cross-attention block
CROSS_ATTENTION_HEADS = 4

# Dropout inside cross-attention
CROSS_ATTENTION_DROPOUT = 0.1

# ============================================================================
# TRAINING - PRETRAINING
# ============================================================================
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 1e-4
PRETRAIN_WEIGHT_DECAY = 0.01

# Masking strategy
MASK_PROBABILITY = 0.15

# ============================================================================
# TRAINING - FINE-TUNING
# ============================================================================
FINETUNE_BATCH_SIZE = 32
FINETUNE_EPOCHS = 20
FINETUNE_LR = 2e-5
FINETUNE_WEIGHT_DECAY = 0.01

# Class imbalance handling
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
USE_CLASS_WEIGHTS = True

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Using device: {DEVICE}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch as _torch

    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
