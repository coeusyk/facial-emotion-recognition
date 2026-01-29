"""
Configuration file for Facial Emotion Recognition Project
===========================================================

Centralized configuration for all training stages, data processing,
and model parameters. Modify these values to customize training behavior.

Usage:
    from config import Config
    
    # Access configuration
    lr = Config.STAGE1_LR
    batch_size = Config.DATA_BATCH_SIZE
"""

from pathlib import Path


class Config:
    """Central configuration for the facial emotion recognition project."""
    
    # ========================================================================
    # PROJECT PATHS
    # ========================================================================
    
    # Root directory
    PROJECT_ROOT = Path(__file__).parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "raw_balanced"
    
    # Model directories
    MODELS_DIR = PROJECT_ROOT / "models"
    STAGE1_CHECKPOINT = MODELS_DIR / "emotion_stage1_warmup.pth"
    STAGE2_CHECKPOINT = MODELS_DIR / "emotion_stage2_progressive.pth"
    STAGE3_CHECKPOINT = MODELS_DIR / "emotion_stage3_deep.pth"
    
    # Log directories
    LOGS_DIR = PROJECT_ROOT / "logs"
    STAGE1_LOG = LOGS_DIR / "emotion_stage1_training.csv"
    STAGE2_LOG = LOGS_DIR / "emotion_stage2_training.csv"
    STAGE3_LOG = LOGS_DIR / "emotion_stage3_training.csv"
    
    # Results directory
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    
    # Dataset parameters
    NUM_CLASSES = 7
    CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    IMG_SIZE = 48  # FER2013 standard
    
    # Data loading
    DATA_BATCH_SIZE = 64
    DATA_NUM_WORKERS = 4
    DATA_VAL_SPLIT = 0.2  # 80/20 train/val split
    
    # Data augmentation (training only)
    AUG_ROTATION_RANGE = 10  # degrees
    AUG_HORIZONTAL_FLIP = True
    AUG_TRANSLATE = (0.2, 0.2)  # fraction of image
    AUG_SCALE = (0.8, 1.2)
    AUG_RANDOM_ERASING = 0.5  # probability
    
    # ImageNet normalization (for VGG16 pretrained weights)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Grayscale normalization (convert to single channel)
    GRAYSCALE_MEAN = [0.485]  # Average of RGB channels
    GRAYSCALE_STD = [0.229]
    
    # Class weights
    CLASS_WEIGHT_METHOD = 'effective_number'  # 'inverse_frequency' or 'effective_number'
    EFFECTIVE_NUMBER_BETA = 0.9999  # For Effective Number method
    
    # ========================================================================
    # IMAGE PREPROCESSING (Unsharp Mask + CLAHE)
    # ========================================================================
    
    # Enable/disable preprocessing pipeline
    PREPROCESSING_ENABLED = True
    
    # Unsharp Mask parameters (sharpens edges, recovers detail from downsampling)
    UNSHARP_RADIUS = 2.0       # Blur kernel radius (1-5, recommended: 2.0 for 48×48)
    UNSHARP_PERCENT = 150      # Sharpening strength % (100-200, recommended: 150)
    UNSHARP_THRESHOLD = 3      # Minimum brightness change to sharpen (0-10, recommended: 3)
    
    # CLAHE parameters (normalizes contrast across varying lighting)
    CLAHE_CLIP_LIMIT = 2.0     # Contrast limiting threshold (1.0-4.0, recommended: 2.0)
    CLAHE_TILE_GRID = (8, 8)   # Grid dimensions for adaptive equalization (recommended: (8, 8))
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    # VGG16 architecture
    MODEL_BACKBONE = 'vgg16'
    MODEL_PRETRAINED = True
    MODEL_INPUT_CHANNELS = 1  # Grayscale
    
    # Classifier head
    CLASSIFIER_HIDDEN_DIMS = [512, 256]  # Two hidden layers
    CLASSIFIER_DROPOUT = 0.5
    CLASSIFIER_BATCH_NORM = True
    
    # VGG16 Block structure (for progressive unfreezing)
    VGG16_BLOCK_RANGES = {
        1: (0, 5),    # Block 1: features[0-4]
        2: (5, 10),   # Block 2: features[5-9]
        3: (10, 17),  # Block 3: features[10-16]
        4: (17, 24),  # Block 4: features[17-23]
        5: (24, 31),  # Block 5: features[24-30]
    }
    
    # ========================================================================
    # STAGE 1: WARMUP TRAINING
    # ========================================================================
    
    STAGE1_EPOCHS = 20
    STAGE1_LR = 1e-4
    STAGE1_WEIGHT_DECAY = 1e-5
    STAGE1_OPTIMIZER = 'Adam'
    
    # LR Warmup
    STAGE1_WARMUP_ENABLED = True
    STAGE1_WARMUP_EPOCHS = 3
    STAGE1_WARMUP_START_FACTOR = 0.01  # Start at 1% of base LR
    
    # Early stopping (disabled for Stage 1)
    STAGE1_EARLY_STOPPING = False
    STAGE1_EARLY_STOP_PATIENCE = None
    
    # Freezing strategy
    STAGE1_FROZEN_BLOCKS = [1, 2, 3, 4, 5]  # All backbone frozen
    STAGE1_TRAINABLE = ['classifier']  # Only classifier trainable
    
    # Success criteria
    STAGE1_TARGET_VAL_ACC_MIN = 40.0
    STAGE1_TARGET_VAL_ACC_MAX = 42.0
    
    # ========================================================================
    # STAGE 2: PROGRESSIVE FINE-TUNING
    # ========================================================================
    
    STAGE2_EPOCHS = 20
    STAGE2_LR = 1e-5
    STAGE2_WEIGHT_DECAY = 1e-4
    STAGE2_OPTIMIZER = 'Adam'
    
    # LR Scheduler
    STAGE2_SCHEDULER = 'ReduceLROnPlateau'
    STAGE2_SCHEDULER_MODE = 'min'  # Reduce on val_loss
    STAGE2_SCHEDULER_PATIENCE = 5
    STAGE2_SCHEDULER_FACTOR = 0.5  # Halve LR
    
    # Early stopping
    STAGE2_EARLY_STOPPING = True
    STAGE2_EARLY_STOP_PATIENCE = 10
    STAGE2_EARLY_STOP_MIN_DELTA = 0.0
    
    # Unfreezing strategy
    STAGE2_FROZEN_BLOCKS = [1, 2, 3]  # Keep blocks 1-3 frozen
    STAGE2_UNFROZEN_BLOCKS = [4, 5]   # Unfreeze blocks 4-5
    
    # Success criteria
    STAGE2_TARGET_VAL_ACC_MIN = 62.0
    STAGE2_TARGET_VAL_ACC_MAX = 65.0
    STAGE2_TARGET_IMPROVEMENT = 20.0  # +20% over Stage 1
    STAGE2_MAX_OVERFITTING_GAP = 0.15  # Max train/val loss gap
    
    # ========================================================================
    # STAGE 3: DEEP FINE-TUNING
    # ========================================================================
    
    STAGE3_EPOCHS = 15
    STAGE3_LR = 5e-6
    STAGE3_WEIGHT_DECAY = 1e-4
    STAGE3_OPTIMIZER = 'Adam'
    
    # LR Scheduler (more aggressive)
    STAGE3_SCHEDULER = 'ReduceLROnPlateau'
    STAGE3_SCHEDULER_MODE = 'min'
    STAGE3_SCHEDULER_PATIENCE = 3  # Shorter patience
    STAGE3_SCHEDULER_FACTOR = 0.3  # More aggressive reduction
    
    # Early stopping
    STAGE3_EARLY_STOPPING = True
    STAGE3_EARLY_STOP_PATIENCE = 8
    STAGE3_EARLY_STOP_MIN_DELTA = 0.0
    
    # Unfreezing strategy
    STAGE3_FROZEN_BLOCKS = [1]         # Keep block 1 frozen
    STAGE3_UNFROZEN_BLOCKS = [2, 3, 4, 5]  # Unfreeze blocks 2-5
    
    # Success criteria
    STAGE3_TARGET_VAL_ACC_MIN = 64.0
    STAGE3_TARGET_VAL_ACC_MAX = 67.0
    STAGE3_TARGET_IMPROVEMENT = 2.0  # +2% over Stage 2
    STAGE3_MAX_OVERFITTING_GAP = 0.20  # Stop if train/val gap exceeds this
    STAGE3_MIN_IMPROVEMENT_THRESHOLD = 1.0  # Stop if gain < 1%
    
    # ========================================================================
    # TRAINING UTILITIES
    # ========================================================================
    
    # Device
    DEVICE_CUDA_ENABLED = True
    DEVICE_FALLBACK_CPU = True
    
    # Reproducibility
    RANDOM_SEED = 42
    CUDNN_DETERMINISTIC = True
    CUDNN_BENCHMARK = False
    
    # Checkpointing
    SAVE_BEST_ONLY = True
    SAVE_CHECKPOINT_FREQ = None  # Save every N epochs (None = disabled)
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    LOG_CSV_ENABLED = True
    LOG_TENSORBOARD_ENABLED = False  # Optional: requires tensorboard
    
    # Metrics
    TRACK_PER_CLASS_ACCURACY = True
    TRACK_CONFUSION_MATRIX = True
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    EVAL_BATCH_SIZE = 64
    EVAL_NUM_WORKERS = 4
    
    # Test-Time Augmentation (TTA)
    TTA_ENABLED = False
    TTA_NUM_CROPS = 5
    
    # ========================================================================
    # DEPLOYMENT
    # ========================================================================
    
    # ONNX Export
    ONNX_OPSET_VERSION = 12
    ONNX_INPUT_NAMES = ['input']
    ONNX_OUTPUT_NAMES = ['output']
    ONNX_DYNAMIC_AXES = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    # Real-time detection
    REALTIME_CAMERA_INDEX = 0
    REALTIME_FACE_CASCADE = 'haarcascade_frontalface_default.xml'
    REALTIME_CONFIDENCE_THRESHOLD = 0.5
    REALTIME_FPS_TARGET = 30
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @classmethod
    def get_preprocessing_config(cls) -> dict:
        """
        Get preprocessing configuration as dictionary.
        
        Returns:
            dict: Preprocessing parameters for Unsharp Mask and CLAHE
        """
        return {
            'unsharp_radius': cls.UNSHARP_RADIUS,
            'unsharp_percent': cls.UNSHARP_PERCENT,
            'unsharp_threshold': cls.UNSHARP_THRESHOLD,
            'clahe_clip_limit': cls.CLAHE_CLIP_LIMIT,
            'clahe_tile_grid': cls.CLAHE_TILE_GRID
        }
    
    @classmethod
    def get_stage_config(cls, stage: int) -> dict:
        """
        Get configuration dictionary for a specific training stage.
        
        Args:
            stage (int): Training stage (1, 2, or 3)
            
        Returns:
            dict: Configuration parameters for the specified stage
        """
        if stage == 1:
            return {
                'epochs': cls.STAGE1_EPOCHS,
                'lr': cls.STAGE1_LR,
                'weight_decay': cls.STAGE1_WEIGHT_DECAY,
                'optimizer': cls.STAGE1_OPTIMIZER,
                'warmup_enabled': cls.STAGE1_WARMUP_ENABLED,
                'warmup_epochs': cls.STAGE1_WARMUP_EPOCHS,
                'warmup_start_factor': cls.STAGE1_WARMUP_START_FACTOR,
                'early_stopping': cls.STAGE1_EARLY_STOPPING,
                'early_stop_patience': cls.STAGE1_EARLY_STOP_PATIENCE,
                'frozen_blocks': cls.STAGE1_FROZEN_BLOCKS,
                'checkpoint_path': cls.STAGE1_CHECKPOINT,
                'log_path': cls.STAGE1_LOG,
            }
        elif stage == 2:
            return {
                'epochs': cls.STAGE2_EPOCHS,
                'lr': cls.STAGE2_LR,
                'weight_decay': cls.STAGE2_WEIGHT_DECAY,
                'optimizer': cls.STAGE2_OPTIMIZER,
                'scheduler': cls.STAGE2_SCHEDULER,
                'scheduler_mode': cls.STAGE2_SCHEDULER_MODE,
                'scheduler_patience': cls.STAGE2_SCHEDULER_PATIENCE,
                'scheduler_factor': cls.STAGE2_SCHEDULER_FACTOR,
                'early_stopping': cls.STAGE2_EARLY_STOPPING,
                'early_stop_patience': cls.STAGE2_EARLY_STOP_PATIENCE,
                'unfrozen_blocks': cls.STAGE2_UNFROZEN_BLOCKS,
                'checkpoint_path': cls.STAGE2_CHECKPOINT,
                'log_path': cls.STAGE2_LOG,
                'prev_checkpoint': cls.STAGE1_CHECKPOINT,
            }
        elif stage == 3:
            return {
                'epochs': cls.STAGE3_EPOCHS,
                'lr': cls.STAGE3_LR,
                'weight_decay': cls.STAGE3_WEIGHT_DECAY,
                'optimizer': cls.STAGE3_OPTIMIZER,
                'scheduler': cls.STAGE3_SCHEDULER,
                'scheduler_mode': cls.STAGE3_SCHEDULER_MODE,
                'scheduler_patience': cls.STAGE3_SCHEDULER_PATIENCE,
                'scheduler_factor': cls.STAGE3_SCHEDULER_FACTOR,
                'early_stopping': cls.STAGE3_EARLY_STOPPING,
                'early_stop_patience': cls.STAGE3_EARLY_STOP_PATIENCE,
                'unfrozen_blocks': cls.STAGE3_UNFROZEN_BLOCKS,
                'max_overfitting_gap': cls.STAGE3_MAX_OVERFITTING_GAP,
                'checkpoint_path': cls.STAGE3_CHECKPOINT,
                'log_path': cls.STAGE3_LOG,
                'prev_checkpoint': cls.STAGE2_CHECKPOINT,
            }
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")
    
    @classmethod
    def print_config(cls, stage: int = None):
        """
        Print configuration summary.
        
        Args:
            stage (int, optional): Print config for specific stage (1, 2, or 3).
                                  If None, prints all configuration.
        """
        print("=" * 80)
        print("FACIAL EMOTION RECOGNITION - CONFIGURATION")
        print("=" * 80)
        
        if stage is None:
            # Print all configuration
            print("\nDATA CONFIGURATION:")
            print(f"  Dataset directory: {cls.DATA_DIR}")
            print(f"  Image size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
            print(f"  Batch size: {cls.DATA_BATCH_SIZE}")
            print(f"  Num classes: {cls.NUM_CLASSES}")
            print(f"  Class names: {cls.CLASS_NAMES}")
            
            print("\nMODEL CONFIGURATION:")
            print(f"  Backbone: {cls.MODEL_BACKBONE}")
            print(f"  Pretrained: {cls.MODEL_PRETRAINED}")
            print(f"  Classifier layers: {cls.CLASSIFIER_HIDDEN_DIMS}")
            print(f"  Dropout: {cls.CLASSIFIER_DROPOUT}")
            
            for s in [1, 2, 3]:
                config = cls.get_stage_config(s)
                print(f"\nSTAGE {s} CONFIGURATION:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
        else:
            # Print stage-specific configuration
            config = cls.get_stage_config(stage)
            print(f"\nSTAGE {stage} CONFIGURATION:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        print("=" * 80)
    
    @classmethod
    def validate_paths(cls):
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print("✓ All directories validated/created")


# ========================================================================
# USAGE EXAMPLES
# ========================================================================

if __name__ == "__main__":
    # Example 1: Print all configuration
    Config.print_config()
    
    # Example 2: Print stage-specific configuration
    print("\n" + "=" * 80)
    Config.print_config(stage=1)
    
    # Example 3: Access individual parameters
    print("\n" + "=" * 80)
    print("EXAMPLE: Accessing configuration parameters")
    print("=" * 80)
    print(f"Stage 1 Learning Rate: {Config.STAGE1_LR}")
    print(f"Stage 2 Epochs: {Config.STAGE2_EPOCHS}")
    print(f"Batch Size: {Config.DATA_BATCH_SIZE}")
    print(f"Class Names: {Config.CLASS_NAMES}")
    
    # Example 4: Get stage configuration as dictionary
    print("\n" + "=" * 80)
    print("EXAMPLE: Stage configuration dictionary")
    print("=" * 80)
    stage2_config = Config.get_stage_config(2)
    for key, value in stage2_config.items():
        print(f"  {key}: {value}")
    
    # Example 5: Validate paths
    print("\n" + "=" * 80)
    print("EXAMPLE: Validating directories")
    print("=" * 80)
    Config.validate_paths()
