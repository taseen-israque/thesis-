import os

class Config:
    """Configuration for small dataset experiment (subjects 1-5)"""
    
    # Dataset paths
    DATASET_ROOT = "dataset"
    BENGALI_DATASET_PATH = os.path.join(DATASET_ROOT, "BHSig260-Bengali", "BHSig260-Bengali")
    HINDI_DATASET_PATH = os.path.join(DATASET_ROOT, "BHSig260-Hindi", "BHSig260-Hindi")
    
    # Small dataset configuration - only subjects 1-5
    USE_SMALL_DATASET = True
    SMALL_DATASET_SUBJECTS = list(range(1, 6))  # Subjects 1-5
    
    # Image processing
    IMAGE_SIZE = (224, 224)
    CHANNELS = 1  # Grayscale
    NORMALIZATION_FACTOR = 255.0
    
    # Data augmentation
    ROTATION_RANGE = 15
    TRANSLATION_RANGE = 0.1
    ZOOM_RANGE = 0.1
    BRIGHTNESS_RANGE = 0.2
    CONTRAST_RANGE = 0.2
    GAUSSIAN_NOISE_VAR = 0.01
    
    # Training parameters - reduced for small dataset
    BATCH_SIZE = 16  # Smaller batch size
    EPOCHS = 20      # Fewer epochs for quick testing
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5     # Early stopping patience
    
    # Model parameters
    NUM_CLASSES = 2  # Genuine vs Forged
    DROPOUT_RATE = 0.5
    MOBILENET_DROPOUT = 0.5
    
    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'resnet50': 0.4,
        'inceptionv3': 0.3,
        'vgg19': 0.3
    }
    
    # YOLOv5 parameters - reduced for small dataset
    YOLO_EPOCHS = 10
    YOLO_BATCH_SIZE = 8
    YOLO_IMG_SIZE = 224
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_IOU_THRESHOLD = 0.45
    
    # Data split ratios (matching original config naming)
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Output directories
    MODELS_DIR = "models_small"
    RESULTS_DIR = "results_small"
    LOGS_DIR = "logs_small"
    PROCESSED_DATA_DIR = "processed_data_small"
    
    # System parameters
    RANDOM_SEED = 42
    NUM_WORKERS = 2  # Reduced for small dataset
    PIN_MEMORY = True
    
    # Evaluation metrics
    METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    # Visualization
    PLOT_TRAINING_CURVES = True
    SAVE_CONFUSION_MATRIX = True
    SAVE_SAMPLE_PREDICTIONS = True
    
    def __init__(self):
        # Create output directories
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        
    def get_model_path(self, model_name: str) -> str:
        """Get the path to save/load a model"""
        return os.path.join(self.MODELS_DIR, f"{model_name}_small.pth")
    
    def get_results_path(self, filename: str) -> str:
        """Get the path to save results"""
        return os.path.join(self.RESULTS_DIR, filename)
    
    def get_logs_path(self, filename: str) -> str:
        """Get the path to save logs"""
        return os.path.join(self.LOGS_DIR, filename)
