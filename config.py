"""
Configuration file for Handwritten Signature Verification System
Based on the research paper "Advancing Handwritten Signature Verification Through Deep Learning"
"""

import os
from pathlib import Path

class Config:
    # Dataset paths
    DATASET_ROOT = "dataset"
    BENGALI_DATASET_PATH = os.path.join(DATASET_ROOT, "BHSig260-Bengali", "BHSig260-Bengali")
    HINDI_DATASET_PATH = os.path.join(DATASET_ROOT, "BHSig260-Hindi", "BHSig260-Hindi")
    
    # Image preprocessing parameters
    IMAGE_SIZE = (299, 299)  # Standard size for InceptionV3 (minimum required)
    CHANNELS = 1  # Grayscale images
    NORMALIZATION_FACTOR = 255.0  # For pixel value normalization
    
    # Data augmentation parameters
    ROTATION_RANGE = 15  # Degrees
    TRANSLATION_RANGE = 0.1  # Fraction of image size
    ZOOM_RANGE = 0.1  # Fraction of image size
    BRIGHTNESS_RANGE = 0.2  # Fraction of brightness
    CONTRAST_RANGE = 0.2  # Fraction of contrast
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    EARLY_STOPPING_PATIENCE = 10  # Number of epochs to wait before early stopping
    
    # Custom learning rates for specific models (lower for pretrained models)
    MODEL_LEARNING_RATES = {
        'vgg19': 0.0001,      # Lower LR for VGG19 (pretrained weights)
        'resnet50': 0.0001,   # Lower LR for ResNet50 (pretrained weights)
        'inceptionv3': 0.001, # Default LR for InceptionV3
        'ensemble': 0.001     # Default LR for ensemble
    }
    
    # Model parameters
    NUM_CLASSES = 2  # Genuine vs Forged
    DROPOUT_RATE = 0.5
    WEIGHT_DECAY = 1e-4
    
    # Training order - exclude InceptionV3 since it's already trained
    TRAIN_MODELS = ['vgg19', 'resnet50']  # Only train VGG19 and ResNet50
    
    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'resnet50': 0.4,
        'inceptionv3': 0.4,
        'vgg19': 0.2
    }
    
    # YOLOv5 parameters
    YOLO_IMG_SIZE = 640
    YOLO_BATCH_SIZE = 16
    YOLO_EPOCHS = 100
    
    # MobileNet parameters
    MOBILENET_ALPHA = 1.0
    MOBILENET_DROPOUT = 0.2
    
    # File paths for saving models
    MODELS_DIR = "models"
    MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_signature_verifier.pth")
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo_signature_detector.pt")
    ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_signature_verifier.pth")
    
    # Results and logs
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"
    
    # Threshold for signature verification
    VERIFICATION_THRESHOLD = 0.5
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Batch processing configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Adjust based on your CPU cores
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
