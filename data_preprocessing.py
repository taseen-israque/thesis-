"""
Data Preprocessing Module for BHSig260 Dataset
Handles image preprocessing, augmentation, and dataset preparation for signature verification
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config

class SignaturePreprocessor:
    """
    Handles preprocessing of signature images from BHSig260 dataset
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline"""
        return A.Compose([
            A.Rotate(limit=self.config.ROTATION_RANGE, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=self.config.TRANSLATION_RANGE,
                scale_limit=self.config.ZOOM_RANGE,
                rotate_limit=self.config.ROTATION_RANGE,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=self.config.BRIGHTNESS_RANGE,
                contrast_limit=self.config.CONTRAST_RANGE,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from path and convert to grayscale
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Grayscale image as numpy array
        """
        try:
            # Load image using PIL for better format support
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            return np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, apply_augmentation: bool = False) -> np.ndarray:
        """
        Preprocess a single image with all required transformations
        
        Args:
            image: Input image as numpy array
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Preprocessed image
        """
        if image is None:
            return None
            
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply noise removal and background cleaning
        image = self._remove_noise(image)
        
        # Apply data augmentation if requested
        if apply_augmentation:
            augmented = self.augmentation_pipeline(image=image)
            image = augmented['image']
        
        # Resize to standard size
        image = cv2.resize(image, self.config.IMAGE_SIZE)
        
        # Normalize pixel values
        image = image.astype(np.float32) / self.config.NORMALIZATION_FACTOR
        
        # Add channel dimension for CNN input
        image = np.expand_dims(image, axis=-1)
        
        return image
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise and enhance signature clarity
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply morphological operations to clean the image
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        denoised = cv2.equalizeHist(denoised.astype(np.uint8))
        
        return denoised
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse BHSig260 filename to extract metadata
        
        Args:
            filename: BHSig260 filename (e.g., 'B-S-1-G-01.tif')
            
        Returns:
            Dictionary with parsed metadata
        """
        parts = filename.replace('.tif', '').split('-')
        if len(parts) >= 4:
            return {
                'language': parts[0],  # B for Bengali, H for Hindi
                'type': parts[1],      # S for signature
                'person_id': parts[2], # Person identifier
                'category': parts[3],  # G for genuine, F for forged
                'sample_id': parts[4] if len(parts) > 4 else '01'
            }
        return {}

class BHSig260Dataset:
    """
    Dataset class for BHSig260 (Bengali and Hindi signatures)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = SignaturePreprocessor(config)
        self.data = []
        self.labels = []
        self.metadata = []
        
    def load_dataset(self, dataset_path: str, language: str) -> None:
        """
        Load BHSig260 dataset from the specified path
        
        Args:
            dataset_path: Path to the dataset directory
            language: Language identifier ('Bengali' or 'Hindi')
        """
        print(f"Loading {language} dataset from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist!")
            return
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        for person_dir in tqdm(person_dirs, desc=f"Loading {language} signatures"):
            person_path = os.path.join(dataset_path, person_dir)
            
            # Get all signature files for this person
            signature_files = [f for f in os.listdir(person_path) 
                             if f.endswith('.tif')]
            
            for filename in signature_files:
                file_path = os.path.join(person_path, filename)
                
                # Parse filename to get metadata
                metadata = self.preprocessor.parse_filename(filename)
                metadata['person_id'] = person_dir
                metadata['language'] = language
                metadata['file_path'] = file_path
                
                # Determine label (0 for genuine, 1 for forged)
                label = 1 if metadata.get('category') == 'F' else 0
                
                # Load and preprocess image
                image = self.preprocessor.load_image(file_path)
                if image is not None:
                    processed_image = self.preprocessor.preprocess_image(image)
                    
                    if processed_image is not None:
                        self.data.append(processed_image)
                        self.labels.append(label)
                        self.metadata.append(metadata)
        
        print(f"Loaded {len(self.data)} {language} signatures")
    
    def load_both_datasets(self) -> None:
        """Load both Bengali and Hindi datasets"""
        # Load Bengali dataset
        if os.path.exists(self.config.BENGALI_DATASET_PATH):
            self.load_dataset(self.config.BENGALI_DATASET_PATH, 'Bengali')
        
        # Load Hindi dataset (if available)
        if os.path.exists(self.config.HINDI_DATASET_PATH):
            self.load_dataset(self.config.HINDI_DATASET_PATH, 'Hindi')
        
        print(f"Total signatures loaded: {len(self.data)}")
        print(f"Genuine signatures: {sum(1 for label in self.labels if label == 0)}")
        print(f"Forged signatures: {sum(1 for label in self.labels if label == 1)}")
    
    def split_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train, validation, and test sets
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_both_datasets() first.")
        
        # Convert to numpy arrays
        X = np.array(self.data)
        y = np.array(self.labels)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SPLIT,
            stratify=y,
            random_state=self.config.RANDOM_SEED
        )
        
        # Second split: separate validation set from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VALIDATION_SPLIT / (1 - self.config.TEST_SPLIT),
            stratify=y_temp,
            random_state=self.config.RANDOM_SEED
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_augmented_dataset(self, X: np.ndarray, y: np.ndarray, 
                                augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create augmented dataset to balance classes and increase sample diversity
        
        Args:
            X: Input images
            y: Labels
            augmentation_factor: Number of augmented samples per original sample
            
        Returns:
            Tuple of augmented images and labels
        """
        augmented_images = []
        augmented_labels = []
        
        for i, (image, label) in enumerate(tqdm(zip(X, y), desc="Creating augmented dataset")):
            # Add original sample
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Add augmented samples
            for _ in range(augmentation_factor):
                # Remove channel dimension for augmentation
                img_2d = np.squeeze(image)
                
                # Apply augmentation
                augmented = self.preprocessor.augmentation_pipeline(image=img_2d)
                aug_img = augmented['image']
                
                # Normalize and add channel dimension
                aug_img = aug_img.astype(np.float32) / self.config.NORMALIZATION_FACTOR
                aug_img = np.expand_dims(aug_img, axis=-1)
                
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def visualize_samples(self, num_samples: int = 8) -> None:
        """
        Visualize sample signatures from the dataset
        
        Args:
            num_samples: Number of samples to visualize
        """
        if not self.data:
            print("No data to visualize. Load dataset first.")
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Sample both genuine and forged signatures
        genuine_indices = [i for i, label in enumerate(self.labels) if label == 0]
        forged_indices = [i for i, label in enumerate(self.labels) if label == 1]
        
        for i in range(min(num_samples // 2, len(genuine_indices))):
            idx = genuine_indices[i]
            img = np.squeeze(self.data[idx])
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Genuine - {self.metadata[idx]['person_id']}")
            axes[i].axis('off')
        
        for i in range(min(num_samples // 2, len(forged_indices))):
            idx = forged_indices[i]
            img = np.squeeze(self.data[idx])
            axes[i + 4].imshow(img, cmap='gray')
            axes[i + 4].set_title(f"Forged - {self.metadata[idx]['person_id']}")
            axes[i + 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'sample_signatures.png'))
        plt.show()

def main():
    """Main function to test data preprocessing"""
    config = Config()
    config.create_directories()
    
    # Initialize dataset
    dataset = BHSig260Dataset(config)
    
    # Load datasets
    dataset.load_both_datasets()
    
    # Visualize samples
    dataset.visualize_samples()
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_dataset()
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

if __name__ == "__main__":
    main()
