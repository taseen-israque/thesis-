#!/usr/bin/env python3
"""
Streaming Dataset for Memory-Efficient Training
Loads images in batches of 50 at a time instead of loading all images into memory
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class StreamingBHSig260Dataset:
    """
    Memory-efficient dataset that loads images in batches
    """
    
    def __init__(self, config, batch_size: int = 50):
        self.config = config
        self.batch_size = batch_size
        self.image_paths = []
        self.labels = []
        self.metadata = []
        self.current_batch_idx = 0
        
    def scan_dataset(self, max_samples: int = None):
        """Scan dataset to get file paths without loading images"""
        print("Scanning dataset structure...")
        
        # Scan Bengali dataset
        if os.path.exists(self.config.BENGALI_DATASET_PATH):
            self._scan_language_dataset(self.config.BENGALI_DATASET_PATH, 'Bengali', max_samples)
        
        # Scan Hindi dataset
        if os.path.exists(self.config.HINDI_DATASET_PATH):
            self._scan_language_dataset(self.config.HINDI_DATASET_PATH, 'Hindi', max_samples)
        
        print(f"Found {len(self.image_paths)} total signature files")
        print(f"Genuine: {sum(1 for label in self.labels if label == 0)}")
        print(f"Forged: {sum(1 for label in self.labels if label == 1)}")
        
    def _scan_language_dataset(self, dataset_path: str, language: str, max_samples: int = None):
        """Scan a language dataset to collect file paths"""
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        sample_count = 0
        
        for person_dir in tqdm(person_dirs, desc=f"Scanning {language} dataset"):
            if max_samples and sample_count >= max_samples:
                break
                
            person_path = os.path.join(dataset_path, person_dir)
            signature_files = [f for f in os.listdir(person_path) if f.endswith('.tif')]
            
            for filename in signature_files:
                if max_samples and sample_count >= max_samples:
                    break
                    
                file_path = os.path.join(person_path, filename)
                
                # Parse filename to get metadata
                metadata = self._parse_filename(filename)
                metadata['person_id'] = person_dir
                metadata['language'] = language
                metadata['file_path'] = file_path
                
                # Determine label (0 for genuine, 1 for forged)
                label = 1 if metadata.get('category') == 'F' else 0
                
                # Store file path and metadata (not the actual image)
                self.image_paths.append(file_path)
                self.labels.append(label)
                self.metadata.append(metadata)
                sample_count += 1
    
    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse BHSig260 filename to extract metadata"""
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
    
    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load a batch of images (50 at a time)"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        
        if start_idx >= len(self.image_paths):
            return None, None, None
        
        batch_images = []
        batch_labels = []
        batch_metadata = []
        
        print(f"Loading batch {batch_idx + 1}: images {start_idx + 1}-{end_idx}")
        
        for i in range(start_idx, end_idx):
            try:
                # Load image from disk
                image = self._load_and_preprocess_image(self.image_paths[i])
                if image is not None:
                    batch_images.append(image)
                    batch_labels.append(self.labels[i])
                    batch_metadata.append(self.metadata[i])
            except Exception as e:
                print(f"Error loading image {self.image_paths[i]}: {e}")
                continue
        
        if not batch_images:
            return None, None, None
        
        return np.array(batch_images), np.array(batch_labels), batch_metadata
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # Remove noise
            image = self._remove_noise(image)
            
            # Resize
            image = cv2.resize(image, self.config.IMAGE_SIZE)
            
            # Normalize
            image = image.astype(np.float32) / self.config.NORMALIZATION_FACTOR
            
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return denoised
    
    def get_total_batches(self) -> int:
        """Get total number of batches"""
        return (len(self.image_paths) + self.batch_size - 1) // self.batch_size
    
    def get_dataset_info(self) -> Dict:
        """Get dataset information"""
        return {
            'total_images': len(self.image_paths),
            'total_batches': self.get_total_batches(),
            'batch_size': self.batch_size,
            'genuine_count': sum(1 for label in self.labels if label == 0),
            'forged_count': sum(1 for label in self.labels if label == 1)
        }
    
    def reset_batch_counter(self):
        """Reset batch counter for new epoch"""
        self.current_batch_idx = 0
    
    def get_next_batch(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Get next batch and increment counter"""
        batch = self.get_batch(self.current_batch_idx)
        self.current_batch_idx += 1
        return batch



