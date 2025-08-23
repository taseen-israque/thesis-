#!/usr/bin/env python3
"""
Streaming Trainer for Memory-Efficient Training
Trains models using batches of 50 images at a time
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, List
warnings.filterwarnings('ignore')

from config import Config
from models import ModelFactory
from streaming_dataset import StreamingBHSig260Dataset

class StreamingSignatureTrainer:
    """
    Memory-efficient trainer that processes images in batches
    """
    
    def __init__(self, config: Config, batch_size: int = 50):
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Initialize streaming dataset
        self.streaming_dataset = StreamingBHSig260Dataset(config, batch_size)
        
    def prepare_streaming_training(self, max_samples: int = None):
        """Prepare streaming dataset for training"""
        print("Preparing streaming dataset...")
        self.streaming_dataset.scan_dataset(max_samples)
        
        dataset_info = self.streaming_dataset.get_dataset_info()
        print(f"\nDataset Information:")
        print(f"  Total images: {dataset_info['total_images']}")
        print(f"  Total batches: {dataset_info['total_batches']}")
        print(f"  Batch size: {dataset_info['batch_size']}")
        print(f"  Genuine: {dataset_info['genuine_count']}")
        print(f"  Forged: {dataset_info['forged_count']}")
        
        # Split dataset into train/val/test batches
        self._split_dataset_batches()
        
    def _split_dataset_batches(self):
        """Split dataset into train/val/test batch indices"""
        total_batches = self.streaming_dataset.get_total_batches()
        
        # Calculate split indices
        train_end = int(total_batches * (1 - self.config.VALIDATION_SPLIT - self.config.TEST_SPLIT))
        val_end = int(total_batches * (1 - self.config.TEST_SPLIT))
        
        self.train_batch_indices = list(range(0, train_end))
        self.val_batch_indices = list(range(train_end, val_end))
        self.test_batch_indices = list(range(val_end, total_batches))
        
        print(f"\nBatch Split:")
        print(f"  Training batches: {len(self.train_batch_indices)} ({train_end})")
        print(f"  Validation batches: {len(self.val_batch_indices)} ({val_end - train_end})")
        print(f"  Test batches: {len(self.test_batch_indices)} ({total_batches - val_end})")
    
    def train_model_streaming(self, model: nn.Module, model_name: str) -> nn.Module:
        """Train model using streaming batches"""
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        print(f"\nTraining {model_name} with streaming batches...")
        print(f"Total epochs: {self.config.EPOCHS}")
        print(f"Training batches per epoch: {len(self.train_batch_indices)}")
        print(f"Validation batches per epoch: {len(self.val_batch_indices)}")
        
        for epoch in range(self.config.EPOCHS):
            print(f"\n{'='*20} Epoch {epoch+1}/{self.config.EPOCHS} {'='*20}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            print("Training phase...")
            for batch_idx in tqdm(self.train_batch_indices, desc="Training"):
                # Load batch of 50 images
                batch_images, batch_labels, _ = self.streaming_dataset.get_batch(batch_idx)
                
                if batch_images is None:
                    continue
                
                # Convert to tensors
                batch_tensor = torch.FloatTensor(batch_images).permute(0, 3, 1, 2)
                labels_tensor = torch.LongTensor(batch_labels)
                
                # Move to device
                batch_tensor = batch_tensor.to(self.device)
                labels_tensor = labels_tensor.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_tensor)
                loss = criterion(outputs, labels_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_tensor.size(0)
                train_correct += (predicted == labels_tensor).sum().item()
                train_loss += loss.item()
                
                # Clear batch from memory
                del batch_tensor, labels_tensor, outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            print("Validation phase...")
            with torch.no_grad():
                for batch_idx in tqdm(self.val_batch_indices, desc="Validation"):
                    # Load batch of 50 images
                    batch_images, batch_labels, _ = self.streaming_dataset.get_batch(batch_idx)
                    
                    if batch_images is None:
                        continue
                    
                    # Convert to tensors
                    batch_tensor = torch.FloatTensor(batch_images).permute(0, 3, 1, 2)
                    labels_tensor = torch.LongTensor(batch_labels)
                    
                    # Move to device
                    batch_tensor = batch_tensor.to(self.device)
                    labels_tensor = labels_tensor.to(self.device)
                    
                    # Forward pass
                    outputs = model(batch_tensor)
                    loss = criterion(outputs, labels_tensor)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_tensor.size(0)
                    val_correct += (predicted == labels_tensor).sum().item()
                    val_loss += loss.item()
                    
                    # Clear batch from memory
                    del batch_tensor, labels_tensor, outputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Calculate metrics
            avg_train_loss = train_loss / len(self.train_batch_indices)
            avg_val_loss = val_loss / len(self.val_batch_indices)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # Store history
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model(model, model_name, epoch, val_accuracy)
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return model
    
    def evaluate_model_streaming(self, model: nn.Module, model_name: str) -> Dict:
        """Evaluate model using streaming test batches"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        print(f"\nEvaluating {model_name} with streaming test batches...")
        
        with torch.no_grad():
            for batch_idx in tqdm(self.test_batch_indices, desc=f'Evaluating {model_name}'):
                # Load batch of 50 images
                batch_images, batch_labels, _ = self.streaming_dataset.get_batch(batch_idx)
                
                if batch_images is None:
                    continue
                
                # Convert to tensors
                batch_tensor = torch.FloatTensor(batch_images).permute(0, 3, 1, 2)
                labels_tensor = torch.LongTensor(batch_labels)
                
                # Move to device
                batch_tensor = batch_tensor.to(self.device)
                labels_tensor = labels_tensor.to(self.device)
                
                # Forward pass
                outputs = model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels_tensor.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Clear batch from memory
                del batch_tensor, labels_tensor, outputs, probabilities
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Classification report
        report = classification_report(all_targets, all_predictions, output_dict=True)
        
        # Save results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        # Save to file
        results_path = os.path.join(self.config.RESULTS_DIR, f'{model_name}_streaming_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, model_name)
        
        return results
    
    def save_model(self, model: nn.Module, model_name: str, epoch: int, accuracy: float) -> None:
        """Save the trained model"""
        model_path = os.path.join(self.config.MODELS_DIR, f'{model_name}_streaming_epoch_{epoch}_acc_{accuracy:.2f}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'config': self.config.__dict__
        }, model_path)
        print(f'Model saved: {model_path}')
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Genuine', 'Forged'],
                   yticklabels=['Genuine', 'Forged'])
        plt.title(f'Confusion Matrix - {model_name} (Streaming)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, f'{model_name}_streaming_confusion_matrix.png'))
        plt.close()
    
    def train_all_models_streaming(self) -> Dict:
        """Train all models using streaming batches"""
        all_results = {}
        
        # Model types to train
        model_types = ['mobilenet', 'resnet50', 'inceptionv3', 'vgg19', 'ensemble']
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} with streaming batches")
            print(f"{'='*60}")
            
            try:
                # Create model
                model = ModelFactory.create_model(model_type, self.config)
                
                # Train model with streaming
                trained_model = self.train_model_streaming(model, model_type)
                
                # Evaluate model with streaming
                results = self.evaluate_model_streaming(trained_model, model_type)
                all_results[model_type] = results
                
                # Reset training history for next model
                self.train_losses = []
                self.val_losses = []
                self.train_accuracies = []
                self.val_accuracies = []
                
                print(f"{model_type.upper()} training completed successfully!")
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
                continue
        
        return all_results

