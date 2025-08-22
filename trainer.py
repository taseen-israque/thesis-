"""
Training Module for Handwritten Signature Verification Models
Handles training, validation, and evaluation of all deep learning models
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
from typing import Tuple, Dict
warnings.filterwarnings('ignore')

from config import Config
from models import ModelFactory, YOLOv5SignatureDetector
from data_preprocessing import BHSig260Dataset

class SignatureTrainer:
    """
    Trainer class for signature verification models
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data_loaders(self, X_train: np.ndarray, X_val: np.ndarray, 
                           y_train: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare PyTorch data loaders for training
        
        Args:
            X_train: Training images
            X_val: Validation images
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # (N, C, H, W)
        X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        return train_loader, val_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> nn.Module:
        """
        Train a signature verification model
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model for saving
            
        Returns:
            Trained model
        """
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
        
        print(f"\nTraining {model_name}...")
        print(f"Total epochs: {self.config.EPOCHS}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS} - Training')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS} - Validation')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100 * val_correct / val_total:.2f}%'
                    })
            
            # Calculate average losses and accuracies
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
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
    
    def save_model(self, model: nn.Module, model_name: str, epoch: int, accuracy: float) -> None:
        """
        Save the trained model
        
        Args:
            model: Trained model
            model_name: Name of the model
            epoch: Current epoch
            accuracy: Validation accuracy
        """
        model_path = os.path.join(self.config.MODELS_DIR, f'{model_name}_epoch_{epoch}_acc_{accuracy:.2f}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'config': self.config.__dict__
        }, model_path)
        print(f'Model saved: {model_path}')
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
        """
        Evaluate the trained model on test set
        
        Args:
            model: Trained model
            test_loader: Test data loader
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f'Evaluating {model_name}'):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
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
        results_path = os.path.join(self.config.RESULTS_DIR, f'{model_name}_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, model_name)
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Genuine', 'Forged'],
                   yticklabels=['Genuine', 'Forged'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
        plt.close()
    
    def plot_training_history(self, model_name: str) -> None:
        """
        Plot and save training history
        
        Args:
            model_name: Name of the model
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title(f'Training and Validation Loss - {model_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title(f'Training and Validation Accuracy - {model_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, f'{model_name}_training_history.png'))
        plt.close()
    
    def train_all_models(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train all signature verification models
        
        Args:
            X_train, X_val, X_test: Training, validation, and test images
            y_train, y_val, y_test: Training, validation, and test labels
            
        Returns:
            Dictionary with all evaluation results
        """
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(X_train, X_val, y_train, y_val)
        test_loader, _ = self.prepare_data_loaders(X_test, X_test, y_test, y_test)
        
        # Model types to train
        model_types = ['mobilenet', 'resnet50', 'inceptionv3', 'vgg19', 'ensemble']
        all_results = {}
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*60}")
            
            try:
                # Create model
                model = ModelFactory.create_model(model_type, self.config)
                
                # Train model
                trained_model = self.train_model(model, train_loader, val_loader, model_type)
                
                # Evaluate model
                results = self.evaluate_model(trained_model, test_loader, model_type)
                all_results[model_type] = results
                
                # Plot training history
                self.plot_training_history(model_type)
                
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

class YOLOv5Trainer:
    """
    Trainer class for YOLOv5 signature detection model
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = YOLOv5SignatureDetector(config)
    
    def train_yolo_model(self, X_train: np.ndarray, X_val: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train YOLOv5 model for signature detection
        
        Args:
            X_train, X_val: Training and validation images
            y_train, y_val: Training and validation labels
        """
        print("\nTraining YOLOv5 Signature Detector...")
        
        # Create YOLOv5-compatible datasets
        train_data_path = os.path.join(self.config.RESULTS_DIR, 'yolo_train_data')
        val_data_path = os.path.join(self.config.RESULTS_DIR, 'yolo_val_data')
        
        self.detector.create_custom_dataset(X_train, y_train, train_data_path)
        self.detector.create_custom_dataset(X_val, y_val, val_data_path)
        
        # Train the model
        self.detector.train(train_data_path, val_data_path)
        
        print("YOLOv5 training completed!")

def main():
    """Main function to run complete training pipeline"""
    config = Config()
    config.create_directories()
    
    # Load and preprocess dataset
    print("Loading BHSig260 dataset...")
    dataset = BHSig260Dataset(config)
    dataset.load_both_datasets()
    
    if not dataset.data:
        print("No data loaded. Please check dataset paths.")
        return
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_dataset()
    
    # Create augmented training data
    print("Creating augmented training dataset...")
    X_train_aug, y_train_aug = dataset.create_augmented_dataset(X_train, y_train, augmentation_factor=2)
    
    # Train all models
    trainer = SignatureTrainer(config)
    results = trainer.train_all_models(X_train_aug, X_val, X_test, y_train_aug, y_val, y_test)
    
    # Train YOLOv5 model
    yolo_trainer = YOLOv5Trainer(config)
    yolo_trainer.train_yolo_model(X_train_aug, X_val, y_train_aug, y_val)
    
    # Print summary of results
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1-Score: {result['f1_score']:.4f}")
    
    print(f"\nAll models trained and evaluated successfully!")
    print(f"Results saved in: {config.RESULTS_DIR}")
    print(f"Models saved in: {config.MODELS_DIR}")

if __name__ == "__main__":
    main()
