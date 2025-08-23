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
    
    def train_all_models(self, train_loader, val_loader, test_loader, y_train, y_val, y_test):
        """
        Train all signature verification models
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            y_train: Training labels (for evaluation)
            y_val: Validation labels (for evaluation)
            y_test: Test labels (for evaluation)
            
        Returns:
            Dictionary with training results for each model
        """
        results = {}
        
        # Model factory uses static methods
        
        # Train each model
        for model_name in ['mobilenet', 'resnet50', 'vgg19', 'inceptionv3']:
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()} model")
            print(f"{'='*50}")
            
            # Create model using static method
            model = ModelFactory.create_model(model_name, self.config)
            
            # Train model using data loaders
            trained_model = self.train_model_with_loaders(model, train_loader, val_loader, model_name)
            
            # Evaluate on test set
            evaluation_results = self.evaluate_model(
                trained_model, test_loader, model_name
            )
            
            test_accuracy = evaluation_results['accuracy']
            test_precision = evaluation_results['precision']
            test_recall = evaluation_results['recall']
            test_f1 = evaluation_results['f1_score']
            
            # Store results
            results[model_name] = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'model': trained_model
            }
            
            print(f"{model_name.upper()} Results:")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test Precision: {test_precision:.4f}")
            print(f"  Test Recall: {test_recall:.4f}")
            print(f"  Test F1-Score: {test_f1:.4f}")
        
        return results
    
    def train_model_with_loaders(self, model: nn.Module, train_loader, 
                                val_loader, model_name: str) -> nn.Module:
        """
        Train a model using data loaders
        
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
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS}')):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Store history
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Print progress
            print(f'Epoch {epoch+1}/{self.config.EPOCHS}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(self.config.MODELS_DIR, f'{model_name}_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f'{model_name}_best.pth')))
        
        return model
    
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