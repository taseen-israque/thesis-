#!/usr/bin/env python3
"""
Small Dataset Experiment Runner
Runs the signature verification system on subjects 1-5 only
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_small import Config
from data_preprocessing import BHSig260Dataset, SignaturePreprocessor
from models import ModelFactory
from trainer import SignatureTrainer
from inference import SignatureVerifier

def setup_environment():
    """Setup the environment and check dependencies"""
    print("=" * 80)
    print("SMALL DATASET SIGNATURE VERIFICATION EXPERIMENT")
    print("Using subjects 1-5 from BHSig260 dataset")
    print("=" * 80)
    
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    return device

def load_and_preprocess_small_dataset(config):
    """Load and preprocess the small dataset (subjects 1-5)"""
    print("\n" + "="*50)
    print("LOADING AND PREPROCESSING SMALL DATASET")
    print("="*50)
    
    # Initialize dataset and preprocessor
    dataset = BHSig260Dataset(config)
    preprocessor = SignaturePreprocessor(config)
    
    print("Loading dataset...")
    # Load both Bengali and Hindi datasets with limited samples for subjects 1-5
    dataset.load_both_datasets(max_samples=50)  # Load more samples to ensure we get subjects 1-5
    
    if not dataset.data:
        print("ERROR: No data loaded. Please check dataset paths.")
        return None, None, None, None, None, None
    
    print(f"Original dataset size: {len(dataset.data)} samples")
    print(f"Classes distribution: {np.bincount(dataset.labels)}")
    
    # Filter for subjects 1-5 only
    print("\nFiltering for subjects 1-5...")
    filtered_indices = []
    for i, meta in enumerate(dataset.metadata):
        # Extract person_id from metadata (could be from filename or directory)
        person_id = meta.get('person_id', '0')
        try:
            subject_id = int(person_id)
            if subject_id in config.SMALL_DATASET_SUBJECTS:
                filtered_indices.append(i)
        except (ValueError, TypeError):
            # Try to extract from directory name if person_id is not a number
            try:
                # If person_id is like 'Person001', extract the number
                subject_id = int(''.join(filter(str.isdigit, person_id)))
                if subject_id in config.SMALL_DATASET_SUBJECTS:
                    filtered_indices.append(i)
            except:
                continue
    
    if not filtered_indices:
        print("ERROR: No samples found for subjects 1-5")
        print("Available person_ids:", set(meta.get('person_id', 'unknown') for meta in dataset.metadata[:10]))
        return None, None, None, None, None, None
    
    # Apply filtering
    X_filtered = [dataset.data[i] for i in filtered_indices]
    y_filtered = [dataset.labels[i] for i in filtered_indices]
    metadata_filtered = [dataset.metadata[i] for i in filtered_indices]
    
    print(f"Filtered dataset size: {len(X_filtered)} samples")
    # Extract subject IDs for display
    subject_ids = []
    for meta in metadata_filtered:
        person_id = meta.get('person_id', '0')
        try:
            subject_ids.append(int(person_id))
        except:
            try:
                subject_ids.append(int(''.join(filter(str.isdigit, person_id))))
            except:
                pass
    print(f"Subjects included: {sorted(set(subject_ids)) if subject_ids else 'Could not determine'}")
    
    # Update dataset with filtered data
    dataset.data = X_filtered
    dataset.labels = y_filtered
    dataset.metadata = metadata_filtered
    
    # Split the data
    print("\nSplitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_dataset()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(config, X_train, X_val, X_test, y_train, y_val, y_test, device):
    """Train all models on the small dataset"""
    print("\n" + "="*50)
    print("TRAINING MODELS ON SMALL DATASET")
    print("="*50)
    
    results = {}
    trainer = SignatureTrainer(config)
    
    # Models to train (excluding YOLOv5 for simplicity in small experiment)
    models_to_train = ['mobilenet', 'resnet50', 'inceptionv3', 'vgg19', 'ensemble']
    
    for model_name in models_to_train:
        print(f"\n{'='*20} Training {model_name.upper()} {'='*20}")
        
        try:
            # Create model using static method
            model = ModelFactory.create_model(model_name, config)
            if model is None:
                print(f"Failed to create {model_name} model")
                continue
            
            # Prepare data loaders
            train_loader, val_loader = trainer.prepare_data_loaders(X_train, X_val, y_train, y_val)
            
            # Train model
            model = trainer.train_model(model, train_loader, val_loader, model_name)
            
            # Evaluate model
            test_metrics = trainer.evaluate_model(model, X_test, y_test)
            
            # Store results
            results[model_name] = {
                'model': model,
                'test_metrics': test_metrics
            }
            
            print(f"{model_name} - Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"{model_name} - Test F1-Score: {test_metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def display_results(results, config):
    """Display comprehensive results"""
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    if not results:
        print("No results to display.")
        return
    
    # Create results summary
    summary_data = []
    for model_name, result in results.items():
        if 'test_metrics' in result:
            metrics = result['test_metrics']
            summary_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
    
    # Display table
    if summary_data:
        print("\nModel Performance Comparison:")
        print("-" * 70)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        for data in summary_data:
            print(f"{data['Model']:<15} {data['Accuracy']:<10} {data['Precision']:<12} {data['Recall']:<10} {data['F1-Score']:<10}")
        print("-" * 70)
    
    # Find best model
    best_model = None
    best_accuracy = 0
    for model_name, result in results.items():
        if 'test_metrics' in result:
            accuracy = result['test_metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
    
    if best_model:
        print(f"\nBest performing model: {best_model.upper()} (Accuracy: {best_accuracy:.4f})")
    
    # Save results
    results_file = config.get_results_path(f"small_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_file, 'w') as f:
        f.write("Small Dataset Experiment Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: BHSig260 (subjects 1-5)\n")
        f.write(f"Total samples processed: Limited subset\n\n")
        
        for data in summary_data:
            f.write(f"{data['Model']}: Acc={data['Accuracy']}, F1={data['F1-Score']}\n")
        
        if best_model:
            f.write(f"\nBest model: {best_model.upper()} (Accuracy: {best_accuracy:.4f})\n")
    
    print(f"\nResults saved to: {results_file}")

def main():
    """Main function to run the small dataset experiment"""
    try:
        # Setup
        device = setup_environment()
        config = Config()
        
        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_small_dataset(config)
        
        if X_train is None:
            print("Failed to load dataset. Exiting.")
            return
        
        # Train models
        results = train_models(config, X_train, X_val, X_test, y_train, y_val, y_test, device)
        
        # Display results
        display_results(results, config)
        
        print("\n" + "="*50)
        print("SMALL DATASET EXPERIMENT COMPLETED!")
        print("="*50)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
