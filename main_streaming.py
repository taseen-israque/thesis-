#!/usr/bin/env python3
"""
Streaming Main Script for Memory-Efficient Training
Uses batches of 50 images at a time instead of loading all images into memory
"""

import os
import sys
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from streaming_dataset import StreamingBHSig260Dataset
from streaming_trainer import StreamingSignatureTrainer

def setup_environment():
    """Setup environment and check dependencies"""
    print("Setting up environment...")
    
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")
    
    print("Environment setup completed successfully!")
    return True

def run_streaming_training(config: Config, batch_size: int = 50, max_samples: int = None):
    """Run streaming training pipeline"""
    print("\n" + "="*60)
    print("STREAMING TRAINING PIPELINE")
    print("="*60)
    print(f"Batch size: {batch_size} images per batch")
    print(f"Memory efficient: Only {batch_size} images in RAM at once")
    print("="*60)
    
    # Initialize streaming trainer
    trainer = StreamingSignatureTrainer(config, batch_size)
    
    # Prepare streaming dataset
    trainer.prepare_streaming_training(max_samples)
    
    # Train all models with streaming
    print("\nStarting streaming training...")
    results = trainer.train_all_models_streaming()
    
    return results

def display_streaming_results(results: dict, config: Config):
    """Display streaming training results"""
    print("\n" + "="*60)
    print("STREAMING TRAINING RESULTS")
    print("="*60)
    
    if not results:
        print("No results to display.")
        return
    
    # Create results summary
    summary_data = []
    for model_name, result in results.items():
        if 'accuracy' in result:
            metrics = result
            summary_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
    
    # Display table
    if summary_data:
        print("\nModel Performance Comparison (Streaming Training):")
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
        if 'accuracy' in result:
            accuracy = result['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
    
    if best_model:
        print(f"\n🏆 Best performing model: {best_model.upper()} (Accuracy: {best_accuracy:.4f})")
    
    # Save results
    results_file = os.path.join(config.RESULTS_DIR, f"streaming_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_file, 'w') as f:
        f.write("Streaming Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Training method: Streaming (50 images per batch)\n")
        f.write(f"Memory efficient: Yes\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for data in summary_data:
            f.write(f"{data['Model']}: Acc={data['Accuracy']}, F1={data['F1-Score']}\n")
        
        if best_model:
            f.write(f"\nBest model: {best_model.upper()} (Accuracy: {best_accuracy:.4f})\n")
    
    print(f"\nResults saved to: {results_file}")

def main():
    """Main execution function for streaming training"""
    parser = argparse.ArgumentParser(description='Streaming Handwritten Signature Verification System')
    parser.add_argument('--batch-size', type=int, default=50, 
                       help='Number of images to load per batch (default: 50)')
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    
    args = parser.parse_args()
    
    print("🚀 STREAMING HANDWRITTEN SIGNATURE VERIFICATION SYSTEM")
    print("Based on 'Advancing Handwritten Signature Verification Through Deep Learning'")
    print("=" * 80)
    print("\n📊 STREAMING TRAINING FEATURES:")
    print("   • Loads images in batches of 50 at a time")
    print("   • Memory efficient: Only 50 images in RAM")
    print("   • Scalable to any dataset size")
    print("   • Real-time image processing")
    print("   • Automatic memory cleanup")
    print("=" * 80)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Load configuration
    config = Config()
    config.create_directories()
    
    # Set random seed for reproducibility
    import torch
    import numpy as np
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Run streaming training
    print(f"\nStarting streaming training with batch size: {args.batch_size}")
    results = run_streaming_training(config, args.batch_size, args.max_samples)
    
    # Display results
    display_streaming_results(results, config)
    
    print("\n" + "="*80)
    print("🎉 STREAMING TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved in: {config.RESULTS_DIR}")
    print(f"Models saved in: {config.MODELS_DIR}")
    print(f"Memory efficient training completed!")
    print("="*80)

if __name__ == "__main__":
    main()



