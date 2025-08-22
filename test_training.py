#!/usr/bin/env python3
"""
Simple test script to check if training works
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_preprocessing import BHSig260Dataset
from models import ModelFactory
from trainer import SignatureTrainer

def test_simple_training():
    """Test simple training with minimal data"""
    print("=" * 60)
    print("TESTING SIMPLE TRAINING")
    print("=" * 60)
    
    try:
        # Load config
        config = Config()
        config.create_directories()
        
        # Reduce training parameters for quick testing
        config.EPOCHS = 2  # Just 2 epochs for testing
        config.BATCH_SIZE = 8  # Small batch size
        
        print(f"Using config: EPOCHS={config.EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}")
        
        # Load minimal dataset
        dataset = BHSig260Dataset(config)
        dataset.load_both_datasets(max_samples=20)  # Just 20 samples
        
        if not dataset.data:
            print("ERROR: No data loaded")
            return False
        
        print(f"Loaded {len(dataset.data)} samples")
        
        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_dataset()
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Test model creation
        print("\nTesting model creation...")
        try:
            model = ModelFactory.create_model('mobilenet', config)
            print("✓ MobileNet model created successfully")
        except Exception as e:
            print(f"✗ MobileNet model creation failed: {e}")
            return False
        
        # Test training
        print("\nTesting training...")
        try:
            trainer = SignatureTrainer(config)
            
            # Prepare data loaders
            train_loader, val_loader = trainer.prepare_data_loaders(X_train, X_val, y_train, y_val)
            print(f"✓ Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}")
            
            # Train for just 1 epoch to test
            trained_model = trainer.train_model(model, train_loader, val_loader, 'mobilenet_test')
            print("✓ Training completed successfully")
            
            # Test evaluation
            test_loader, _ = trainer.prepare_data_loaders(X_test, X_test, y_test, y_test)
            results = trainer.evaluate_model(trained_model, test_loader, 'mobilenet_test')
            print(f"✓ Evaluation completed: Accuracy={results['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_training()
    if success:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED! Training system is working.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ TESTS FAILED! Training system has issues.")
        print("=" * 60)
