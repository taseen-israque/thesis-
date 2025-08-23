"""
Test Script for Handwritten Signature Verification System
Verifies that all components work correctly
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import Config
        print("✓ Config imported successfully")
        
        from data_preprocessing import BHSig260Dataset, SignaturePreprocessor
        print("✓ Data preprocessing imported successfully")
        
        from models import ModelFactory, YOLOv5SignatureDetector
        print("✓ Models imported successfully")
        
        from trainer import SignatureTrainer
        print("✓ Trainer imported successfully")
        
        from inference import SignatureVerifier, SignatureVerificationAPI
        print("✓ Inference imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        config = Config()
        config.create_directories()
        
        # Check if directories were created
        directories = [config.MODELS_DIR, config.RESULTS_DIR, config.LOGS_DIR]
        for directory in directories:
            if os.path.exists(directory):
                print(f"✓ Directory created: {directory}")
            else:
                print(f"✗ Directory not created: {directory}")
                return False
        
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessing...")
    
    try:
        from config import Config
        from data_preprocessing import SignaturePreprocessor
        
        config = Config()
        preprocessor = SignaturePreprocessor(config)
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test preprocessing
        processed_image = preprocessor.preprocess_image(dummy_image)
        
        if processed_image is not None:
            expected_shape = (*config.IMAGE_SIZE, config.CHANNELS)
            if processed_image.shape == expected_shape:
                print(f"✓ Image preprocessing successful: {processed_image.shape}")
            else:
                print(f"✗ Unexpected image shape: {processed_image.shape}, expected: {expected_shape}")
                return False
        else:
            print("✗ Image preprocessing failed")
            return False
        
        print("✓ Data preprocessing test passed")
        return True
    except Exception as e:
        print(f"✗ Data preprocessing error: {e}")
        return False

def test_models():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from config import Config
        from models import ModelFactory
        
        config = Config()
        
        # Test each model type
        model_types = ['mobilenet', 'resnet50', 'inceptionv3', 'vgg19', 'ensemble']
        
        for model_type in model_types:
            try:
                model = ModelFactory.create_model(model_type, config)
                
                # Test forward pass with dummy input
                dummy_input = torch.randn(1, config.CHANNELS, *config.IMAGE_SIZE)
                with torch.no_grad():
                    output = model(dummy_input)
                
                expected_output_shape = (1, config.NUM_CLASSES)
                if output.shape == expected_output_shape:
                    print(f"✓ {model_type} model created and tested successfully")
                else:
                    print(f"✗ {model_type} model output shape incorrect: {output.shape}")
                    return False
                    
            except Exception as e:
                print(f"✗ {model_type} model error: {e}")
                return False
        
        print("✓ Model creation test passed")
        return True
    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\nTesting dataset loading...")
    
    try:
        from config import Config
        from data_preprocessing import BHSig260Dataset
        
        config = Config()
        dataset = BHSig260Dataset(config)
        
        # Check if dataset path exists
        if os.path.exists(config.BENGALI_DATASET_PATH):
            print(f"✓ Bengali dataset path found: {config.BENGALI_DATASET_PATH}")
            
            # Try to load a small subset
            person_dirs = [d for d in os.listdir(config.BENGALI_DATASET_PATH) 
                          if os.path.isdir(os.path.join(config.BENGALI_DATASET_PATH, d))][:3]
            
            if person_dirs:
                print(f"✓ Found {len(person_dirs)} person directories")
                
                # Check if signature files exist
                sample_person_path = os.path.join(config.BENGALI_DATASET_PATH, person_dirs[0])
                signature_files = [f for f in os.listdir(sample_person_path) if f.endswith('.tif')]
                
                if signature_files:
                    print(f"✓ Found {len(signature_files)} signature files in sample directory")
                    print("✓ Dataset loading test passed")
                    return True
                else:
                    print("✗ No signature files found")
                    return False
            else:
                print("✗ No person directories found")
                return False
        else:
            print(f"✗ Bengali dataset path not found: {config.BENGALI_DATASET_PATH}")
            print("  This is expected if the dataset is not yet downloaded")
            return True  # Not a critical error for testing
        
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def test_training_setup():
    """Test training setup"""
    print("\nTesting training setup...")
    
    try:
        from config import Config
        from trainer import SignatureTrainer
        
        config = Config()
        trainer = SignatureTrainer(config)
        
        # Create dummy data
        X_train = np.random.rand(10, *config.IMAGE_SIZE, config.CHANNELS)
        X_val = np.random.rand(5, *config.IMAGE_SIZE, config.CHANNELS)
        y_train = np.random.randint(0, config.NUM_CLASSES, 10)
        y_val = np.random.randint(0, config.NUM_CLASSES, 5)
        
        # Test data loader creation
        train_loader, val_loader = trainer.prepare_data_loaders(X_train, X_val, y_train, y_val)
        
        if train_loader and val_loader:
            print(f"✓ Data loaders created successfully")
            print(f"  Training batches: {len(train_loader)}")
            print(f"  Validation batches: {len(val_loader)}")
            print("✓ Training setup test passed")
            return True
        else:
            print("✗ Data loader creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Training setup error: {e}")
        return False

def test_inference_setup():
    """Test inference setup"""
    print("\nTesting inference setup...")
    
    try:
        from config import Config
        from inference import SignatureVerificationAPI
        
        config = Config()
        
        # Test with empty model paths (should handle gracefully)
        model_paths = {}
        api = SignatureVerificationAPI(config, model_paths)
        
        available_models = api.get_available_models()
        print(f"✓ API initialized successfully")
        print(f"  Available models: {available_models}")
        
        print("✓ Inference setup test passed")
        return True
        
    except Exception as e:
        print(f"✗ Inference setup error: {e}")
        return False

def main():
    """Run all tests"""
    print("HANDWRITTEN SIGNATURE VERIFICATION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_data_preprocessing,
        test_models,
        test_dataset_loading,
        test_training_setup,
        test_inference_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Ensure your BHSig260 dataset is in the correct location")
        print("2. Run: python main.py --mode full")
        print("3. Check the results/ directory for outputs")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify dataset structure")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
