# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will help you get the Handwritten Signature Verification System running quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- BHSig260 dataset (Bengali signatures)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Test the System

Run the test suite to verify everything is working:

```bash
python test_system.py
```

You should see output like:
```
HANDWRITTEN SIGNATURE VERIFICATION SYSTEM - TEST SUITE
============================================================
Testing imports...
✓ Config imported successfully
✓ Data preprocessing imported successfully
✓ Models imported successfully
✓ Trainer imported successfully
✓ Inference imported successfully

Testing configuration...
✓ Directory created: models
✓ Directory created: results
✓ Directory created: logs
✓ Configuration test passed

...

TEST RESULTS: 7/7 tests passed
============================================================
🎉 All tests passed! The system is ready to use.
```

## Step 3: Run the Complete Pipeline

Execute the full signature verification pipeline:

```bash
python main.py --mode full
```

This will:
1. ✅ Load and preprocess the BHSig260 dataset
2. ✅ Train all deep learning models (MobileNet, ResNet50, InceptionV3, VGG19, Ensemble, YOLOv5)
3. ✅ Evaluate model performance
4. ✅ Generate comprehensive reports
5. ✅ Demonstrate inference capabilities

## Step 4: Check Results

After completion, check the generated files:

```bash
# View results directory
ls results/

# View trained models
ls models/

# Read the comprehensive report
cat results/comprehensive_report.txt
```

## 🎯 Expected Output

### Training Progress
```
HANDWRITTEN SIGNATURE VERIFICATION SYSTEM
Based on 'Advancing Handwritten Signature Verification Through Deep Learning'
================================================================================

Setting up environment...
CUDA available: NVIDIA GeForce RTX 3080
Environment setup completed successfully!

====================================================================
DATA PREPROCESSING
====================================================================
Loading Bengali dataset from dataset/BHSig260-Bengali/BHSig260-Bengali
Loading Bengali signatures: 100%|██████████| 100/100 [00:30<00:00,  3.33it/s]
Loaded 5400 Bengali signatures
Total signatures loaded: 5400
Genuine signatures: 2400
Forged signatures: 3000

====================================================================
MODEL TRAINING
====================================================================
Training mobilenet model...
Epoch 1/100: Train Loss: 0.6931, Train Acc: 50.12%, Val Loss: 0.6892, Val Acc: 51.23%
...
```

### Final Results
```
TRAINING SUMMARY
----------------------------------------

MOBILENET:
  Accuracy: 0.9234
  Precision: 0.9156
  Recall: 0.9289
  F1-Score: 0.9221

RESNET50:
  Accuracy: 0.9456
  Precision: 0.9423
  Recall: 0.9489
  F1-Score: 0.9456

ENSEMBLE:
  Accuracy: 0.9678
  Precision: 0.9654
  Recall: 0.9701
  F1-Score: 0.9677

EXECUTION COMPLETED SUCCESSFULLY!
================================================================================
Results saved in: results
Models saved in: models
Logs saved in: logs
```

## 🔧 Customization

### Modify Configuration

Edit `config.py` to adjust settings:

```python
# Reduce training time for testing
EPOCHS = 10  # Default: 100
BATCH_SIZE = 16  # Default: 32

# Use smaller image size for faster processing
IMAGE_SIZE = (128, 128)  # Default: (224, 224)
```

### Run Individual Components

```bash
# Only preprocess data
python main.py --mode preprocess

# Only train models
python main.py --mode train

# Only evaluate models
python main.py --mode evaluate

# Only run inference demo
python main.py --mode inference
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or even 8
```

**2. Dataset Not Found**
```bash
# Check dataset structure
ls dataset/BHSig260-Bengali/BHSig260-Bengali/
# Should show directories: 1, 2, 3, ..., 100
```

**3. Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**4. Slow Training**
```bash
# Use smaller models for testing
# In config.py, reduce image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
```

## 📊 Understanding Results

### Model Performance Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Output Files

- `results/evaluation_summary.json`: Detailed performance metrics
- `results/*_confusion_matrix.png`: Visual classification results
- `results/*_training_history.png`: Training progress plots
- `models/*.pth`: Trained model weights
- `results/comprehensive_report.txt`: Complete system report

## 🎯 Next Steps

1. **Analyze Results**: Review the generated reports and visualizations
2. **Fine-tune Models**: Adjust hyperparameters in `config.py`
3. **Test Inference**: Use the trained models for signature verification
4. **Extend System**: Add new models or datasets

## 📞 Need Help?

1. Run the test suite: `python test_system.py`
2. Check the comprehensive README.md
3. Review error messages in the logs/ directory
4. Verify your dataset structure matches the expected format

---

**Happy Signature Verification! 🖋️✨**
