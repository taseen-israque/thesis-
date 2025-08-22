# Handwritten Signature Verification System

A comprehensive offline handwritten signature verification system based on the research paper **"Advancing Handwritten Signature Verification Through Deep Learning"** (IJETT, 2024). This system implements multiple deep learning models for high-precision signature classification and forgery detection.

## 🎯 Overview

This system provides a complete pipeline for handwritten signature verification using:

- **MobileNet**: Efficient signature classification optimized for mobile/edge deployment
- **YOLOv5**: Signature detection and classification
- **Ensemble Model**: Combines ResNet50, InceptionV3, and VGG19 for high-precision verification
- **BHSig260 Dataset**: Bengali and Hindi signatures for training and evaluation

## 🚀 Features

- **Complete Data Preprocessing**: Grayscale conversion, normalization, resizing, and augmentation
- **Multiple Model Architectures**: MobileNet, ResNet50, InceptionV3, VGG19, and ensemble
- **YOLOv5 Integration**: Object detection for signature localization
- **Comprehensive Training**: Training loops with validation, early stopping, and model saving
- **Advanced Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices
- **Inference API**: Easy-to-use interface for signature verification
- **Visualization**: Training history, confusion matrices, and verification results
- **Batch Processing**: Support for processing multiple signatures simultaneously

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space

### Dependencies
Install all required packages using:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
thesis-/
├── dataset/
│   └── BHSig260-Bengali/          # Bengali signature dataset
│       └── BHSig260-Bengali/
│           ├── 1/                 # Person 1 signatures
│           ├── 2/                 # Person 2 signatures
│           └── ...                # More persons
├── config.py                      # Configuration settings
├── data_preprocessing.py          # Data preprocessing module
├── models.py                      # Deep learning models
├── trainer.py                     # Training and evaluation
├── inference.py                   # Inference and verification
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd thesis-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py --mode preprocess
   ```

## 📊 Dataset

The system is designed to work with the **BHSig260 dataset** containing:
- **Bengali signatures**: 100 persons × 24 genuine + 30 forged signatures
- **Hindi signatures**: 160 persons × 24 genuine + 30 forged signatures

### Dataset Structure
```
BHSig260-Bengali/
└── BHSig260-Bengali/
    ├── 1/
    │   ├── B-S-1-G-01.tif    # Genuine signature
    │   ├── B-S-1-G-02.tif    # Genuine signature
    │   ├── B-S-1-F-01.tif    # Forged signature
    │   └── ...
    ├── 2/
    └── ...
```

### Filename Format
- `B-S-{person_id}-{G/F}-{sample_id}.tif`
- `B`: Bengali, `H`: Hindi
- `S`: Signature
- `G`: Genuine, `F`: Forged

## 🚀 Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py --mode full
```

This will:
1. Preprocess the BHSig260 dataset
2. Train all models (MobileNet, ResNet50, InceptionV3, VGG19, Ensemble, YOLOv5)
3. Evaluate model performance
4. Generate comprehensive reports
5. Demonstrate inference capabilities

### Step-by-Step Execution

#### 1. Data Preprocessing Only
```bash
python main.py --mode preprocess
```

#### 2. Model Training Only
```bash
python main.py --mode train
```

#### 3. Model Evaluation Only
```bash
python main.py --mode evaluate
```

#### 4. Inference Demonstration Only
```bash
python main.py --mode inference
```

### Custom Configuration

Modify `config.py` to adjust:
- Image size and preprocessing parameters
- Training hyperparameters
- Model architectures
- File paths and directories

## 🔧 Configuration

Key configuration parameters in `config.py`:

```python
# Image preprocessing
IMAGE_SIZE = (224, 224)          # Standard size for deep learning models
CHANNELS = 1                     # Grayscale images
NORMALIZATION_FACTOR = 255.0     # Pixel value normalization

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model parameters
NUM_CLASSES = 2                  # Genuine vs Forged
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-4

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'resnet50': 0.4,
    'inceptionv3': 0.3,
    'vgg19': 0.3
}
```

## 📈 Model Performance

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🔍 Inference API

### Single Signature Verification

```python
from inference import SignatureVerificationAPI
from config import Config

# Initialize API
config = Config()
model_paths = {
    'ensemble': 'models/ensemble_signature_verifier.pth'
}
api = SignatureVerificationAPI(config, model_paths)

# Verify signature
result = api.verify('path/to/signature.tif', method='ensemble')
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Batch Verification

```python
# Verify multiple signatures
image_paths = ['sig1.tif', 'sig2.tif', 'sig3.tif']
results = api.batch_verify(image_paths, method='ensemble')

for i, result in enumerate(results):
    print(f"Signature {i+1}: {result['verdict']} ({result['confidence']:.3f})")
```

### Signature Comparison

```python
# Compare two signatures
comparison = api.compare('signature1.tif', 'signature2.tif', method='ensemble')
print(f"Similarity Score: {comparison['similarity_score']:.3f}")
print(f"Same Person: {comparison['same_person']}")
```

## 📊 Output Files

After execution, the following files are generated:

### Models Directory (`models/`)
- `mobilenet_signature_verifier.pth`
- `resnet50_signature_verifier.pth`
- `inceptionv3_signature_verifier.pth`
- `vgg19_signature_verifier.pth`
- `ensemble_signature_verifier.pth`
- `yolo_signature_detector.pt`

### Results Directory (`results/`)
- `evaluation_summary.json` - Comprehensive evaluation results
- `comprehensive_report.txt` - Detailed system report
- `*_confusion_matrix.png` - Confusion matrix visualizations
- `*_training_history.png` - Training history plots
- `verification_results.png` - Verification result visualizations

### Logs Directory (`logs/`)
- Training logs and error messages

## 🎯 Research Paper Implementation

This system implements the methodology described in **"Advancing Handwritten Signature Verification Through Deep Learning"** (IJETT, 2024):

### Key Contributions
1. **Multi-Model Ensemble**: Combines ResNet50, InceptionV3, and VGG19 for robust verification
2. **MobileNet Integration**: Lightweight model for edge deployment
3. **YOLOv5 Detection**: Object detection for signature localization
4. **Advanced Preprocessing**: Comprehensive data augmentation and noise removal
5. **BHSig260 Dataset**: Evaluation on Bengali and Hindi signatures

### Technical Approach
- **Data Preprocessing**: Grayscale conversion, normalization, resizing (224×224), augmentation
- **Model Architecture**: Transfer learning with custom classification heads
- **Training Strategy**: Adam optimizer, learning rate scheduling, early stopping
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Ensemble Method**: Weighted averaging of model predictions

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.py`
   - Use smaller image size
   - Train models sequentially

2. **Dataset Not Found**
   - Verify dataset path in `config.py`
   - Check dataset structure matches expected format

3. **Missing Dependencies**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+)

4. **Model Loading Errors**
   - Ensure models are trained before inference
   - Check model file paths

### Performance Optimization

1. **GPU Acceleration**
   - Use CUDA-compatible GPU
   - Install CUDA toolkit and cuDNN

2. **Memory Management**
   - Adjust batch size based on available memory
   - Use gradient checkpointing for large models

3. **Training Speed**
   - Use multiple GPU training
   - Enable mixed precision training

## 📚 References

1. **Research Paper**: "Advancing Handwritten Signature Verification Through Deep Learning" (IJETT, 2024)
2. **Dataset**: BHSig260 - Bengali and Hindi Signature Dataset
3. **Models**: 
   - MobileNet: Howard et al. (2017)
   - ResNet: He et al. (2016)
   - Inception: Szegedy et al. (2016)
   - VGG: Simonyan & Zisserman (2014)
   - YOLOv5: Jocher et al. (2020)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- Create an issue in the repository
- Contact the development team
- Refer to the comprehensive documentation

---

**Note**: This system is designed for research and educational purposes. For production use, additional security measures and validation should be implemented.