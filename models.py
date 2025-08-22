"""
Deep Learning Models for Handwritten Signature Verification
Implements MobileNet, YOLOv5, and ensemble models (ResNet50, Inceptionv3, VGG19)
Based on the research paper "Advancing Handwritten Signature Verification Through Deep Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

from config import Config

class MobileNetSignatureVerifier(nn.Module):
    """
    MobileNet-based signature verification model
    Optimized for mobile and edge deployment
    """
    
    def __init__(self, config: Config):
        super(MobileNetSignatureVerifier, self).__init__()
        self.config = config
        
        # Load pretrained MobileNet
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Modify the first layer to accept grayscale images
        self.mobilenet.features[0][0] = nn.Conv2d(
            config.CHANNELS, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Remove the classifier and add custom layers
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=config.MOBILENET_DROPOUT),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_CLASSES)
        )
        
    def forward(self, x):
        return self.mobilenet(x)

class ResNet50SignatureVerifier(nn.Module):
    """
    ResNet50-based signature verification model
    """
    
    def __init__(self, config: Config):
        super(ResNet50SignatureVerifier, self).__init__()
        self.config = config
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(
            config.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Remove the classifier and add custom layers
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_CLASSES)
        )
        
    def forward(self, x):
        return self.resnet(x)

class InceptionV3SignatureVerifier(nn.Module):
    """
    InceptionV3-based signature verification model with transfer learning adaptation
    Uses pretrained ImageNet weights adapted for grayscale signature images
    """
    
    def __init__(self, config: Config):
        super(InceptionV3SignatureVerifier, self).__init__()
        self.config = config
        
        # Create new InceptionV3 model for grayscale input (no pretrained weights initially)
        self.inception = models.inception_v3(pretrained=False, aux_logits=True)
        
        # Load pretrained InceptionV3 model for RGB input
        pretrained_inception = models.inception_v3(pretrained=True, aux_logits=True)
        
        # Transfer weights with channel adaptation for grayscale
        self._transfer_weights_with_adaptation(pretrained_inception)
        
        # Set to evaluation mode to avoid batch normalization issues
        self.inception.eval()
        
        # Remove the classifier and add custom layers
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_CLASSES)
        )
        
    def _transfer_weights_with_adaptation(self, pretrained_model):
        """
        Transfer weights from pretrained RGB model to grayscale model
        with channel dimension adaptation
        """
        # First, modify the first convolutional layer to accept grayscale input
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(
            self.config.CHANNELS, 32, kernel_size=3, stride=2, bias=False
        )
        
        # Get the pretrained weights for the first layer
        pretrained_first_layer = pretrained_model.Conv2d_1a_3x3.conv
        old_weights = pretrained_first_layer.weight.data
        
        # Sum across RGB channels to create grayscale weights
        # old_weights shape: [32, 3, 3, 3] -> new_weights shape: [32, 1, 3, 3]
        new_weights = old_weights.sum(dim=1, keepdim=True)
        self.inception.Conv2d_1a_3x3.conv.weight.data = new_weights
        
        # Now transfer weights for all other layers
        new_layers = list(self.inception.modules())
        pretrained_layers = list(pretrained_model.modules())
        
        # Skip the first few layers as they've been handled
        for i in range(3, len(new_layers)):
            if hasattr(new_layers[i], 'weight') and hasattr(pretrained_layers[i], 'weight'):
                old_weights = pretrained_layers[i].weight.data
                
                # For other layers, transfer weights directly if shapes match
                if new_layers[i].weight.shape == old_weights.shape:
                    new_layers[i].weight.data = old_weights
                
                # Transfer bias if it exists
                if hasattr(new_layers[i], 'bias') and hasattr(pretrained_layers[i], 'bias'):
                    if new_layers[i].bias is not None and pretrained_layers[i].bias is not None:
                        new_layers[i].bias.data = pretrained_layers[i].bias.data
        
    def forward(self, x):
        # Ensure model is in eval mode for inference
        self.inception.eval()
        
        # Forward pass through InceptionV3 (returns tuple with aux_logits=True)
        with torch.no_grad():
            output = self.inception(x)
        
        if isinstance(output, tuple):
            return output[0]  # Return main output, ignore aux output
        return output

class VGG19SignatureVerifier(nn.Module):
    """
    VGG19-based signature verification model
    """
    
    def __init__(self, config: Config):
        super(VGG19SignatureVerifier, self).__init__()
        self.config = config
        
        # Load pretrained VGG19
        self.vgg = models.vgg19(pretrained=True)
        
        # Modify the first layer to accept grayscale images
        self.vgg.features[0] = nn.Conv2d(
            config.CHANNELS, 64, kernel_size=3, padding=1
        )
        
        # Remove the classifier and add custom layers
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_CLASSES)
        )
        
    def forward(self, x):
        return self.vgg(x)

class EnsembleSignatureVerifier(nn.Module):
    """
    Ensemble model combining ResNet50, InceptionV3, and VGG19
    """
    
    def __init__(self, config: Config):
        super(EnsembleSignatureVerifier, self).__init__()
        self.config = config
        
        # Initialize individual models
        self.resnet50 = ResNet50SignatureVerifier(config)
        self.inceptionv3 = InceptionV3SignatureVerifier(config)
        self.vgg19 = VGG19SignatureVerifier(config)
        
        # Ensemble weights
        self.weights = nn.Parameter(torch.tensor([
            config.ENSEMBLE_WEIGHTS['resnet50'],
            config.ENSEMBLE_WEIGHTS['inceptionv3'],
            config.ENSEMBLE_WEIGHTS['vgg19']
        ]))
        
        # Final classification layer
        self.final_classifier = nn.Sequential(
            nn.Linear(6, 64),  # 3 models * 2 classes = 6
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(64, config.NUM_CLASSES)
        )
        
    def forward(self, x):
        # Get predictions from each model
        resnet_out = self.resnet50(x)
        inception_out = self.inceptionv3(x)
        vgg_out = self.vgg19(x)
        
        # Apply softmax to get probabilities
        resnet_probs = F.softmax(resnet_out, dim=1)
        inception_probs = F.softmax(inception_out, dim=1)
        vgg_probs = F.softmax(vgg_out, dim=1)
        
        # Weighted ensemble
        weighted_resnet = resnet_probs * self.weights[0]
        weighted_inception = inception_probs * self.weights[1]
        weighted_vgg = vgg_probs * self.weights[2]
        
        # Concatenate weighted predictions
        ensemble_input = torch.cat([
            weighted_resnet, weighted_inception, weighted_vgg
        ], dim=1)
        
        # Final classification
        output = self.final_classifier(ensemble_input)
        
        return output

class YOLOv5SignatureDetector:
    """
    YOLOv5-based signature detection and classification
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        
    def create_custom_dataset(self, images: np.ndarray, labels: np.ndarray, 
                            output_dir: str) -> None:
        """
        Create YOLOv5-compatible dataset format
        
        Args:
            images: Input images
            labels: Corresponding labels
            output_dir: Directory to save the dataset
        """
        import os
        from PIL import Image
        
        # Create directories
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Save image
            img = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
            img_path = os.path.join(images_dir, f'image_{i:06d}.jpg')
            img.save(img_path)
            
            # Create YOLO label file
            label_path = os.path.join(labels_dir, f'image_{i:06d}.txt')
            with open(label_path, 'w') as f:
                # YOLO format: class_id center_x center_y width height
                # For signature verification, we use the entire image as one object
                f.write(f"{label} 0.5 0.5 1.0 1.0\n")
    
    def train(self, train_data_path: str, val_data_path: str) -> None:
        """
        Train YOLOv5 model on signature dataset
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
        """
        # Initialize YOLOv5 model
        self.model = YOLO('yolov5s.pt')
        
        # Train the model
        self.model.train(
            data={
                'train': train_data_path,
                'val': val_data_path,
                'nc': self.config.NUM_CLASSES,
                'names': ['genuine', 'forged']
            },
            epochs=self.config.YOLO_EPOCHS,
            imgsz=self.config.YOLO_IMG_SIZE,
            batch=self.config.YOLO_BATCH_SIZE,
            save=True,
            project=self.config.RESULTS_DIR,
            name='yolo_signature_detector'
        )
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict signature authenticity using YOLOv5
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predictions, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert image to PIL format
        from PIL import Image
        img = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
        
        # Make prediction
        results = self.model(img)
        
        # Extract predictions
        predictions = []
        confidences = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    predictions.append(cls)
                    confidences.append(conf)
        
        if predictions:
            return np.array(predictions), np.mean(confidences)
        else:
            return np.array([0]), 0.0

class ModelFactory:
    """
    Factory class for creating different signature verification models
    """
    
    @staticmethod
    def create_model(model_type: str, config: Config) -> nn.Module:
        """
        Create a signature verification model based on type
        
        Args:
            model_type: Type of model ('mobilenet', 'resnet50', 'inceptionv3', 'vgg19', 'ensemble')
            config: Configuration object
            
        Returns:
            Initialized model
        """
        if model_type.lower() == 'mobilenet':
            return MobileNetSignatureVerifier(config)
        elif model_type.lower() == 'resnet50':
            return ResNet50SignatureVerifier(config)
        elif model_type.lower() == 'inceptionv3':
            return InceptionV3SignatureVerifier(config)
        elif model_type.lower() == 'vgg19':
            return VGG19SignatureVerifier(config)
        elif model_type.lower() == 'ensemble':
            return EnsembleSignatureVerifier(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_summary(model: nn.Module) -> str:
        """
        Get a summary of the model architecture
        
        Args:
            model: PyTorch model
            
        Returns:
            Model summary as string
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = f"Model: {model.__class__.__name__}\n"
        summary += f"Total parameters: {total_params:,}\n"
        summary += f"Trainable parameters: {trainable_params:,}\n"
        summary += f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n"
        
        return summary

def test_models():
    """Test function to verify model creation and forward pass"""
    config = Config()
    
    # Test each model
    model_types = ['mobilenet', 'resnet50', 'inceptionv3', 'vgg19', 'ensemble']
    
    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} model:")
        print("-" * 50)
        
        try:
            # Create model
            model = ModelFactory.create_model(model_type, config)
            
            # Print model summary
            print(ModelFactory.get_model_summary(model))
            
            # Test forward pass
            batch_size = 2
            input_tensor = torch.randn(batch_size, config.CHANNELS, 
                                     config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
            
            with torch.no_grad():
                output = model(input_tensor)
            
            print(f"Input shape: {input_tensor.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output sample: {output[0]}")
            
        except Exception as e:
            print(f"Error testing {model_type}: {e}")
    
    # Test YOLOv5 detector
    print(f"\nTesting YOLOv5 Signature Detector:")
    print("-" * 50)
    try:
        detector = YOLOv5SignatureDetector(config)
        print("YOLOv5 detector created successfully")
    except Exception as e:
        print(f"Error creating YOLOv5 detector: {e}")

if __name__ == "__main__":
    test_models()
