"""
Inference Module for Handwritten Signature Verification
Provides prediction capabilities for trained models and complete signature verification system
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import json
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config
from models import ModelFactory, YOLOv5SignatureDetector
from data_preprocessing import SignaturePreprocessor

class SignatureVerifier:
    """
    Complete signature verification system using trained models
    """
    
    def __init__(self, config: Config, model_paths: Dict[str, str]):
        """
        Initialize signature verifier with trained models
        
        Args:
            config: Configuration object
            model_paths: Dictionary mapping model names to their saved paths
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = SignaturePreprocessor(config)
        
        # Load trained models
        self.models = {}
        self.load_models(model_paths)
        
        print(f"Signature verifier initialized on {self.device}")
    
    def load_models(self, model_paths: Dict[str, str]) -> None:
        """
        Load trained models from saved paths
        
        Args:
            model_paths: Dictionary mapping model names to their saved paths
        """
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    # Create model architecture
                    model = ModelFactory.create_model(model_name, self.config)
                    
                    # Load trained weights
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    
                    self.models[model_name] = model
                    print(f"Loaded {model_name} model from {model_path}")
                    
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
            else:
                print(f"Model path not found: {model_path}")
    
    def preprocess_signature(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a signature image for prediction
        
        Args:
            image_path: Path to the signature image
            
        Returns:
            Preprocessed image or None if error
        """
        try:
            # Load image
            image = self.preprocessor.load_image(image_path)
            if image is None:
                return None
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess_image(image)
            return processed_image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_single_model(self, model_name: str, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction using a single model
        
        Args:
            model_name: Name of the model to use
            image: Preprocessed image
            
        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).permute(0, 3, 1, 2)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def predict_ensemble(self, image: np.ndarray) -> Tuple[int, float, Dict[str, np.ndarray]]:
        """
        Make prediction using ensemble of models
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple of (ensemble_prediction, ensemble_confidence, individual_predictions)
        """
        individual_predictions = {}
        all_probabilities = []
        
        # Get predictions from all models
        for model_name in self.models.keys():
            if model_name != 'ensemble':  # Skip ensemble model itself
                pred, conf, probs = self.predict_single_model(model_name, image)
                individual_predictions[model_name] = {
                    'prediction': pred,
                    'confidence': conf,
                    'probabilities': probs
                }
                all_probabilities.append(probs)
        
        if not all_probabilities:
            raise ValueError("No models available for ensemble prediction")
        
        # Calculate ensemble prediction (weighted average)
        ensemble_probs = np.mean(all_probabilities, axis=0)
        ensemble_prediction = np.argmax(ensemble_probs)
        ensemble_confidence = np.max(ensemble_probs)
        
        return ensemble_prediction, ensemble_confidence, individual_predictions
    
    def verify_signature(self, image_path: str, method: str = 'ensemble') -> Dict:
        """
        Verify signature authenticity
        
        Args:
            image_path: Path to the signature image
            method: Prediction method ('single_model' or 'ensemble')
            
        Returns:
            Dictionary with verification results
        """
        # Preprocess image
        processed_image = self.preprocess_signature(image_path)
        if processed_image is None:
            return {'error': 'Failed to preprocess image'}
        
        results = {
            'image_path': image_path,
            'method': method,
            'prediction': None,
            'confidence': None,
            'is_genuine': None,
            'individual_predictions': {}
        }
        
        try:
            if method == 'ensemble':
                # Use ensemble prediction
                pred, conf, individual_preds = self.predict_ensemble(processed_image)
                results['prediction'] = pred
                results['confidence'] = conf
                results['individual_predictions'] = individual_preds
                
            else:
                # Use single model
                if method not in self.models:
                    return {'error': f'Model {method} not available'}
                
                pred, conf, probs = self.predict_single_model(method, processed_image)
                results['prediction'] = pred
                results['confidence'] = conf
                results['probabilities'] = probs.tolist()
            
            # Determine if signature is genuine (0) or forged (1)
            results['is_genuine'] = (results['prediction'] == 0)
            results['verdict'] = 'GENUINE' if results['is_genuine'] else 'FORGED'
            
            # Apply confidence threshold
            if results['confidence'] < self.config.VERIFICATION_THRESHOLD:
                results['verdict'] = 'UNCERTAIN'
                results['warning'] = 'Low confidence prediction'
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def batch_verify(self, image_paths: List[str], method: str = 'ensemble') -> List[Dict]:
        """
        Verify multiple signatures in batch
        
        Args:
            image_paths: List of image paths
            method: Prediction method
            
        Returns:
            List of verification results
        """
        results = []
        
        for image_path in image_paths:
            result = self.verify_signature(image_path, method)
            results.append(result)
        
        return results
    
    def compare_signatures(self, signature1_path: str, signature2_path: str, 
                          method: str = 'ensemble') -> Dict:
        """
        Compare two signatures for similarity
        
        Args:
            signature1_path: Path to first signature
            signature2_path: Path to second signature
            method: Prediction method
            
        Returns:
            Dictionary with comparison results
        """
        # Verify both signatures
        result1 = self.verify_signature(signature1_path, method)
        result2 = self.verify_signature(signature2_path, method)
        
        comparison = {
            'signature1': result1,
            'signature2': result2,
            'similarity_score': None,
            'same_person': None
        }
        
        # Calculate similarity based on individual model predictions
        if 'individual_predictions' in result1 and 'individual_predictions' in result2:
            similarity_scores = []
            
            for model_name in result1['individual_predictions'].keys():
                if model_name in result2['individual_predictions']:
                    # Compare probabilities
                    probs1 = result1['individual_predictions'][model_name]['probabilities']
                    probs2 = result2['individual_predictions'][model_name]['probabilities']
                    
                    # Calculate cosine similarity
                    similarity = np.dot(probs1, probs2) / (np.linalg.norm(probs1) * np.linalg.norm(probs2))
                    similarity_scores.append(similarity)
            
            if similarity_scores:
                comparison['similarity_score'] = np.mean(similarity_scores)
                comparison['same_person'] = comparison['similarity_score'] > 0.7
        
        return comparison

class SignatureVerificationAPI:
    """
    API-like interface for signature verification
    """
    
    def __init__(self, config: Config, model_paths: Dict[str, str]):
        self.verifier = SignatureVerifier(config, model_paths)
    
    def verify(self, image_path: str, method: str = 'ensemble') -> Dict:
        """Verify a single signature"""
        return self.verifier.verify_signature(image_path, method)
    
    def batch_verify(self, image_paths: List[str], method: str = 'ensemble') -> List[Dict]:
        """Verify multiple signatures"""
        return self.verifier.batch_verify(image_paths, method)
    
    def compare(self, signature1_path: str, signature2_path: str, method: str = 'ensemble') -> Dict:
        """Compare two signatures"""
        return self.verifier.compare_signatures(signature1_path, signature2_path, method)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.verifier.models.keys())

def visualize_verification_results(results: List[Dict], save_path: str = None) -> None:
    """
    Visualize verification results
    
    Args:
        results: List of verification results
        save_path: Path to save visualization
    """
    if not results:
        return
    
    # Count results
    genuine_count = sum(1 for r in results if r.get('is_genuine') == True)
    forged_count = sum(1 for r in results if r.get('is_genuine') == False)
    error_count = sum(1 for r in results if 'error' in r)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pie chart of results
    labels = ['Genuine', 'Forged', 'Error']
    sizes = [genuine_count, forged_count, error_count]
    colors = ['green', 'red', 'gray']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Verification Results Distribution')
    
    # Confidence distribution
    confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
    if confidences:
        ax2.hist(confidences, bins=20, alpha=0.7, color='blue')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
    
    # Model comparison (if ensemble method used)
    ensemble_results = [r for r in results if 'individual_predictions' in r]
    if ensemble_results:
        model_names = list(ensemble_results[0]['individual_predictions'].keys())
        model_accuracies = []
        
        for model_name in model_names:
            correct = 0
            total = 0
            for result in ensemble_results:
                if 'individual_predictions' in result and model_name in result['individual_predictions']:
                    pred = result['individual_predictions'][model_name]['prediction']
                    actual = 0 if result.get('is_genuine') else 1
                    if pred == actual:
                        correct += 1
                    total += 1
            
            if total > 0:
                model_accuracies.append(correct / total)
            else:
                model_accuracies.append(0)
        
        ax3.bar(model_names, model_accuracies)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Model Performance Comparison')
        ax3.tick_params(axis='x', rotation=45)
    
    # Results summary
    ax4.text(0.1, 0.8, f'Total Signatures: {len(results)}', fontsize=12)
    ax4.text(0.1, 0.7, f'Genuine: {genuine_count}', fontsize=12, color='green')
    ax4.text(0.1, 0.6, f'Forged: {forged_count}', fontsize=12, color='red')
    ax4.text(0.1, 0.5, f'Errors: {error_count}', fontsize=12, color='gray')
    
    if confidences:
        ax4.text(0.1, 0.4, f'Avg Confidence: {np.mean(confidences):.3f}', fontsize=12)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main function to demonstrate signature verification"""
    config = Config()
    config.create_directories()
    
    # Example model paths (update with actual trained model paths)
    model_paths = {
        'mobilenet': os.path.join(config.MODELS_DIR, 'mobilenet_signature_verifier.pth'),
        'resnet50': os.path.join(config.MODELS_DIR, 'resnet50_signature_verifier.pth'),
        'inceptionv3': os.path.join(config.MODELS_DIR, 'inceptionv3_signature_verifier.pth'),
        'vgg19': os.path.join(config.MODELS_DIR, 'vgg19_signature_verifier.pth'),
        'ensemble': os.path.join(config.MODELS_DIR, 'ensemble_signature_verifier.pth')
    }
    
    # Initialize API
    api = SignatureVerificationAPI(config, model_paths)
    
    print("Available models:", api.get_available_models())
    
    # Example usage
    print("\nExample signature verification:")
    print("=" * 50)
    
    # You would replace these with actual image paths
    example_image_path = "path/to/signature/image.tif"
    
    if os.path.exists(example_image_path):
        # Single signature verification
        result = api.verify(example_image_path, method='ensemble')
        print(f"Verification result: {result}")
        
        # Batch verification
        image_paths = [example_image_path]  # Add more paths for batch processing
        batch_results = api.batch_verify(image_paths, method='ensemble')
        print(f"Batch verification results: {batch_results}")
        
        # Visualize results
        visualize_verification_results(batch_results, 
                                     os.path.join(config.RESULTS_DIR, 'verification_results.png'))
    else:
        print(f"Example image not found: {example_image_path}")
        print("Please provide valid image paths for testing")

if __name__ == "__main__":
    main()
