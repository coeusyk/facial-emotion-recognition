"""
Part 5.2: Ensemble Predictor (Optional)
Combine multiple models for improved accuracy using soft/hard voting.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vgg16_emotion import build_emotion_model


class EnsembleEmotionPredictor:
    """
    Ensemble predictor combining multiple emotion recognition models.
    
    Provides both soft voting (average probabilities) and hard voting (majority vote).
    """
    
    def __init__(self, model_paths, num_classes=7, device='cpu'):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths (list): List of paths to trained model checkpoints
            num_classes (int): Number of emotion classes
            device (str): Device to run inference on
        """
        self.device = device
        self.num_classes = num_classes
        self.models = []
        
        print(f"Loading {len(model_paths)} models for ensemble...")
        
        for i, model_path in enumerate(model_paths):
            print(f"  Loading model {i+1}: {model_path}")
            model = self._load_model(model_path)
            self.models.append(model)
        
        print(f"✓ Loaded {len(self.models)} models successfully")
    
    def _load_model(self, model_path):
        """Load a single model."""
        # Build model architecture
        model = build_emotion_model(
            num_classes=self.num_classes,
            pretrained=False,
            verbose=False
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_soft_voting(self, images):
        """
        Predict using soft voting (average probabilities).
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            
        Returns:
            tuple: (predictions, probabilities)
        """
        all_probabilities = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                all_probabilities.append(probabilities)
        
        # Average probabilities across all models
        avg_probabilities = torch.stack(all_probabilities).mean(dim=0)
        
        # Get predictions
        predictions = torch.argmax(avg_probabilities, dim=1)
        
        return predictions, avg_probabilities
    
    def predict_hard_voting(self, images):
        """
        Predict using hard voting (majority vote).
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            
        Returns:
            torch.Tensor: Predictions based on majority vote
        """
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions)
        
        # Stack predictions [num_models, batch_size]
        all_predictions = torch.stack(all_predictions)
        
        # Get majority vote for each sample
        # Convert to numpy for easier processing
        all_predictions_np = all_predictions.cpu().numpy()
        
        # Calculate mode (most common prediction) for each sample
        from scipy import stats
        majority_predictions = []
        
        for i in range(all_predictions_np.shape[1]):
            votes = all_predictions_np[:, i]
            mode_result = stats.mode(votes, keepdims=False)
            majority_predictions.append(mode_result.mode)
        
        return torch.tensor(majority_predictions, device=self.device)
    
    def predict_weighted_voting(self, images, weights=None):
        """
        Predict using weighted voting.
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            weights (list): Weight for each model (defaults to equal weights)
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        all_probabilities = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, weights):
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                all_probabilities.append(probabilities * weight)
        
        # Weighted sum of probabilities
        weighted_probabilities = torch.stack(all_probabilities).sum(dim=0)
        
        # Get predictions
        predictions = torch.argmax(weighted_probabilities, dim=1)
        
        return predictions, weighted_probabilities


def evaluate_ensemble(ensemble, test_loader, device):
    """
    Evaluate ensemble model on test set.
    
    Args:
        ensemble (EnsembleEmotionPredictor): Ensemble model
        test_loader (DataLoader): Test data loader
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics for each voting method
    """
    print("\n" + "="*60)
    print("EVALUATING ENSEMBLE MODEL")
    print("="*60)
    
    soft_predictions = []
    hard_predictions = []
    true_labels = []
    
    for images, labels in test_loader:
        images = images.to(device)
        
        # Soft voting
        soft_pred, _ = ensemble.predict_soft_voting(images)
        soft_predictions.extend(soft_pred.cpu().numpy())
        
        # Hard voting
        hard_pred = ensemble.predict_hard_voting(images)
        hard_predictions.extend(hard_pred.cpu().numpy())
        
        true_labels.extend(labels.numpy())
    
    soft_predictions = np.array(soft_predictions)
    hard_predictions = np.array(hard_predictions)
    true_labels = np.array(true_labels)
    
    # Calculate accuracies
    from sklearn.metrics import accuracy_score
    
    soft_acc = accuracy_score(true_labels, soft_predictions) * 100
    hard_acc = accuracy_score(true_labels, hard_predictions) * 100
    
    print(f"\nEnsemble Results:")
    print(f"  Soft Voting Accuracy: {soft_acc:.2f}%")
    print(f"  Hard Voting Accuracy: {hard_acc:.2f}%")
    
    return {
        'soft_accuracy': soft_acc,
        'hard_accuracy': hard_acc,
        'soft_predictions': soft_predictions,
        'hard_predictions': hard_predictions,
        'true_labels': true_labels
    }


def main():
    """Example usage of ensemble predictor."""
    print("="*60)
    print("ENSEMBLE EMOTION PREDICTOR - EXAMPLE USAGE")
    print("="*60)
    
    # Example: Load 3 different models
    # In practice, you would train multiple models with different:
    # - Random seeds
    # - Hyperparameters
    # - Data augmentation strategies
    # - Architecture variations
    
    model_paths = [
        'models/emotion_model_final_weights.pth',
        'models/emotion_model_stage2_best.pth',
        'models/emotion_model_best.pth'
    ]
    
    # Check which models exist
    import os
    existing_models = [p for p in model_paths if os.path.exists(p)]
    
    if len(existing_models) < 2:
        print("\n⚠ Not enough trained models found for ensemble")
        print("You need at least 2 different models.")
        print("\nTo create multiple models:")
        print("1. Train with different random seeds")
        print("2. Train with different hyperparameters")
        print("3. Train on different data splits")
        return
    
    print(f"\nFound {len(existing_models)} models:")
    for p in existing_models:
        print(f"  - {p}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create ensemble
    ensemble = EnsembleEmotionPredictor(
        model_paths=existing_models,
        num_classes=7,
        device=device
    )
    
    # Test with dummy batch
    print("\n" + "="*60)
    print("TESTING WITH DUMMY BATCH")
    print("="*60)
    
    dummy_images = torch.randn(8, 1, 48, 48).to(device)
    
    # Soft voting
    soft_pred, soft_prob = ensemble.predict_soft_voting(dummy_images)
    print(f"\nSoft Voting Predictions: {soft_pred.cpu().numpy()}")
    print(f"Soft Voting Probabilities shape: {soft_prob.shape}")
    
    # Hard voting
    hard_pred = ensemble.predict_hard_voting(dummy_images)
    print(f"\nHard Voting Predictions: {hard_pred.cpu().numpy()}")
    
    # Weighted voting (give more weight to first model)
    weights = [0.5, 0.3, 0.2][:len(existing_models)]
    weighted_pred, weighted_prob = ensemble.predict_weighted_voting(dummy_images, weights)
    print(f"\nWeighted Voting Predictions: {weighted_pred.cpu().numpy()}")
    print(f"Weights used: {weights}")
    
    # Evaluate on test set if available
    from pytorch_data_pipeline import create_dataloaders
    
    try:
        _, _, test_loader, class_names = create_dataloaders(
            data_dir='data/raw',
            batch_size=64,
            img_size=48,
            num_workers=2,
            val_split=0.0
        )
        
        if test_loader is not None:
            results = evaluate_ensemble(ensemble, test_loader, device)
            
            print("\n" + "="*60)
            print("ENSEMBLE EVALUATION COMPLETE")
            print("="*60)
            print(f"Ensemble typically improves accuracy by 2-5% over single models")
    except Exception as e:
        print(f"\n⚠ Could not evaluate on test set: {e}")
    
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTOR DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
