"""
Part 4.1: Training and Validation Functions
Core training loop functions for PyTorch model training.
"""

import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=None):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        epoch (int): Current epoch number (for display)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]' if epoch else 'Training')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch=None, desc='Validation'):
    """
    Validate the model.
    
    Args:
        model (torch.nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch (int): Current epoch number (for display)
        desc (str): Description for progress bar
        
    Returns:
        tuple: (average_loss, accuracy, per_class_acc)
    """
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = {}
    class_total = {}
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [{desc}]' if epoch else desc)
    
    # No gradient computation during validation
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy tracking
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if pred == label:
                    class_correct[label_item] += 1
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    # Calculate validation metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for class_idx in sorted(class_total.keys()):
        if class_total[class_idx] > 0:
            per_class_acc[class_idx] = 100.0 * class_correct[class_idx] / class_total[class_idx]
        else:
            per_class_acc[class_idx] = 0.0
    
    return val_loss, val_acc, per_class_acc


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_acc: Validation accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load model on
        
    Returns:
        tuple: (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', 0.0)
    val_acc = checkpoint.get('val_acc', 0.0)
    
    return model, optimizer, epoch, {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve.
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'max' for accuracy, 'min' for loss
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, metric):
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value (accuracy or loss)
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f'✓ Metric improved to {metric:.4f}')
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'⚠ No improvement for {self.counter}/{self.patience} epochs')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'✗ Early stopping triggered!')
                return True
            
            return False


class MetricTracker:
    """
    Track and save training metrics.
    """
    
    def __init__(self):
        """Initialize metric storage."""
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """
        Update metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Current learning rate
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
    
    def save_to_csv(self, filepath):
        """
        Save metrics to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
    
    def plot_metrics(self, save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path: Path to save plot (optional)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], 
                       label='Train Loss', marker='o')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], 
                       label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['epoch'], self.history['train_acc'], 
                       label='Train Acc', marker='o')
        axes[0, 1].plot(self.history['epoch'], self.history['val_acc'], 
                       label='Val Acc', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['epoch'], self.history['lr'], 
                       marker='o', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss gap plot (overfitting indicator)
        loss_gap = [t - v for t, v in zip(self.history['train_loss'], 
                                          self.history['val_loss'])]
        axes[1, 1].plot(self.history['epoch'], loss_gap, 
                       marker='o', color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Loss - Val Loss')
        axes[1, 1].set_title('Overfitting Indicator (lower is better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics plot saved to: {save_path}")
        
        plt.show()


def main():
    """Test training utilities."""
    print("=" * 60)
    print("TRAINING UTILITIES TEST")
    print("=" * 60)
    
    # Test EarlyStopping
    print("\nTesting EarlyStopping (mode='max', patience=3):")
    early_stop = EarlyStopping(patience=3, mode='max')
    
    test_accuracies = [70.0, 72.0, 73.0, 72.5, 72.3, 72.1, 71.8]
    for i, acc in enumerate(test_accuracies):
        print(f"  Epoch {i+1}: Accuracy = {acc:.1f}%")
        should_stop = early_stop(acc)
        if should_stop:
            print(f"  --> Training stopped at epoch {i+1}")
            break
    
    # Test MetricTracker
    print("\nTesting MetricTracker:")
    tracker = MetricTracker()
    
    for epoch in range(1, 6):
        tracker.update(
            epoch=epoch,
            train_loss=2.0 - epoch*0.2,
            train_acc=50.0 + epoch*5,
            val_loss=2.1 - epoch*0.15,
            val_acc=48.0 + epoch*4.5,
            lr=0.001 / (epoch*0.5)
        )
    
    print("  Metric history:")
    for key, values in tracker.history.items():
        print(f"    {key}: {values}")
    
    print("\n✓ Training utilities test complete!")


if __name__ == "__main__":
    main()
