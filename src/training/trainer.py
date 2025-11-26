"""Reusable training loops and orchestration."""

import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        tuple: (train_loss, train_acc)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    return train_loss, train_acc


def validate(model, val_loader, criterion, device, desc='Validation'):
    """
    Validate model.

    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to run on
        desc: Progress bar description

    Returns:
        tuple: (val_loss, val_acc, per_class_acc)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=desc)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if pred == label:
                    class_correct[label_item] += 1

    val_loss = running_loss / total
    val_acc = 100. * correct / total

    per_class_acc = {
        cls: 100. * class_correct[cls] / class_total[cls]
        for cls in class_correct
    }

    return val_loss, val_acc, per_class_acc
