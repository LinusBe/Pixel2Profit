import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score # NEU: Import für F1-Score

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn, device: str):
    """Performs a single training epoch for the given model.

    This function iterates over the provided dataloader, setting the model to
    training mode. For each batch, it performs a forward pass, calculates the
    loss, and executes a backward pass to update the model's weights via the
    optimizer. Progress is displayed using a tqdm progress bar.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    dataloader : DataLoader
        The DataLoader providing the training data batches.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's weights.
    loss_fn : callable
        The loss function used to compute the difference between predictions
        and ground truth. It should accept predictions and target tensors.
    device : str
        The device ('cpu' or 'cuda') on which to perform computations.

    Returns
    -------
    float
        The average loss calculated over all batches in the epoch.
    """
    model.train()
    total_loss = 0
    
    # NEU: tqdm-Objekt in einer Variable speichern, um es aktualisieren zu können
    progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)
    
    for i, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y.float().unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / len(dataloader)
    # Entferne den print(), da die Info in der tqdm-Bar steht
    return avg_loss

def validate_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, device: str) -> dict:
    """Performs a single validation epoch and computes evaluation metrics.

    This function sets the model to evaluation mode and iterates over the
    validation dataloader without computing gradients. It calculates the
    average loss, prediction accuracy, and the F1-score for the entire
    validation set.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be evaluated.
    dataloader : DataLoader
        The DataLoader providing the validation data batches.
    loss_fn : callable
        The loss function used to compute the validation loss.
    device : str
        The device ('cpu' or 'cuda') on which to perform computations.

    Returns
    -------
    dict
        A dictionary containing the validation metrics:
        - 'val_loss' (float): The average loss over the validation set.
        - 'val_accuracy' (float): The prediction accuracy in percent.
        - 'f1_score' (float): The F1-score for the binary classification task.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []
    progress_bar = tqdm(dataloader, desc="Validation Epoch", leave=False)

    with torch.no_grad():
        for i, (X, y) in enumerate(progress_bar):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y.float().unsqueeze(1)).item()
            
            predicted_labels = (torch.sigmoid(pred) > 0.5).float()
            correct_predictions += (predicted_labels == y.unsqueeze(1)).sum().item()
            total_samples += y.size(0)
            
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0.0)
    
    print(f"Validation -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1-Score: {f1:.4f}")
    
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'f1_score': f1}