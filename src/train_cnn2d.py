"""
Single-file 2D CNN training for EEG seizure detection using spectrograms.
Uses the existing CHBMITDataset from data_loader.py
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import signal
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from pathlib import Path
import sys

# Import existing data loader
from data_loader import CHBMITDataset


def make_spectrogram(window, nperseg=64, noverlap=32, target_fs=256):
    """
    Convert a multi-channel EEG window to a single-channel spectrogram.
    
    Args:
        window: Array of shape (23, 1024) - 23 channels, 1024 samples
        nperseg: Length of each segment for STFT
        noverlap: Number of points to overlap between segments
        target_fs: Sampling frequency (256 Hz)
        
    Returns:
        spectrogram: Array of shape (1, freq_bins, time_bins)
    """
    # Average across channels to get single signal
    # Alternative: could use max, sum, or other aggregation
    averaged_signal = np.mean(window, axis=0)
    
    # Compute spectrogram using scipy
    f, t, Sxx = signal.spectrogram(
        averaged_signal,
        fs=target_fs,
        nperseg=nperseg,
        noverlap=noverlap,
        mode='magnitude',
    )
    
    # Log-scale normalization (common for EEG spectrograms)
    Sxx = np.log1p(Sxx)  # log(1 + Sxx) to avoid log(0)
    
    # Min-max normalize to [0, 1]
    Sxx_min = Sxx.min()
    Sxx_max = Sxx.max()
    if Sxx_max > Sxx_min:
        Sxx = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)
    
    # Add channel dimension: (freq_bins, time_bins) -> (1, freq_bins, time_bins)
    Sxx = Sxx[np.newaxis, :, :]
    
    return Sxx.astype(np.float32)


class SpectrogramDataset(Dataset):
    """
    Wrapper around CHBMITDataset that converts windows to spectrograms.
    """
    
    def __init__(self, base_dataset, nperseg=64, noverlap=32):
        """
        Args:
            base_dataset: CHBMITDataset instance
            nperseg: STFT segment length
            noverlap: STFT overlap
        """
        self.base_dataset = base_dataset
        self.nperseg = nperseg
        self.noverlap = noverlap
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get window and label from base dataset
        window, label = self.base_dataset[idx]
        
        # Convert to numpy if tensor
        if isinstance(window, torch.Tensor):
            window = window.numpy()
        
        # Convert window to spectrogram
        spectrogram = make_spectrogram(
            window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        
        # Convert to tensor
        spectrogram = torch.FloatTensor(spectrogram)
        label = torch.LongTensor([label])[0]
        
        return spectrogram, label


class CNN2D(nn.Module):
    """
    Simple 2D CNN for spectrogram-based seizure detection.
    """
    
    def __init__(self, num_classes=2):
        super(CNN2D, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block (optional, for deeper network)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # We'll compute flattened size dynamically after first forward pass
        self.flatten_size = None
        self.fc = None
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, freq_bins, time_bins)
            
        Returns:
            logits: Output logits of shape (batch, num_classes)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Initialize fully connected layer on first forward pass
        if self.flatten_size is None:
            self.flatten_size = x.size(1)
            self.fc = nn.Linear(self.flatten_size, 2).to(x.device)
        
        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def compute_metrics(y_true, y_pred, y_scores=None):
    """
    Compute classification metrics.
    
    Returns:
        Dictionary with accuracy, sensitivity, specificity, auc
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_scores is not None and isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity (Recall, True Positive Rate)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    if y_scores is not None:
        try:
            # If y_scores is 2D (probabilities), take positive class
            if len(y_scores.shape) > 1 and y_scores.shape[1] > 1:
                y_scores = y_scores[:, 1]
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for spectrograms, labels in dataloader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return epoch_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for spectrograms, labels in dataloader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores),
    )
    
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train 2D CNN on EEG spectrograms")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to EEG data directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="cnn2d.pth",
        help="Path to save best model (default: cnn2d.pth)",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=64,
        help="STFT segment length (default: 64)",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=32,
        help="STFT overlap (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check data directory exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\nERROR: Data directory does not exist: {data_path}")
        print(f"Please check the path and try again.")
        return
    
    if not data_path.is_dir():
        print(f"\nERROR: Path is not a directory: {data_path}")
        return
    
    # Debug: Check what's in the data directory
    print(f"\nData directory: {data_path}")
    print(f"Contents of data directory:")
    items = list(data_path.iterdir())
    if len(items) == 0:
        print("  (empty directory)")
    else:
        for item in items[:10]:  # Show first 10 items
            item_type = "DIR" if item.is_dir() else "FILE"
            print(f"  {item_type}: {item.name}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more items")
    
    # Check for patient directories
    patient_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(patient_dirs)} patient directories")
    if len(patient_dirs) == 0:
        print("ERROR: No patient directories found!")
        print("Expected structure:")
        print("  EEG_data/")
        print("    chb01/")
        print("      chb01_01.edf")
        print("      chb01-summary.txt")
        print("    chb02/")
        print("      ...")
        return
    
    # Check for EDF files
    edf_count = 0
    for patient_dir in patient_dirs[:5]:  # Check first 5 patients
        edf_files = list(patient_dir.glob("*.edf"))
        edf_count += len(edf_files)
        if len(edf_files) > 0:
            print(f"  {patient_dir.name}: {len(edf_files)} EDF files")
    
    if edf_count == 0:
        print("WARNING: No .edf files found in patient directories!")
        print("Make sure your data directory contains .edf files.")
    
    # Load base dataset
    print("\nLoading CHB-MIT dataset...")
    base_dataset = CHBMITDataset(args.data)
    
    if len(base_dataset) == 0:
        print("\nERROR: Dataset loaded 0 windows!")
        print("This usually means:")
        print("  1. No .edf files were found")
        print("  2. EDF files couldn't be read")
        print("  3. Data directory structure is incorrect")
        return
    
    # Create spectrogram dataset
    print("Creating spectrogram dataset...")
    dataset = SpectrogramDataset(
        base_dataset,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
    )
    
    # Create train/val/test splits (random split for now)
    # TODO: Implement patient-independent split for final evaluation
    n = len(dataset)
    
    if n == 0:
        print("\nERROR: No data loaded! Cannot proceed with training.")
        print("Please check:")
        print("  1. Data path is correct")
        print("  2. Data directory contains patient subdirectories (chb01, chb02, etc.)")
        print("  3. Patient directories contain .edf files")
        return
    
    indices = np.random.permutation(n)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create data loaders
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create model
    print("\nCreating 2D CNN model...")
    model = CNN2D(num_classes=2)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_auc = 0.0
    best_epoch = 0
    
    print("\nStarting training...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics['accuracy'])
        val_accs.append(val_metrics['accuracy'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        val_auc = val_metrics['auc'] if not np.isnan(val_metrics['auc']) else 0.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_metrics': val_metrics,
            }, args.save_model)
            print(f"  ✓ Saved best model (Val AUC: {val_auc:.4f})")
        
        print()
    
    # Load best model
    print(f"\nLoaded best model from epoch {best_epoch}")
    checkpoint = torch.load(args.save_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

