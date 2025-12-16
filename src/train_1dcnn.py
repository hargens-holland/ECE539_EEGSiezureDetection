import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from data_loader import CHBMITDataset


class CNN1D(nn.Module):
    def __init__(self, in_channels=23, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  

            # Block 3
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  

            # Block 4
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1) 

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x).squeeze(-1) 
        return self.head(x)


def compute_metrics(y_true, y_pred, y_scores=None):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_scores is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics["auc"] = np.nan
    else:
        metrics["auc"] = np.nan

    return metrics

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device) 
        y = y.to(device) 

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    epoch_loss = running_loss / max(len(loader), 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / max(len(loader), 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
    return epoch_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="Train 1D CNN on raw EEG windows (temporal conv across channels)")
    parser.add_argument("--data", type=Path, required=True, help="Path to CHB-MIT data directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-model", type=str, default="cnn1d.pth")
    parser.add_argument("--seed", type=int, default=42)

    # These rely on your CURRENT data_loader supporting them.
    # If your data_loader already has these args, they will work.
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-files-per-patient", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)

    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use class-weighted loss (recommended for seizure imbalance)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (DO NOT change data_loader.py)
    print("\nLoading CHB-MIT dataset...")
    base_dataset = CHBMITDataset(
        args.data,
        max_patients=args.max_patients,
        max_files_per_patient=args.max_files_per_patient,
        max_windows=args.max_windows,
    )

    if len(base_dataset) == 0:
        print("ERROR: Dataset loaded 0 windows.")
        return

    # Random split
    n = len(base_dataset)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    train_ds = Subset(base_dataset, train_indices)
    val_ds = Subset(base_dataset, val_indices)
    test_ds = Subset(base_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("\nCreating 1D CNN model...")
    model = CNN1D(in_channels=23, num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    if args.use_class_weights:
        train_labels = []
        for i in train_indices:
            _, y = base_dataset[i]
            train_labels.append(int(y))
        train_labels = np.array(train_labels)

        num_pos = int((train_labels == 1).sum())
        num_neg = int((train_labels == 0).sum())
        print(f"Train class counts -> pos: {num_pos}, neg: {num_neg}")

        w_pos = num_neg / max(num_pos, 1)
        weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: [neg=1.0, pos={w_pos:.4f}]")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_epoch = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        val_auc = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_model)
            print(f"  ✓ Saved best model weights (Val AUC: {best_val_auc:.4f})")

        print()

    # Load best weights
    if best_epoch > 0 and os.path.exists(args.save_model):
        print(f"\nLoading best model weights from epoch {best_epoch} (Val AUC: {best_val_auc:.4f})")
        model.load_state_dict(torch.load(args.save_model, map_location=device))
    else:
        print("\nNo best checkpoint saved; using final weights.")

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)

    print(f"Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Test Sensitivity:  {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity:  {test_metrics['specificity']:.4f}")
    print(f"Test AUC:          {test_metrics['auc']:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()