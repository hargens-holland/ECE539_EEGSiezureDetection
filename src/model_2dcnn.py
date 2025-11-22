import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import get_dataloaders

class EEG2DCNN(nn.Module):

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 7),
            padding=(1, 3),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 5),
            padding=(1, 2),
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(1, 2))  
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.global_pool(x)  
        x = torch.flatten(x, 1)  
        x = self.dropout(x)
        logits = self.fc(x)      
        return logits

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)  
        y = y.to(device)  
        optimizer.zero_grad()
        logits = model(x)            
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description="Train 2D CNN on EEG")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
    )
    model = EEG2DCNN(n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    torch.save(model.state_dict(), "eeg_2d_cnn_best.pt")
    print("Saved model to eeg_2d_cnn_best.pt")

if __name__ == "__main__":
    main()
