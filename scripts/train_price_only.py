# scripts/train_price_only.py
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data.dataset import make_time_splits, PriceWindowDataset
from src.models.price_only import PriceOnlyMLP


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss, total_correct, total_count = 0.0, 0, 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_count += y.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss, total_correct, total_count = 0.0, 0, 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_count += y.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    # --- config (hard-coded for now; we can move this to yaml later) ---
    window_size = 30
    batch_size = 64
    num_epochs = 20
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_end = "2018-12-31"
    val_end = "2021-12-31"

    processed_path = Path("data/processed/daily_merged.parquet")

    # feature columns: all numeric features except date, label, future_*
    df = pd.read_parquet(processed_path)
    df["date"] = pd.to_datetime(df["date"])

    drop_cols = ["date", "future_price", "future_ret_1d", "label_up"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print("Feature columns:", feature_cols)

    splits = make_time_splits(df, train_end=train_end, val_end=val_end)

    train_ds = PriceWindowDataset(df, splits.train_idx, feature_cols, window_size)
    val_ds = PriceWindowDataset(df, splits.val_idx, feature_cols, window_size)
    test_ds = PriceWindowDataset(df, splits.test_idx, feature_cols, window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_features = len(feature_cols)
    model = PriceOnlyMLP(window_size=window_size, num_features=num_features)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

    # final test metrics
    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"\nTest loss={test_loss:.4f} acc={test_acc:.4f}")

    # save model
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/price_only_mlp.pt")
    print("Saved model to checkpoints/price_only_mlp.pt")


if __name__ == "__main__":
    main()
