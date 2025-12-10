# src/data/dataset.py
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_time_splits(df: pd.DataFrame, train_end: str, val_end: str) -> SplitIndices:
    """Create index splits based on date ranges."""
    dates = pd.to_datetime(df["date"])

    train_mask = dates <= pd.to_datetime(train_end)
    val_mask = (dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(val_end))
    test_mask = dates > pd.to_datetime(val_end)

    return SplitIndices(
        train_idx=np.where(train_mask)[0],
        val_idx=np.where(val_mask)[0],
        test_idx=np.where(test_mask)[0],
    )


class PriceWindowDataset(Dataset):
    """
    Windowed dataset for price-only (and later price+macro) features.
    Returns: (X_window, y) where
      X_window: (window_size, num_features)
      y: scalar label for last day in window (0/1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        feature_cols: List[str],
        window_size: int,
    ):
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.feature_cols = feature_cols
        self.window_size = window_size

        self.X = self.df[feature_cols].values.astype("float32")
        self.y = self.df["label_up"].values.astype("float32")

    def __len__(self):
        # we need full windows ending at each index
        return len(self.indices) - (self.window_size - 1)

    def __getitem__(self, idx):
        # idx is local to this split, map to global index
        idx_global = self.indices[idx + (self.window_size - 1)]
        start = idx_global - self.window_size + 1
        end = idx_global + 1

        x_window = self.X[start:end]          # (W, F)
        y = self.y[idx_global]                # label at the end of window

        return torch.from_numpy(x_window), torch.tensor(y)
