"""Dataset protocol for research experiments."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader, Dataset


class BaseForecastDataset(Dataset):
    """Minimal contract every forecasting dataset should satisfy."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


def default_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def split_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = default_loader(train_set, batch_size, True, num_workers, True)
    val_loader = default_loader(val_set, batch_size, False, num_workers, True)
    test_loader = default_loader(test_set, batch_size, False, num_workers, False)
    return train_loader, val_loader, test_loader
