from __future__ import annotations
from .beijing_pm25 import BeijingPM25Dataset
from .timeseries_forecast_wrapper import TimeseriesForecastDatasetWrapper
import pytorch_lightning as pl
import torch
from typing import Optional


class BeijingPM25LightningDataModule(pl.LightningDataModule):
    def __init__(self, 
        root: str, 
        feature_cols: list[int], 
        target_cols: list[int], 
        history: int, 
        horizon: int, 
        split: float,
        batch_size: int,
        ):
        self.root = root
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.history = history
        self.horizon = horizon
        self.split = split
        self.batch_size = batch_size

    def prepare_data(self):
        # Download the dataset.
        BeijingPM25Dataset(
            root=self.root,
            download=True,
            )

    def setup(self, stage: Optional[str] = None):

        # Create train/val datasets for dataloaders.
        if stage == 'fit' or stage is None:
            dataset_train_full = BeijingPM25Dataset(
                root=self.root,
                download=False,
                train=True,
                split=self.split,
                )
            train_n = len(dataset_train_full)
            train_val_cutoff = train_n - round(train_n*.25) # 75% train, 25% val

            self.dataset_train = torch.utils.data.Subset(dataset_train_full, list(range(0, train_val_cutoff)))
            self.dataset_val = torch.utils.data.Subset(dataset_train_full, list(range(train_val_cutoff, train_n)))

            self.dataset_train_wrap = TimeseriesForecastDatasetWrapper(
                dataset=self.dataset_train,
                feature_cols=self.feature_cols,
                target_cols=self.target_cols,
                history=self.history,
                horizon=self.horizon,
                )
            self.dataset_val_wrap = TimeseriesForecastDatasetWrapper(
                dataset=self.dataset_val,
                feature_cols=self.feature_cols,
                target_cols=self.target_cols,
                history=self.history,
                horizon=self.horizon,
                )

        # Create test dataset for dataloaders.
        if stage == 'test' or stage is None:
            self.dataset_test = BeijingPM25Dataset(
                root=self.root,
                download=False,
                train=False,
                split=self.split,
                )
            self.dataset_test_wrap = TimeseriesForecastDatasetWrapper(
                dataset=self.dataset_test,
                feature_cols=self.feature_cols,
                target_cols=self.target_cols,
                history=self.history,
                horizon=self.horizon,
                )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train_wrap,
            batch_size=self.batch_size,
            )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val_wrap,
            batch_size=self.batch_size,
            )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_test_wrap,
            batch_size=self.batch_size,
            )