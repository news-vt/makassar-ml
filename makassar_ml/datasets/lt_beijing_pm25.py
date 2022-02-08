from .beijing_pm25 import BeijingPM25Dataset
import pytorch_lightning as pl
import torch


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