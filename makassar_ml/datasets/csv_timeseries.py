from __future__ import annotations
import numpy as np
import pandas as pd
import pathlib
import torch


class CsvTimeseriesDataset(torch.utils.data.Dataset):

    def __init__(self,
        filepath: pathlib.Path,
        train: bool = False,
        split: float = 1., # split to use for testing (i.e., split=0.15 means 85% train and 15% test).
        drop_features: list[str] = [],
        ):
        self.filepath = filepath
        self.train = train
        self.split = split
        self.drop_features = drop_features if drop_features else [] # Protect against passing `None`.

        # Load dataset contents.
        self.load(filepath)

    @property
    def features(self) -> np.ndarray:
        return np.array([f for f in self.df.columns.values.tolist() if f not in set(self.drop_features)])

    def load(self, filepath: pathlib.Path):
        """Load CSV contents from file.

        Args:
            filepath (pathlib.Path): Path to CSV file.
        """

        # Read the input file.
        self.df = pd.read_csv(filepath)

        # Compute cutoff index for train/test split.
        n = self.df.shape[0] # Number of data records.
        cutoff = n - round(self.split * n) # Index where test set starts.

        # Remove rows from dataframe that are not needed for the current split.
        if self.train:
            self.df = self.df[:cutoff]
        else:
            self.df = self.df[cutoff:]

    def __len__(self):
        assert self.df is not None
        return self.df.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        assert self.df is not None
        return torch.from_numpy(
            self.df.drop(
                columns=self.drop_features,
                ).iloc[index].to_numpy()
            )