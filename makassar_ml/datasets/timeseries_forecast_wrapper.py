
from .csv_timeseries import CsvTimeseriesDataset
import torch
import numpy as np


class TimeseriesForecastDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self,
        dataset: CsvTimeseriesDataset,
        feature_cols: list[int],
        target_cols: list[int],
        history: int, # number of points to use as context for forecast
        horizon: int, # number of points to predict in the future
        ):
        assert history > 0
        assert horizon > 0
        self.dataset = dataset
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.history = history
        self.horizon = horizon

        # Compute starting index for each forecasting window.
        self.window_indexes = np.arange(len(self.dataset) - (self.history + self.horizon) + 1)

    def __len__(self):
        return self.window_indexes.shape[0]

    def window2index(self, index: int) -> int:
        """Convert window index to raw dataset index."""
        return self.window_indexes[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        # Retrieve start index for window.
        start_idx = self.window_indexes[index]

        # Collect history and horizon values for the feature and target columns.
        history_vals = self.dataset[start_idx:start_idx+self.history][:,self.feature_cols]
        horizon_vals = self.dataset[start_idx+self.history:start_idx+self.history+self.horizon][:,self.target_cols]

        # Return tuple of history and horizon values.
        return (history_vals, horizon_vals)
