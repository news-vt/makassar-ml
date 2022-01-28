
from .csv_timeseries import CsvTimeseriesDataset
import torch
import numpy as np


class TimeseriesForecastDatasetWrapper(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for Timeseries Forecasting.

    Splits a CSV timeseries dataset into windows for
    forecasting tasks.

    The `history` is the number of points to consider as
    context for forecasting. The `horizon` is the number
    of points to predict in the future.

    Calls to `__getitem__` return a `tuple` of the form:
        - `history_x`: History feature values.
        - `history_y`: History target values.
        - `horizon_x`: Horizon feature values.
        - `horizon_y`: Horizon target values.
    """

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Retrieve start index for window.
        start_idx = self.window_indexes[index]

        # History source and target values.
        history_x = self.dataset[start_idx:start_idx+self.history][:,self.feature_cols]
        history_y = self.dataset[start_idx:start_idx+self.history][:,self.target_cols]

        # Horizon source and target values.
        horizon_x = self.dataset[start_idx+self.history:start_idx+self.history+self.horizon][:,self.feature_cols]
        horizon_y = self.dataset[start_idx+self.history:start_idx+self.history+self.horizon][:,self.target_cols]

        # Return tuple of history and horizon values.
        return (history_x, history_y, horizon_x, horizon_y)
