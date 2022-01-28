import pandas as pd
import torch


class CsvTimeseriesDataset(torch.utils.data.Dataset):
    # CSV file columns to use.
    features = []

    def __init__(self,
        train: bool = False,
        split: float = 1., # split to use for testing (i.e., split=0.15 means 85% train and 15% test).
        tensor_drop_columns: list[str] = [],
        ):
        super().__init__()

        self.df: pd.DataFrame = None
        self.train = train
        self.split = split
        self.tensor_drop_columns = tensor_drop_columns
        self.features_tensor = [f for f in self.features if f not in set(tensor_drop_columns)] # Feature list for use with PyTorch tensors.

    def __len__(self):
        assert self.df is not None
        return self.df.shape[0]

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        assert self.df is not None
        # Collect dataset as dictionary of PyTorch tensors.
        return dict(zip(
            self.features_tensor,
            torch.from_numpy(
                self.df.iloc[index].drop(
                    columns=self.tensor_drop_columns,
                    ).to_numpy()
                ).T,
            ))