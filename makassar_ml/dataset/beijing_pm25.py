
import pandas as pd
import torch

class BeijingPM25Dataset(torch.utils.data.Dataset):
    """Wrapper for Beijing PM2.5 dataset.

    https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data/

    Dataset features:
        - `No`: (NOT USED) row number
        - `year`: year of data in this row
        - `month`: month of data in this row
        - `day`: day of data in this row
        - `hour`: hour of data in this row
        - `pm2.5`: PM2.5 concentration (ug/m^3)
        - `DEWP`: Dew Point (â„ƒ)
        - `TEMP`: Temperature (â„ƒ)
        - `PRES`: Pressure (hPa)
        - `cbwd`: (NOT USED) Combined wind direction
        - `Iws`: Cumulated wind speed (m/s)
        - `Is`: Cumulated hours of snow
        - `Ir`: Cumulated hours of rain
    """

    # Dataset columnds.
    features = [
        # 'No',
        'year',
        'month',
        'day',
        'hour',
        'pm2.5',
        'DEWP',
        'TEMP',
        'PRES',
        # 'cbwd',
        'Iws',
        'Is',
        'Ir',
        ]

    def __init__(self, path: str):
        # Read the input file.
        self.df = pd.read_csv(path, usecols=self.features)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Collect dataset as dictionary of PyTorch tensors.
        return dict(zip(
            self.features,
            torch.from_numpy(
                self.df.iloc[index].to_numpy()
                ).T,
            ))