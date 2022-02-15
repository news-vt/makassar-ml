from __future__ import annotations
from .csv_timeseries import CsvTimeseriesDataset
import functools
import pandas as pd
import pathlib
import requests
import shutil
import torch
from tqdm.auto import tqdm

def download_file(url: str, dst: pathlib.Path):
    """Download a file from a URL.

    Args:
        url (str): URL string.
        dst (pathlib.Path): Destination path.
    """

    # If file already exists, then do not download.
    if dst.exists():
        return

    # Create root path tree.
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Get file from URL and save to disk with progress bar.
    res = requests.get(url, stream=True, allow_redirects=True)
    file_size = int(res.headers.get('Content-Length', 0))
    desc = "(Unknown total file size)" if file_size == 0 else ""
    res.raw.read = functools.partial(res.raw.read, decode_content=True) # Decompress if necessary.
    with tqdm.wrapattr(res.raw, 'read', total=file_size, desc=desc) as res_raw:
        with dst.open('wb') as f:
            shutil.copyfileobj(res_raw, f)


class BeijingPM25Dataset(CsvTimeseriesDataset):
    """Wrapper for Beijing PM2.5 dataset.

    https://archive-beta.ics.uci.edu/ml/datasets/beijing+pm2+5+data

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
        - `datetime`: (NOT USED) dynamically generated datetime string
    """
    dataset_root = pathlib.Path('beijing_pm2.5')
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'

    def __init__(self, 
        root: str,
        download: bool = False,
        train: bool = False,
        split: float = 1., # split to use for testing (i.e., split=0.15 means 85% train and 15% test).
        drop_features: list[str] = ['No','cbwd','datetime'], # columns to omit from PyTorch retrieval.
        normalize: str = None, # Supports ['standard', 'minmax'].
        ):
        self.dataset_root = root / self.dataset_root # Join path elements.
        self.normalize = normalize

        # Downlaod if necessary.
        filename = pathlib.Path(self.dataset_url.rsplit('/', 1)[1])
        filepath = (self.dataset_root / filename).expanduser().resolve()
        if download:
            download_file(url=self.dataset_url, dst=filepath)

        # Initialize parent class to load dataset from disk.
        super().__init__(
            filepath=filepath,
            train=train,
            split=split,
            drop_features=drop_features,
            )

    def load(self, filepath: pathlib.Path):
        """Load CSV from file.

        This override function dynamically adds a `datetime` column.

        Args:
            filepath (pathlib.Path): Path to CSV file.
        """
        super().load(filepath)

        # Standard normalization.
        cols = [
            'pm2.5',
            'DEWP',
            'TEMP',
            'PRES',
            'Iws',
            'Is',
            'Ir',
        ]
        if self.normalize == 'standard':
            self.df[cols] = (self.df[cols] - self.df[cols].mean())/self.df[cols].std()
        # MinMax normalization.
        elif self.normalize == 'minmax':
            self.df[cols] = (self.df[cols] - self.df[cols].min())/(self.df[cols].max() - self.df[cols].min())

        # Create single date column from independent year/month/day columns.
        self.df['datetime'] = pd.to_datetime(self.df[['year','month','day','hour']])

    def __getitem__(self, index) -> torch.Tensor:
        # Manually convert underlying tensor to float32.
        return super().__getitem__(index).float()