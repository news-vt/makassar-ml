import functools
import pandas as pd
import pathlib
import requests
import shutil
import torch
from tqdm.auto import tqdm

class BeijingPM25Dataset(torch.utils.data.Dataset):
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
    """
    dataset_root = pathlib.Path('beijing_pm2.5')
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'

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

    def __init__(self, root: str, download: bool = False):
        # Join path elements.
        self.dataset_root = root / self.dataset_root

        # Downlaod if necessary.
        if download:
            self._download()

        # Load dataset contents.
        self._load()

    def _get_filepath(self) -> pathlib.Path:
        # Glean name of file from URL and build path.
        filename = pathlib.Path(self.dataset_url.rsplit('/', 1)[1])
        filepath = (self.dataset_root / filename).expanduser().resolve()
        return filepath

    def _download(self):

        # Glean name of file from URL and build path.
        filepath = self._get_filepath()

        # If file already exists, then do not download.
        if filepath.exists():
            return

        # Create root path tree.
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get file from URL and save to disk with progress bar.
        res = requests.get(self.dataset_url, stream=True, allow_redirects=True)
        file_size = int(res.headers.get('Content-Length', 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        res.raw.read = functools.partial(res.raw.read, decode_content=True) # Decompress if necessary.
        with tqdm.wrapattr(res.raw, 'read', total=file_size, desc=desc) as res_raw:
            with filepath.open('wb') as f:
                shutil.copyfileobj(res_raw, f)

    def _load(self):

        # Glean name of file from URL and build path.
        filepath = self._get_filepath()

        # Read the input file.
        self.df = pd.read_csv(filepath, usecols=self.features)

        # Create single date column from independent year/month/day columns.
        self.df = self.df.assign(date=pd.to_datetime(self.df[['year','month','day','hour']]))

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