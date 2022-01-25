import functools
import pandas as pd
import pathlib
import requests
import shutil
import torch
from tqdm.auto import tqdm

class BeijingMultiSiteAirQualityDataset(torch.utils.data.Dataset):
    """Wrapper for Beijing Multi-Site Air-Quality Data.

    https://archive-beta.ics.uci.edu/ml/datasets/beijing+multi+site+air+quality+data

    Dataset features:
        - `No`: row number
        - `year`: year of data in this row
        - `month`: month of data in this row
        - `day`: day of data in this row
        - `hour`: hour of data in this row
        - `PM2.5`: PM2.5 concentration (ug/m^3)
        - `PM10`: PM10 concentration (ug/m^3)
        - `SO2`: SO2 concentration (ug/m^3)
        - `NO2`: NO2 concentration (ug/m^3)
        - `CO`: CO concentration (ug/m^3)
        - `O3`: O3 concentration (ug/m^3)
        - `TEMP`: temperature (degree Celsius) 
        - `PRES`: pressure (hPa) 
        - `DEWP`: dew point temperature (degree Celsius) 
        - `RAIN`: precipitation (mm) 
        - `wd`: wind direction 
        - `WSPM`: wind speed (m/s) 
        - `station`: name of the air-quality monitoring site
    """
    dataset_root = pathlib.Path('beijing_multisite_air_quality')
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip'

    features = [
        'No',
        'year',
        'month',
        'day',
        'hour',
        'PM2.5',
        'PM10',
        'SO2',
        'NO2',
        'CO',
        'O3',
        'TEMP',
        'PRES',
        'DEWP',
        'RAIN',
        'wd',
        'WSPM',
        'station',
        ]

    # TODO write `download` and `load` functions.

    def __init__(self, root: str, download: bool = False):
        raise NotImplementedError