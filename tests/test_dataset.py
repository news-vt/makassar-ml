import csv
import makassar_ml as ml
import os
import pytest


@pytest.fixture
def dataset_root():
    return os.path.abspath('datasets/')


def test_beijing_pm25(dataset_root):
    csvfile = os.path.join(dataset_root, "beijing_pm2.5", "PRSA_data_2010.1.1-2014.12.31.csv")
    dataset = ml.dataset.beijing_pm25.BeijingPM25Dataset(path=csvfile)
    x = dataset[[3,4,5]]
    assert isinstance(x, dict)
    assert set(x.keys()) == set(dataset.features)