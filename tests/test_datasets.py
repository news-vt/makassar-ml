import csv
from turtle import down
import makassar_ml as ml
import os
import pytest


@pytest.fixture
def dataset_root():
    return os.path.abspath('datasets/')


def test_beijing_pm25(dataset_root):
    dataset = ml.datasets.beijing_pm25.BeijingPM25Dataset(root=dataset_root)
    assert os.path.exists(dataset.dataset_root)
    assert dataset.df is not None
    x = dataset[[3,4,5]]
    assert isinstance(x, dict)
    assert set(x.keys()) == set(dataset.features)