import csv
from turtle import down
import makassar_ml as ml
import os
import pytest


@pytest.fixture
def dataset_root():
    return os.path.abspath('datasets/')


def test_beijing_pm25(dataset_root):
    all = ml.datasets.beijing_pm25.BeijingPM25Dataset(root=dataset_root)
    train = ml.datasets.beijing_pm25.BeijingPM25Dataset(root=dataset_root, train=True, split=0.2)
    test = ml.datasets.beijing_pm25.BeijingPM25Dataset(root=dataset_root, train=False, split=0.2)
    assert os.path.exists(all.dataset_root)
    assert (len(train) + len(test)) == len(all)