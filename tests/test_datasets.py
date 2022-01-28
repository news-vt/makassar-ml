import csv
from turtle import down
import makassar_ml as ml
import os
import pytest
import torch


@pytest.fixture
def dataset_root():
    return os.path.abspath('datasets/')


def test_beijing_pm25(dataset_root):
    all = ml.datasets.BeijingPM25Dataset(root=dataset_root)
    train = ml.datasets.BeijingPM25Dataset(root=dataset_root, train=True, split=0.2)
    test = ml.datasets.BeijingPM25Dataset(root=dataset_root, train=False, split=0.2)
    assert os.path.exists(all.dataset_root)
    assert (len(train) + len(test)) == len(all)


def test_timeseries_forecast_wrapper(dataset_root):
    feature_cols = [0,1,2,3]
    target_cols = [-3]
    history = 5
    horizon = 3
    dset = ml.datasets.BeijingPM25Dataset(root=dataset_root)
    wrap = ml.datasets.TimeseriesForecastDatasetWrapper(dset, feature_cols, target_cols, history, horizon)

    def verify(window_idx: int):
        raw_idx = wrap.window2index(window_idx)
        ctx, tgt = wrap[window_idx]
        assert torch.equal(ctx, dset[raw_idx:raw_idx+history][:,feature_cols])
        assert torch.equal(tgt, dset[raw_idx+history:raw_idx+history+horizon][:,target_cols])

    verify(0)
    verify(-1)