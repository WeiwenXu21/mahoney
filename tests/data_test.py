import dask.array as da
import dask.cache
import numpy as np
import torch

import mahoney.data as data


def test_dataset():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    x, y, meta = data.load_dataset('./data', n=8, frames=256, skip=4, subset=['01.00'])
    assert len(x) == len(y) == len(meta) == 8

    # We can successfully iterate over the dataset.
    for i in range(8):
        assert x[i].shape == (256, 512, 512)  # 256 frames per datum
        assert x[i].dtype == 'float64'
        assert y[i].shape == (512, 512)
        assert y[i].dtype == 'int64'
        assert isinstance(x[i], da.Array)
        assert isinstance(y[i], np.ndarray)
        assert isinstance(meta[i], dict)


def test_torchify():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    x, y, meta = data.load_dataset('./data', n=8, frames=256, subset=['01.00'])
    ds = data.Torchify(x, y)
    assert len(ds) == 8

    # We can successfully iterate over the dataset.
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (256, 512, 512)  # 256 frames per datum
        assert y.shape == (512, 512)
        assert x.dtype == 'float64'
        assert y.dtype == 'int64'
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    # We can successfully iterate over the dataset using a dataloader.
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    for batch in loader:
        x, y = batch
        assert x.shape == (4, 256, 512, 512)  # batch of 4, 256 frames per datum
        assert y.shape == (4, 512, 512)  # 2 classes, background and foreground
        assert isinstance(x, torch.DoubleTensor)
        assert isinstance(y, torch.LongTensor)


def test_gridsearch():
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import GridSearchCV

    # Grid search over a stub estimator with one dummy param
    class StubEstimator(BaseEstimator):
        def __init__(self, foo=1):
            self.foo = foo
        def score(self, x, y):
            return self.foo
        def fit(self, x, y):
            assert isinstance(x[0], da.Array)
            assert isinstance(y[0], np.ndarray)
            return self

    # The dataset should be compatible with GridSearchCV
    x, y, meta = data.load_dataset('./data', n=8, frames=256, subset=['01.00'])
    stub = StubEstimator(foo=1)
    grid_stub = GridSearchCV(stub, {'foo': [1,2,3]})
    grid_stub.fit(x, y)
