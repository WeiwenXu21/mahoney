import dask.array as da
import dask.cache
import numpy as np
import torch

import mahoney.data as data


def test_neurofinder():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    ds = data.load_dataset('./data', n=256, stop=8, subset=['01.00'])
    assert len(ds) == 8

    # We can successfully iterate over the dataset.
    for i in range(len(ds)):
        assert ds[i]['x'].shape == (256, 512, 512)  # 256 frames per datum
        assert ds[i]['y'].shape == (2, 512, 512)  # 2 classes, background and foreground
        assert ds[i]['x'].dtype == 'float64'
        assert ds[i]['y'].dtype == 'int64'
        assert isinstance(ds[i]['x'], da.Array)
        assert isinstance(ds[i]['y'], np.ndarray)
        assert isinstance(ds[i]['meta'], dict)


def test_torchify():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    ds = data.load_dataset('./data', n=256, stop=8, subset=['01.00'])
    ds = data.Torchify(ds)
    assert len(ds) == 8

    # We can successfully iterate over the dataset.
    for i in range(len(ds)):
        x, y, code = ds[i]
        assert x.shape == (256, 512, 512)  # 256 frames per datum
        assert y.shape == (2, 512, 512)  # 2 classes, background and foreground
        assert code == '01.00'
        assert x.dtype == 'float64'
        assert y.dtype == 'int64'
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(code, str)


def test_dataloader():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    ds = data.load_dataset('./data', n=256, stop=8, subset=['01.00'])
    ds = data.Torchify(ds)
    assert len(ds) == 8

    # We can successfully iterate over the dataset.
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    for batch in loader:
        x, y, code = batch
        assert x.shape == (4, 256, 512, 512)  # batch of 4, 256 frames per datum
        assert y.shape == (4, 2, 512, 512)  # 2 classes, background and foreground
        assert code == ('01.00',) * 4  # 1 code for each example in the batch
        assert isinstance(x, torch.DoubleTensor)
        assert isinstance(y, torch.LongTensor)


def test_gridsearch():
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import GridSearchCV

    # Grid search over a stub estimator with one dummy param
    class StubEstimator(BaseEstimator):
        def __init__(self, foo=1):
            self.foo = foo
        def score(self, x):
            return self.foo
        def fit(self, x):
            assert isinstance(x[0]['x'], da.Array)
            assert isinstance(x[0]['y'], np.ndarray)
            assert isinstance(x[0]['meta'], dict)
            return self

    # The dataset should be compatible with GridSearchCV
    ds = data.load_dataset('./data', n=256, stop=8, subset=['01.00'])
    stub = StubEstimator(foo=1)
    grid_stub = GridSearchCV(stub, {'foo': [1,2,3]})
    grid_stub.fit(ds)
