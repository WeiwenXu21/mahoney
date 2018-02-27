import dask.array as da
import dask.cache
import numpy as np
import torch

import mahoney.data as data


def test_neurofinder():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    ds = data.Neurofinder('./data', n=256, stop=8, subset=['01.00'])
    assert len(ds) == 8

    # We can successfully iterate over the dataset.
    for i in range(len(ds)):
        assert ds[i]['x'].shape == (256, 512, 512)  # 256 frames per datum
        assert ds[i]['y'].shape == (2, 512, 512)  # 2 classes, background and foreground
        assert ds[i]['x'].dtype == 'float64'
        assert ds[i]['y'].dtype == 'int64'
        assert isinstance(ds[i]['x'], da.Array)
        assert isinstance(ds[i]['y'], np.ndarray)


def test_torchify():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    ds = data.Neurofinder('./data', n=256, stop=8, subset=['01.00'])
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


def test_dataloader():
    # We take the fitst 8 blocks of 256 frames from video '01.00'.
    # Video '01.00' has 2250 frames at a pixel resolution of 512x512.
    ds = data.Neurofinder('./data', n=256, stop=8, subset=['01.00'])
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
