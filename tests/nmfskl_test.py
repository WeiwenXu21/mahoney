from mahoney.nmf_skl import NMF_dcomp, NMF_extract
from mahoney import io
from scipy.misc import imread
from glob import glob
import numpy as np
from numpy import array

def test_nmfskl_decomposition():
    # All of these loads should succeed without error.
    video = io.load_video('./data/neurofinder.01.00')[:10]
    video = video.compute()  # nmf_decomposition does not support dask arrays

    numb_of_frames, dim1, dim2 = np.shape(video)

    k = 5
    model_d = NMF_dcomp(k=k)
    W,H = model_d.fit(X=video)

    assert np.shape(W) == (numb_of_frames, k)  # for 01.00: 2250 frames, 5 clusters
    assert np.shape(H) == (k, dim1*dim2)  # for 01.00: 5 clusters, 512*512 pixels

def test_nmfskl_extraction():
    # All of these loads should succeed without error.
    video = io.load_video('./data/neurofinder.01.00')[:10]
    numb_of_frames, dim1, dim2 = np.shape(video)
    k = 5
    model_e = NMF_extract(k=k)
    coordinates = model_e.fit(X=video)
    # number of neurons found should be over 0 and less than total pixel number
    assert 0 < len(coordinates) < dim1*dim2

    # coordinates is a list of n*2 arrays, one for each neuron.
    # Each array is a list of pixel coordinates.
    for array in coordinates:
        (pixels, coords) = array.shape
        assert coords == 2
