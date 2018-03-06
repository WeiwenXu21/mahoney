from sklearn.utils.estimator_checks import check_estimator
from mahoney.nmf_skl import NMF_dcomp, NMF_extract
from mahoney import io
from scipy.misc import imread
from glob import glob
import numpy as np
from numpy import array

# All of these loads should succeed without error.
video = io.load_video('./data/neurofinder.01.00')[:10]
num_frames, dims, dims = np.shape(video)
k = 5

def test_nmfskl_decomposition():
    model_d = NMF_dcomp(k=k)
    W,H = model.fit(video)

    assert np.shape(W) == (num_frames, k)  # for 01.00: 2250 frames, 5 clusters
    assert np.shape(H) == (k, dims*dims)  # for 01.00: 5 clusters, 512*512 pixels

def test_nmfskl_extraction():
    # All of these loads should succeed without error.
    model_e = NMF_extract(k=k)
    coordinates = model_e.fit(video)

    # number of neurons found should be over 0 and less than total pixel number
    assert 0 < len(coordinates) < dims*dims
