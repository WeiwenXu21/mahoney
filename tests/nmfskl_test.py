from sklearn.utils.estimator_checks import check_estimator
from mahoney.nmf_skl import NMF_dcomp, NMF_extract
from mahoney import io
from scipy.misc import imread
from glob import glob
import numpy as np
from numpy import array


def test_nmfskl_decomposition():
    video = io.load_video('./data/neurofinder.01.00')
    num_frames, dim1, dim2 = np.shape(video)
    video = video.reshape((num_frames,dim1*dim2))
    k = 5
    model_d = NMF_dcomp(k=k)
    W,H = model_d.fit(X=video)

    assert np.shape(W) == (num_frames, k)  # for 01.00: 2250 frames, 5 clusters
    assert np.shape(H) == (k, dim1*dim2)  # for 01.00: 5 clusters, 512*512 pixels

def test_nmfskl_extraction():
    video = io.load_video('./data/neurofinder.01.00')[:10]
    num_frames, dim1, dim2 = np.shape(video)
    k = 5
    # All of these loads should succeed without error.
    model_e = NMF_extract(k=k)
    coordinates = model_e.fit(X=video)

    # number of neurons found should be over 0 and less than total pixel number
    assert 0 < len(coordinates) < dim1*dim2
