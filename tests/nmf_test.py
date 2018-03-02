import mahoney.nmf as nmf
from mahoney import io
from scipy.misc import imread
from glob import glob
from numpy import array

def test_nmf_decomposition():
    # All of these loads should succeed without error.
    files = sorted(glob('./data/neurofinder.01.00/images/image*.tiff'))
    video = array([imread(f) for f in files])
    k = 5
    W,H = nmf.NMF_decomposition(video,k)

    assert W.shape == (3048, 5)  # 3048 frames, 5 clusters
    assert H.shape == (5, 262144)  # 5 clusters, 512*512 pixels

def test_nmf_extraction():
    # All of these loads should succeed without error.
    video = io.load_video('./data/neurofinder.01.00')
    k = 5
    coordinates = nmf.nmf_extraction(video,k)
    
    # number of neurons found should be over 0 and less than total pixel number
    assert 0 < len(coordinates) < 262144




