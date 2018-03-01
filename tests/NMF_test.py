import json

import mahoney.NMF as nmf

def test_NMF():
    # All of these loads should succeed without error.
    W,H = NMF.NMF_decomposition('./data/neurofinder.01.00')

    assert W.shape == (2250, 5)  # 2250 frames, 5 clusters
    assert H.shape == (5, 262144)  # 5 clusters, 512*512 pixels
