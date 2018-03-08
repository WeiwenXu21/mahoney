import numpy as np

from mahoney import nmf
from mahoney import io


def test_nmf_extraction():
    # All of these loads should succeed without error.
    video = io.load_video('./data/neurofinder.01.00')[:10]
    numb_of_frames, dims, dims = np.shape(video)
    k = 5
    coordinates = nmf.nmf_extraction(video,k)

    # number of neurons found should be over 0 and less than total pixel number
    assert 0 < len(coordinates) < dims*dims

    # coordinates is a list of n*2 arrays, one for each neuron.
    # Each array is a list of pixel coordinates.
    for array in coordinates:
        (pixels, coords) = array.shape
        assert coords == 2
