from mahoney import io
from mahoney import cnmf
import numpy as np

def test_cnmf():
    # All of these loads should succeed without error.
    video = io.load_video('../data/neurofinder.01.00')[:100]
    numb_of_frames, dims, dims = np.shape(video)
    
    A = cnmf.caiman_cnmf('01.00',video.compute()) # caiman does not work with Dask
    A_dims, neurons = np.shape(A)
    
    assert A_dims == dims*dims



