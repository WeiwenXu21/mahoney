import dask.array as da
import numpy as np
import mahoney.preprocess as pp

def test_normalize():
    arr = np.random.normal(loc=0, scale=10, size=(10, 5, 5))

    out = pp.normalize(arr)
    mean = np.mean(out)
    std = np.std(out)
    assert -0.9 < mean < 0.1
    assert 0.9 < std < 1.1

    darr = da.from_array(arr, chunks=(1))
    dout = pp.normalize(darr)
    assert isinstance(darr, da.Array)
    assert isinstance(dout, da.Array)
