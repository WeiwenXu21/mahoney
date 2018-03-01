import mahoney.NMF_extraction as nmfe

def test_NMF():
    # All of these loads should succeed without error.
    coordinates = nmfe.NMF_extraction('./data/neurofinder.01.00')

    # number of neurons found should be over 0 and less than total pixel number
    assert 0 < len(coordinates) < 262144
