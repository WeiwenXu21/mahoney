from mahoney import accumulators


def test_mean():
    mean = accumulators.Mean()

    # It can handle batches of different sizes.
    # It can handle both collections and scalars.
    mean.accumulate([1])
    mean.accumulate([2,3])
    mean.accumulate([4,5,6])
    mean.accumulate([7,8])
    mean.accumulate(9)
    assert mean.reduce() == 5

    # Once it's been reduced, the accumulator is reset.
    assert mean.reduce() == 0
