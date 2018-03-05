class Mean:
    '''An accumulator to compute the mean from batches of values.
    '''

    def __init__(self, **kwargs):
        '''Initialize the accumulator.

        Kwargs:
            Forwarded to `batch.mean()` during accumulation.
        '''
        self.kwargs = kwargs
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        '''Add a new batch of data to the accumulator.

        Args:
            batch:
                The data being accumulated. The batch should be either a scalar
                or an object with a `batch.mean()` method to get the mean value
                of the batch.

        Returns:
            self
        '''
        if hasattr(batch, 'mean'):
            n = len(batch)
            val = batch.mean(**self.kwargs)
        elif hasattr(batch, 'double'):
            n = len(batch)
            val = batch.double().mean(**self.kwargs)
        else:
            n = 1
            val = batch

        if self.n == 0:
            self.n = n
            self.val = val

        else:
            delta = val - self.val
            self.n += n
            self.val += delta * n / self.n

        return self

    def reduce(self):
        '''Return the accumulated mean and reset to the initial state.
        '''
        val = self.val
        self.n = 0
        self.val = 0
        return val
