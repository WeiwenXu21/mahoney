import numpy as np


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
        # If it has a `mean` method, delegate to that.
        if hasattr(batch, 'mean'):
            n = len(batch)
            val = batch.mean(**self.kwargs)

        # Some torch Tensors don't have a mean method,
        # so we cast to a DoubleTensor
        elif hasattr(batch, 'double'):
            n = len(batch)
            val = batch.double().mean(**self.kwargs)

        # If it has a `__len__`, we can probably use numpy
        elif hasattr(batch, '__len__'):
            n = len(batch)
            val = np.mean(batch, **self.kwargs)

        # Otherwise assume `batch` is a scalar
        else:
            n = 1
            val = batch

        # Accumulate the mean. This come from Chan et al:
        # Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979),
        # "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."
        # Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
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
