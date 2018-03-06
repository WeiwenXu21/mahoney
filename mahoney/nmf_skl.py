from sklearn.base import BaseEstimator, ClassifierMixin
import factorization
import extraction
import numpy as np

class NMF_dcomp(BaseEstimator, ClassifierMixin):
    '''Builds on the NMF decomposition with sklearn wrapper
    This wrapper allows us to perform NMF decomposition on
    the videos without
    '''
    def __init__(self, k=5):
        '''Initializes sklearn style NMF decomposition model
        Args:
            x: dask array from load_dataset function in data.py
            y: label mask if available, None if not
            meta: meta data from load dataset including "dataset"(i.e. : 01.00)
        '''
        self.k = k
    def fit(self, X):
        frames, dim1, dim2 = np.shape(X)
        full_img = X.reshape((frames,dim1*dim2))

        # Build and fit the model. Default: k=5
        algorithm = factorization.NMF(k=self.k)
        W, H = algorithm.fit(full_img)

        return W, H

class NMF_extract(BaseEstimator, ClassifierMixin):
    '''Builds on the NMF extraction with sklearn wrapper
    This wrapper allows us to perform NMF decomposition on
    the videos without
    '''
    def __init__(self, k=5):
        '''Initializes sklearn style NMF extraction model
        Args:
            x: dask array from load_dataset function in data.py
            y: label mask if available, None if not
            meta: meta data from load dataset including "dataset"(i.e. : 01.00)
        '''
        self.k = k

    def fit(self, X):
        algorithm = extraction.NMF(k=self.k)
        model = algorithm.fit(X)

        # The region of neurons found.
        regions = model.regions
        coordinates = regions.coordinates

        return coordinates
