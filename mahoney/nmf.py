import extraction
import numpy as np


def nmf_extraction(video, k=5):
    '''This is for neuron segmentation only using NMF (using thunder)

        Args:
            video:
                The video data to be processed.
                shape: (frames, dims, dims)
            k:
                The number of components being considered in NMF.
                Also can be interpreted as number of clustering.

        Returns:
            coordinates:
                Array of coordinates for each neuron.
                Length of coordinates will be the number of neuron segmented.
        '''
    # Build and fit the model. Default: k=5
    algorithm = extraction.NMF(k=k)
    model = algorithm.fit(video)

    # The region of neurons found.
    regions = model.regions
    coordinates = regions.coordinates


    return coordinates
