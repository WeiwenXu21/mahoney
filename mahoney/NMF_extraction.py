import numpy as np
#from sklearn.decomposition import NMF, IncrementalPCA
#from factorization import NMF
from extraction import NMF
from mahoney import io


def NMF_extraction(path):
    '''This is for neuron segmentation only using NMF (from thunder)
        
        Args:
            path:
                The base path to the dataset.
                
        Returns:
            coordinates:
                The coordinates for each neuron.
                Length of coordinates will be the number of neuron segmented.
        '''
    # Load in the video. Shape(frames, dims, dims)
    full_img = io.load_video(path)
    
    # Build and fit the model. Default: k=5
    algorithm = NMF()
    model = algorithm.fit(full_img)
    
    # The region of neurons found.
    regions = model.regions
    coordinates = regions.coordinates

    return coordinates
