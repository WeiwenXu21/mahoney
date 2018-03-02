import factorization
import extraction
import numpy as np

def nmf_decomposition(video, k=5):
    '''This is normal NMF decomposition (using thunder)
        
        Args:
            video:
                The video data to be processed.
                shape: (frames, dims, dims)
        
        Returns:
            W:
                W is the derived features matrix with k latent features
                shape(frames, k)
            
            H:
                H is the coefficients matrix that associates with W
                shape(k, pixels)
        '''
    # Load in the video and flatten each frame. Shape(frames, dims*dims)
    frames, dims, dims = np.shape(video)
    full_img = video.reshape((frames,dims*dims))
    
    # Build and fit the model. Default: k=5
    algorithm = factorization.NMF(k=k)
    W, H = algorithm.fit(full_img)
    
    return W, H

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
