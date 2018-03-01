import numpy as np
#from sklearn.decomposition import NMF, IncrementalPCA
from factorization import NMF
#from extraction import NMF
from mahoney import io


def NMF_decomposition(path):
    '''This is normal NMF decomposition (using thunder)
        
        Args:
            path:
                The base path to the dataset.
                
        Returns:
            W: shape(frames, k)
            H: shape(k, pixels)
        '''
    # Load in the video and flatten each frame. Shape(frames, dims*dims)
    files = sorted(glob(f'{path}/images/image*.tiff'))
    full_img = array([imread(f) for f in files])
    frames, dims, dims = np.shape(full_img)
    full_img = full_img.reshape((frames,dims*dims))
    
    # Build and fit the model. Default: k=5
    algorithm = NMF()
    W, H = algorithm.fit(full_img)

    return W, H
