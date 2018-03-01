import dask.array as da
import numpy as np

def normalize(x):
    ''' Normalize the video.

    This function takes in a whole video as a dask array
    and returns a dask array of the normalized video.

    Args:
    - 'x': A dask array representing a raw video.
    '''
    #Generate a mean image across time
    m = np.mean(x)
    std = np.std(x)
    x_norm = (x-m)/std

    #Create a matrix of variance of each pixel across time
    #Make a normalized dask array of the videos
    return x_norm
