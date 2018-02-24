import json

import dask.array as da
import dask.array.image

import numpy as np

import skimage as ski
import skimage.measure


TRAIN_SETS = [
    '00.00', '00.01', '00.02', '00.03', '00.04', '00.05', '00.06', '00.07',
    '00.08', '00.09', '00.10', '00.11', '01.00', '01.01', '02.00', '02.01',
    '03.00', '04.00', '04.01'
]


TEST_SETS = [
    '00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test',
    '02.01.test', '03.00.test', '04.00.test', '04.01.test'
]


def load_instance(path, imread=None, preprocess=None):
    '''Reads a collection of image files into a dask array.

    Args:
        path:
            The path to the image directory. It should contain files matching
            the glob 'image*.tiff' and all images should be the same shape.
        imread:
            Override the function to used read images. The default is
            determined by dask, currently `skimage.io.imread`.
        preprocess:
            A function to apply to each image.

    Returns:
        A dask array with shape (N, H, W) where N is the number of images,
        H is the height of the images, and W is the width of the images.
    '''
    im = da.image.imread(path + '/image*.tiff', imread, preprocess)
    return im


def load_rois(path):
    '''Reads a regions-of-interest file into a Python object.

    Args:
        path: Path to the ROI file.

    Returns:
        A list of dicts, each mapping the key 'coordinates' to a list of pixel
        coordinates belonging to that region of interest.
    '''
    with open(path) as fd:
        rois = json.load(fd)
    return rois


def rois_to_im(rois, shape=(512,512)):
    '''Converts a regions-of-interest list into a segmentation mask.

    Args:
        rois:
            A regions-of-interest list of dicts. Each dict must map the key
            'coordinates' to a list of coordinate pairs for pixels belonging
            to that region of interest. This is the format given by the label
            JSON files.
        shape:
            Shape of the image.

    Returns:
        An array of shape (C, H, W) where C is the number of classes, H is the
        height of the image, and W is the width of the image. There are exactly
        two classes, background and foreground, in that order.
    '''
    fg = np.zeros(shape, dtype='float64')
    for i, roi in enumerate(rois):
        coords = roi['coordinates']
        for y, x in coords:
            fg[y, x] = 1
    bg = (fg == 0).astype('float64')
    return np.stack([bg, fg])


def im_to_rois(im):
    '''Converts a segmentation mask into a regions-of-interest list.

    The segmentation mask must be an array of shape (C, H, W) where C is the
    number of classes, H is the height of the image, and W is the width of the
    image. There must be exactly two classes, background and foreground, in
    that order.

    The regions of interest are the connected components in the foreground.

    Args:
        im:
            The segmentation mask.

    Returns:
        A list of dicts, each mapping the key 'coordinates' to a list of pixel
        coordinates belonging to that region of interest.
    '''
    # Separate the regions of interest from each other.
    im = im[1]  # Just use the foreground.
    im = ski.measure.label(im, connectivity=1)

    rois = []
    for i in range(np.max(im)):
        coords = np.argwhere(im == i+1)
        coords = coords.tolist()
        rois.append({'coordinates': coords})

    return rois
