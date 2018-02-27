import json

import dask.array as da
import dask.array.image

import numpy as np

import skimage as ski
import skimage.measure


def load_metadata(path):
    '''Reads the metadata file of a dataset.

    Args:
        path:
            The base path to the dataset.

    Returns:
        The contents of the `info.json` file, deserialized.
    '''
    with open(f'{path}/info.json') as fd:
        meta = json.load(fd)
    return meta


def load_instance(path, imread=None, preprocess=None):
    '''Reads a collection of image files into a dask array.

    Args:
        path:
            The base path to the dataset.
        imread:
            Override the function to used read images. The default is
            determined by dask, currently `skimage.io.imread`.
        preprocess:
            A function to apply to each image.

    Returns:
        A dask array with shape (N, H, W) where N is the number of images,
        H is the height of the images, and W is the width of the images.
    '''
    # The images are 16bit TIFF, which most applications don't expect.
    # So we convert to float automatically.
    x = da.image.imread(f'{path}/images/image*.tiff', imread, preprocess)
    x = x / (2 ** 16)
    return x


def load_mask(path, shape=(512,512)):
    '''Reads a regions-of-interest file into a segmentation mask.

    Args:
        path:
            The base path to the dataset.
        shape:
            Shape of the mask.

    Returns:
        A one-hot segmentation mask of shape (C, H, W) where C is the number of
        classes, H is the height of the image, and W is the width of the image.
        There are exactly two classes, background and foreground.
    '''
    rois = load_rois(path)
    mask = rois_to_mask(rois, shape=shape)
    return mask


def load_rois(path):
    '''Reads a regions-of-interest file into a Python object.

    Args:
        path:
            The base path to the dataset.

    Returns:
        A list of dicts, each mapping the key 'coordinates' to a list of pixel
        coordinates belonging to that region of interest.
    '''
    path = f'{path}/regions/regions.json'
    with open(path) as fd:
        rois = json.load(fd)
    return rois


def rois_to_mask(rois, shape=(512,512)):
    '''Converts a regions-of-interest list into a segmentation mask.

    Args:
        rois:
            A regions-of-interest list of dicts. Each dict must map the key
            'coordinates' to a list of coordinate pairs for pixels belonging
            to that region of interest. This is the format given by the label
            JSON files.
        shape:
            Shape of the mask.

    Returns:
        A one-hot segmentation mask of shape (C, H, W) where C is the number of
        classes, H is the height of the image, and W is the width of the image.
        There are exactly two classes, background and foreground.
    '''
    fg = np.zeros(shape, dtype='int64')
    for i, roi in enumerate(rois):
        coords = roi['coordinates']
        for y, x in coords:
            fg[y, x] = 1
    bg = (fg == 0).astype('int64')
    return np.stack([bg, fg])


def mask_to_rois(mask):
    '''Converts a segmentation mask into a regions-of-interest list.

    The segmentation mask must be an array of shape (C, H, W) where C is the
    number of classes, H is the height of the image, and W is the width of the
    image. There must be exactly two classes, background and foreground, in
    that order.

    The regions of interest are the connected components in the foreground.

    Args:
        mask:
            The segmentation mask.

    Returns:
        A list of dicts, each mapping the key 'coordinates' to a list of pixel
        coordinates belonging to that region of interest.
    '''
    # Separate the regions of interest from each other.
    mask = mask[1]  # Just use the foreground.
    mask = ski.measure.label(mask, connectivity=1)

    rois = []
    for i in range(np.max(mask)):
        coords = np.argwhere(mask == i+1)
        coords = coords.tolist()
        rois.append({'coordinates': coords})

    return rois
