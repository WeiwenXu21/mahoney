import json
from pathlib import Path

import numpy as np

import skimage as ski
import skimage.measure


def load_rois(path):
    '''Reads a regions-of-interest file into a Python object.

    Args:
        path: Path to the ROI file.

    Returns:
        A list of dicts, each mapping the key 'coordinates' to a list of pixel
        coordinates belonging to that region of interest.
    '''
    path = Path(path)
    with path.open() as fd:
        rois = json.load(fd)
    return rois


def rois_to_im(rois, shape=(512,512), reduce=True):
    '''Converts a regions-of-interest list into a segmentation mask.

    When `reduce=False`, pixels are assigned to a single region of interest.
    If multiple ROIs overlap, the pixel is assigned to the final ROI in which
    it appears.

    Args:
        rois:
            A regions-of-interest list of dicts. Each dict must map the key
            'coordinates' to a list of coordinate pairs for pixels belonging
            to that region of interest. This is the format given by the label
            JSON files.
        shape:
            Shape of the segmentation mask.
        reduce:
            If true, each region of interest is given the label 1.
            If false, each region of interest is given a distinct label.

    Returns:
        A segmentation mask.
    '''
    im = np.zeros(shape, dtype='int64')
    for i, roi in enumerate(rois):
        val = 1 if reduce else i+1
        coords = roi['coordinates']
        for y, x in coords:
            im[y, x] = val
    return im


def im_to_rois(im, reduce=True):
    '''Converts a segmentation mask into a regions-of-interest list.

    Args:
        im:
            The segmentation mask
        reduce:
            Set to true if the segmentation mask is reduced, meaning all
            regions of interest have the same label (1). Set to false if each
            region of interest has a unique label (consecutive integers
            starting at 1).

    Returns:
        A list of dicts, each mapping the key 'coordinates' to a list of pixel
        coordinates belonging to that region of interest.
    '''
    rois = []

    # If the mask is reduced, try to separate regions of interest.
    if reduce:
        im = ski.measure.label(im, connectivity=1)

    n = np.max(im)
    for i in range(n):
        coords = np.argwhere(im == i+1)
        rois.append({'coordinates': coords.tolist()})
    return rois
