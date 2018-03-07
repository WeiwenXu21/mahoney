import torch
from torch.utils.data import Dataset

from mahoney import io


# Neurofinder codes for videos in the train set.
TRAIN_SET = [
    '00.00', '00.01', '00.02', '00.03', '00.04', '00.05', '00.06', '00.07',
    '00.08', '00.09', '00.10', '00.11', '01.00', '01.01', '02.00', '02.01',
    '03.00', '04.00', '04.01'
]


# Neurofinder codes for videos in the test set.
TEST_SET = [
    '00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test',
    '02.01.test', '03.00.test', '04.00.test', '04.01.test'
]


def load_dataset(base_path, subset='train', n=3000, *, frames=32, skip=1, step=1,
        imread=None, preprocess=None):
    '''A high-level interface to the Neurofinder Challenge dataset.

    The Neurofinder dataset is divided into 19 train videos and 9 test videos.
    Each video taken from a single subject over a short period of time. These
    calcium imaging videos showing the action potentials of neurons. For each
    video, there is a single ground truth segmentation of the neurons.

    This dataset generates multiple data from each video by taking blocks of
    `frames` contiguous fames. The first frame of each datum is separated by
    `step` frames from the first frame of the previous datum. Thus data
    overlap if `step < frames`.

    Args:
        base_path:
            The path to the data. This must be a directory containing
            subdirectories with names `neurofinder.DATASET` where `DATASET`
            is a Neurofinder code, e.g. `01.00`.
        n:
            The maximum number of data taken from each video.
        frames:
            The number of frames in each datum.
        skip:
            TODO: Document
        step:
            The number of frames between the first frame of consecutive
            datum from the same video.
        subset:
            The subset of data to consider. This is a collection of
            Neurofinder codes or one of the strings 'train', 'test', or
            'all' refering to the train set, test set, or entire set
            respectivly.
        imread:
            Override the function to used read images. The default is
            determined by dask, currently `skimage.io.imread`.
        preprocess:
            A function to apply to each video. The function takes a Dask
            array of shape (frames, width, height) and should return the
            processed video.

    Returns:
        Three lists: `x, y, metadata`. Same indces corresponds to the same instance.

        For each instance:
        - x is a dask array giving the video as (frames, width, height).
        - y is a label mask or None if it is unknown.
        - meta is a dict of metadata.

        A metadata dict contains the following:
        - All data from `info.json` for the instance.
        - 'dataset': The Neurofinder code for the video, e.g. '01.00'.
    '''
    # `subset` may have the special values 'train', 'test', or 'all'
    # mapping to the TRAIN_SET, TEST_SET, or union of the two respectivly.
    if subset == 'train': subset = TRAIN_SET
    elif subset == 'test': subset = TEST_SET
    elif subset == 'all': subset = TRAIN_SET + TEST_SET

    x = []
    y = []
    metadata = []
    for sub in subset:
        path = f'{base_path}/neurofinder.{sub}'

        # Read the metadata from the `info.json` and add additional info.
        # We need to know the height and width to generate the mask.
        meta = io.load_metadata(path)
        meta['dataset'] = sub
        (height, width, total_frames) = meta['dimensions']

        # Load the video and apply preprocessing to the whole.
        vid = io.load_video(path, imread)
        if preprocess is not None:
            vid = preprocess(vid)

        # Masks are not available for all datasets.
        try:
            mask = io.load_mask(path, shape=(height, width))
        except FileNotFoundError:
            mask = None

        # Extract instances from this video. We try to take `n` instances, but
        # if `n` is too large, `v` will be partial or even empty.
        for i in range(0, n*step, step):
            v = vid[i : i+frames*skip : skip]
            if len(v) == frames:
                x.append(v)
                y.append(mask)
                metadata.append(meta)

    return x, y, metadata


class Torchify(Dataset):
    '''Adapts a sklearn-style dask dataset to a torch-style dataset.

    A Torchify dataset zips a pair of lists into a list of pairs. If either
    half is a dask array (or any other object with a `compute` method) it is
    replaced by its computed value.

    This format works in conjunction with the PyTorch `DataLoader` class, an
    iterator with performance and convenience features for PyTorch based
    models.
    '''

    def __init__(self, x, y=None):
        '''Creates a torchified version of the sklearn dataset.

        If `y` is is falsy, The second halves will always be None.

        Args:
            x: A list of the first halves of the pairs.
            y: A list of the second halves of the pairs.
        '''
        if y is not None: assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        '''Returns the number of data in this dataset.
        '''
        return len(self.x)

    def __getitem__(self, i):
        '''Retrieves the ith datum from the dataset.
        '''
        x = self.x[i]
        y = self.y[i] if self.y is not None else None

        # Collect to numpy if we have Dask arrays
        if hasattr(x, 'compute'): x = x.compute()
        if hasattr(y, 'compute'): y = y.compute()

        # If `self.y` was not given, we have nothing to return.
        # Note that `y` may still be None because `self.y` may mix datasets
        # with and without labels.
        if self.y is not None:
            return x, y
        else:
            return x
