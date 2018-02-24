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


class Neurofinder(Dataset):
    '''A high-level interface to the Neurofinder Challenge dataset.

    The Neurofinder dataset is divided into 19 train videos and 9 test videos.
    Each video taken from a single subject over a short period of time. These
    calcium imaging videos showing the action potentials of neurons. For each
    video, there is a single ground truth segmentation of the neurons.

    This dataset generates multiple data from each video by taking blocks of
    `n` contiguous fames. The first frame of each datum is separated by
    `step` frames from the first frame of the previous datum. Thus data
    overlap if `step < n`.

    Each datum is a dict with the following entries:
    - 'x': A dask array giving the frames in a datum.
    - 'y': The label mask or None if it is unknown.
    - 'meta': A dict of metadata read from `info.json` and more.

    The 'meta' dict contains the following:
    - All data from `info.json` for the corresponding video.
    - 'dataset': The Neurofinder code for the video, e.g. '01.00'.
    '''

    def __init__(self, base_path, n=1024, start=0, stop=3000, step=1,
            subset='train', imread=None, preprocess=None):
        '''Construct a dataset given a path to the data.

        Args:
            base_path:
                The path to the data. This must be a directory containing
                subdirectories with names `neurofinder.DATASET` where `DATASET`
                is a Neurofinder code, e.g. `01.00`.
            n:
                The number of frames in each datum.
            start:
                The index of the first frame of the first datum for each video.
            stop:
                The maximum number of data taken from each video.
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
                A function to apply to each frame.
        '''
        # `subset` may have the special values 'train', 'test', or 'all'
        # mapping to the TRAIN_SET, TEST_SET, or union of the two respectivly.
        if subset == 'train': subset = TRAIN_SET
        elif subset == 'test': subset = TEST_SET
        elif subset == 'all': subset = TRAIN_SET + TEST_SET

        # Collelct the data upfront. It is prefered to do this now rather than
        # when the data is requested because dask arrays are already managed
        # out of core memory.
        self.data = []
        for sub in subset:
            path = f'{base_path}/neurofinder.{sub}'

            x = io.load_instance(path, imread, preprocess)

            meta = io.load_metadata(path)
            meta['dataset'] = sub
            (height, width, frames) = meta['dimensions']

            try:
                y = io.load_mask(path, shape=(height, width))
            except FileNotFoundError:
                y = None

            stop = min(stop, len(x) - n + 1)
            for i in range(start, stop, step):
                self.data.append({
                    'x': x[i:i+n],
                    'y': y,
                    'meta': meta
                })

    def __len__(self):
        '''Returns the number of data in this dataset.
        '''
        return len(self.data)

    def __getitem__(self, i):
        '''Retrieves the ith datum from the dataset.
        '''
        return self.data[i]
