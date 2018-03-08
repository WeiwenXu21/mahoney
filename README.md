# Mahoney
Distributed neuron segmentation


## Deploying to Google Cloud Dataproc
Dataproc is a managed YARN cluster service provided by Google. We provide a template script for easily provisioning a Dataproc cluster, submitting a Mahoney job, and tearing down the cluster once the job completes.

To use the template script, make a copy of it and modify it for to your needs:

```shell
cp ./scripts/gcp-template.sh ./gcp.sh
vim ./gcp.sh
```

Take some time to read through the script. The script is primarily configured by a set of variables at the top. These all have sane, minimal defaults. The only variable that **must** be configured is the `BUCKET`, which should be set to a GCP bucket to which you have write access.

The script is designed to be launched from the root of this repository. To use a different working directory, update the relative paths accordingly.

Calling the script will submit a Mahoney job to a newly created Dataproc cluster. The arguments passed to the script are forwarded to Mahoney on the cluster.


## Running the tests

Mahoney's test suite uses `pytest` and requires that you have the `01.00` dataset at `./data/neurofinder.01.00`. You can download the datasets from the [Neurofinder page](http://neurofinder.codeneuro.org/).

```shell
# Download and extract the 01.00 dataset
curl https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.01.00.zip > neurofinder.01.00.zip
mkdir ./data
unzip ./neurofinder.01.00.zip -d ./data

# Install pytest
conda install pytest

# Run the tests
./setup.py test
```


## API: Interfacing with the data

We provide two modules for interfacing with the data.

The `mahoney.io` module contains low-level routines to load Neurofinder videos as dask arrays, to load region files as both structured objects and segmentation masks, and to load metadata files. This is the most convenient interface for interactive data exploration.

The `mahoney.data` module exports a higher-level interface. The `mahoney.data.Neurofinder` class provides a cohesive view of the entire dataset, out of core. It breaks each video into multiple, consistently shaped datum by considering the subvideos of a given length, and returns the corresponding metadata and labels (if available) simultaneously with each datum. The `mahoney.data.Torchify` adapter wraps a `Neurofinder` dataset into a form compatible with PyTorch's `DataLoader`, an iterator with high-performance and convenience features like shuffling, batching, prefetching, and CUDA pinned memory.

The `mahoney.preprocess` module contains functions that can be called in `mahoney.data.load_dataset` to preprocess the raw video. It includes options to normalize or open the video frames. Opening the video refers to erosion and dilation performed in succession.

## How to run an NMF experiment

After customizing your `gcp.sh`  and moving to the main directory folder, run `./gcp.sh [ARG]` where `[ARG]` can be either `std_nmf` which includes normalization as preprocessing or `property_nmf` which runs opening as preprocesing.

## Setup environment

If you are running this experiment on a Google cloud cluster, the relevant packages should already be installed with the bootstrap script. To install locally create a conda environment using the `REQUIREMENTS.txt` file for the list of packages: `conda create --name MY_NMF_ENV --file REQUIREMENTS.txt` Once you've created the conda environment, enter it using `source activate MY_NMF_ENV`. Once there you will also need to install the `thunder-extraction` package using `pip install thunder-extraction`. To be able to use the CaImAn based CNMF code you will need to additionally install the following in the environment:

**tqmd and ipyparallel**: `pip install tqdm ipyparallel`
**CaImAn**:
```
cd ..
git clone https://github.com/flatironinstitute/CaImAn
cd CaImAn/
python setup.py install
python setup.py build_ext -i
```
