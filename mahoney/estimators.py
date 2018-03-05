import logging
import sys
from pathlib import Path

import numpy as np

from sklearn.base import BaseEstimator

import torch
import torch.autograd as A
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
import torch.utils.data as D

from mahoney import data


logger = logging.getLogger(__name__)


class Mean:
    '''An accumulator to compute the mean from batches of values.
    '''

    def __init__(self, **kwargs):
        '''Initialize the accumulator.

        Kwargs:
            Forwarded to `batch.mean()` during accumulation.
        '''
        self.kwargs = kwargs
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        '''Add a new batch of data to the accumulator.

        Args:
            batch:
                The data being accumulated. The batch should be either a scalar
                or an object with a `batch.mean()` method to get the mean value
                of the batch.

        Returns:
            self
        '''
        if hasattr(batch, 'mean'):
            n = len(batch)
            val = batch.mean(**self.kwargs)
        elif hasattr(batch, 'double'):
            n = len(batch)
            val = batch.double().mean(**self.kwargs)
        else:
            n = 1
            val = batch

        if self.n == 0:
            self.n = n
            self.val = val

        else:
            delta = val - self.val
            self.n += n
            self.val += delta * n / self.n

        return self

    def reduce(self):
        '''Return the accumulated mean and reset to the initial state.
        '''
        val = self.val
        self.n = 0
        self.val = 0
        return val


class TorchEstimator(BaseEstimator):
    '''Wraps a torch network, optimizer, and loss to an sklearn-like estimator.
    '''

    def __init__(self, net_ctor, loss_fn, opt_ctor='SGD', cuda=None,
            name_prefix='model', dry_run=False):
        '''Create an estimator implemented by a torch module.

        Args:
            net_ctor:
                A constructor for the torch module to train. If cuda devices
                are used, the module is wrapped in a `DataParallel` module.
            loss_fn:
                The loss function to minimize.
            opt_ctor:
                A constructor for the optimizer to use for training.
                Alternativly, a string may be passed to represent a member of
                `torch.optim` with default hyperparameters.
            cuda:
                A list of cuda device ids to use. Defaults to all devices.
            name_prefix:
                A name for the estimator.
            dry_run:
                Cut loops short, useful for debugging.
        '''
        self.net_ctor = net_ctor
        self.loss_fn = loss_fn
        self.opt_ctor = opt_ctor
        self.cuda = cuda
        self.name_prefix = name_prefix
        self.dry_run = dry_run

    def module(self, **kwargs):
        '''Construct the estimator's torch module.
        '''
        net = self.net_ctor(**kwargs)
        if self.use_cuda:
            net = net.cuda()
            net = N.parallel.DataParallel(net, device_ids=self.cuda_devices)
        return net

    def loss(self, h, y, **kwargs):
        '''Applies the loss function.

        Args:
            h: The predicted targets.
            y: The true targets.

        Kwargs:
            Forwarded to the loss function.
        '''
        return self.loss_fn(h, y, **kwargs)

    def optimizer(self, module, **kwargs):
        '''Construct the estimator's optimizer.
        '''
        if isinstance(self.opt_ctor, str):
            ctor = getattr(O, self.opt_ctor)
        else:
            ctor = self.opt_ctor
        return ctor(module.parameters, **kwargs)

    def dataloader(self, x, y=None, **kwargs):
        '''Create a DataLoader from a sklearn-style dataset.

        Kwargs are forwarded to the torch `DataLoader` constructor and are
        documented there. The arguments documented here have defaults that
        deviate from those of the `DataLoader` constructor.

        Args:
            x:
                The data to iterate over.
            y:
                If given, it is zipped with `x`, e.g. the resulting dataloader
                iterates over `(x, y)` pairs.

        Kwargs:
            shuffle:
                Should the iterator shuffle the data. Defaults to True.
            pin_memory:
                Should the data be loaded into CUDA pinned memory.
                Defaults to True if this estimator uses any CUDA devices.
        '''
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', self.cuda)
        dl = Torchify(x, y)
        dl = D.DataLoader(dl, **kwargs)
        return dl

    @property
    def cuda_devices(self):
        '''A list of the cuda devices used by this estimator.
        '''
        if self.cuda is None:
            return tuple(range(torch.cuda.device_count()))
        else:
            return self.cuda

    @property
    def use_cuda(self):
        '''Indicates if cuda is being used.
        '''
        return len(self.cuda_devices) > 0

    @property
    def name(self):
        '''A unique name for this estimator.
        '''
        # The collection of hyperparameters that uniquely identify this model.
        # TODO: This must up to date! What's the best ux for devs to do this?
        params = (self.net_ctor, self.loss_fn, self.opt_ctor)

        # The name is the prefix and the hash of the hyperparameters.
        h = hash(params)
        h = h + 1 << 64  # make unsigned
        h = hex(h)[2:].rjust(16, '0')  # turn into a hex string
        return f'{name_prefix}_{h}'

    def variable(self, x, **kwargs):
        '''Cast a tensor to a `Variable` on the same device as the network.

        If the input is already a `Variable`, it is not wrapped,
        but it may be copied to a new device.

        Args:
            x: The tensor to wrap.

        Kwargs:
            Forwarded to the `autograd.Variable` constructor.

        Returns:
            An `autograd.Variable` on the same device as the network.
        '''
        if not isinstance(x, A.Variable):
            x = A.Variable(x, **kwargs)
        if self.use_cuda:
            x = x.cuda(async=True)
        return x

    def save(self, path=None):
        '''Saves the model parameters to disk.

        The default path is based on the name of the estimator.

        Args:
            path: The path to write into.

        Returns:
            Returns `self` to allow method chaining.
        '''
        if path is None:
            path = Path(f'./checkpoints/{self.name}.torch')
        logger.debug(f'saving {self.name} to {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize(True)
        state = self.net_.state_dict()
        torch.save(state, str(path))
        return self

    def load(self, path=None):
        '''Loads the model parameters from disk.

        The default path is based on the name of the estimator.

        Args:
            path: The path to write into.

        Returns:
            Returns `self` to allow method chaining.
        '''
        if path is None:
            path = path = Path(f'./checkpoints/{self.name}.torch')
        logger.debug(f'restoring {self.name} from {path}')
        self.initialize(True)
        state = torch.load(str(path))
        self.net_.load_state_dict(state)
        return self

    def initialize(self, warm_start=False):
        '''Initializes the module and optimizer.
        '''
        if warm_start and hasattr(self, 'net_'): return
        self.net_ = self.module()
        self.opt_ = self.opt_ctorimizer(self.net_)

    def fit(self, x, y, validation=None, epochs=100, patience=None, warm_start=False, **kwargs):
        '''Fit the model to a dataset.

        Args:
            x:
                The data to fit.
            y:
                The labels to fit.
            validation:
                A dataset to use as the validation set.
            epochs:
                The maximum number of epochs to spend training.
            patience:
                Stop if the loss does not improve after this many epochs.
                A negative or falsy value means infinite patience.
            warm_start:
                If true, start from existing learned parameters if any.

        Kwargs:
            Forwarded to `TorchEstimator.dataloader`.

        Returns:
            Returns the validation loss if a validation set is given.
            Returns the train loss otherwise.
        '''
        self.initialize(warm_start)
        ds = self.dataloader(x, y, **kwargs)
        best_loss = float('inf')
        patience = patience or -1
        p = patience

        # Training
        for epoch in range(epochs):
            n = len(ds)
            train_loss = Mean()
            progress = 0
            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=sys.stderr)
            for batch_x, batch_y in ds:
                j = self.partial_fit(batch_x, batch_y)
                train_loss.accumulate(j)
                progress += 1 / n
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=sys.stderr)
                if self.dry_run: break

            # Reporting
            train_loss = train_loss.reduce()
            print('\001b[2K', end='\r', flush=True, file=sys.stderr)  # magic to clear the line
            print(f'epoch {epoch+1}', end=' ', flush=True)
            print(f'[train loss: {train_loss:8.6f}]', end=' ', flush=True)

            # Validation
            if validation:
                val_x, val_y = validation
                val_loss = self.score(val_x, val_y, invert=False, **kwargs)
                print(f'[validation loss: {val_loss:8.6f}]', end=' ', flush=True)

            # Early stopping
            loss = val_loss if validation else train_loss
            if loss < best_loss:
                best_loss = loss
                p = patience
                self.save()
                print('✓')
            else:
                p -= 1
                print()
            if p == 0:
                break

        # Revert to best model if using early stopping.
        if patience > 0:
            self.load()

        return self

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The targets.

        Returns:
            Returns the average loss for this batch.
        '''
        self.net_.train()
        self.opt_.zero_grad()
        x = self.variable(x)
        y = self.variable(y)
        h = self.net_(x)
        j = self.loss(h, y)
        self.opt_.step()
        return j.data.mean()

    def predict(self, x, **kwargs):
        '''Apply the network to some input batch.

        This method simply delegates out to `self._predict` which should return
        a list of torch Tensors, one for each batch. The tensors are then
        concatenated and cast to numpy arrays.

        The default `_predict` delegates out to the module's `forward` method.

        Args:
            x: The input batch.

        Kwargs:
            Forwarded to `TorchEstimator.dataloader`.

        Returns:
            Returns the output of the network as a numpy array.
        '''
        y = self._predict(x, **kwargs)
        y = torch.cat(y)
        return y.cpu().numpy()

    def _predict(self, x, **kwargs):
        '''The torch-side implementation of `predict`.

        Returns:
            Returns the output of the network as a list torch Tensors.
        '''
        # We MUST iterate in order.
        kwargs['shuffle'] = False
        kwargs['sampler'] = None
        kwargs['batch_sampler'] = None

        y = []
        self.net_.eval()
        for batch in  self.dataloader(x, **kwargs):
            batch = self.variable(batch, volatile=True)
            predicted = self.net_(batch)
            y.append(predicted)
        return y

    def score(self, x, y, invert=True, **kwargs):
        '''Returns the inverse of the average loss.

        Args:
            x: The input data.
            y: The targets.
            invert: If False, return the average loss.

        Kwargs:
            Forwarded to `TorchEstimator.dataloader`.
        '''
        loss = Mean()
        self.net_.eval()
        for x, y in self.dataloader(x, y, **kwargs):
            x = self.variable(x)
            y = self.variable(y)
            h = self.net_(x)
            j = self.loss(h, y, reduce=False)
            loss.accumulate(j)
        loss = loss.reduce()
        if invert: loss = 1 / loss
        return loss.cpu().numpy()


class TorchClassifier(TorchEstimator):
    '''Wraps a torch network, optimizer, and loss to an sklearn-like classifier.
    '''

    def predict_proba(self, x, **kwargs):
        '''Compute the likelihoods for some input batch.

        Args:
            x: The input batch.

        Kwargs:
            Forwarded to `TorchEstimator.dataloader`.

        Returns:
            Returns the softmax of the network as a numpy array.
        '''
        y = self._predict_proba(x, **kwargs)
        y = torch.cat(y)
        return y.cpu().numpy()

    def _predict_proba(self, x, **kwargs):
        '''The torch-side implementation of `predict_proba`.

        Returns:
            Returns the softmax of the network as a list torch Tensors.
        '''
        y = super()._predict(x, **kwargs)
        for i, batch in enumerate(y):
            batch = F.softmax(batch, 1)
            y[i] = batch
        return y

    def _predict(self, x, **kwargs):
        '''Override `_predict` to return class labels.

        Returns:
            Returns the argmax of the network as a list torch Tensors.
        '''
        y = super()._predict(x, **kwargs)
        for i, batch in enumerate(y):
            _, batch = torch.max(batch, 1)
            y[i] = batch
        return y


class TorchSegmenter(TorchEstimator):
    '''Torch loss functions don't work on images by default.

    This extends `TorchEstimator` to image segmentation.
    '''
    def loss(self, h, y, **kwargs):
        kwargs.setdefault('reduce', False)

        (count, channels, width, height) = h.shape
        h = h.permute(0, 2, 3, 1)
        h = h.view(count * width * height, channels)

        (count, width, height) = y.shape
        y = y.view(count * width * height)

        j = self.loss_fn(h, y, **kwargs)
        if not kwargs['reduce']:
            j = j.view(count, width, height)
        return j
