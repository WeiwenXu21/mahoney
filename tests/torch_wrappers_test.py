from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as N
import torch.nn.functional as F

from mahoney.data import load_dataset
from mahoney.torch_wrappers import TorchEstimator, TorchSegmenter
from mahoney.unet import UNet


def test_regression():
    torch.set_default_tensor_type('torch.DoubleTensor')

    net_ctor = lambda: N.Linear(13, 1)
    loss = F.mse_loss

    # Supports fit, predict, and score
    x, y = load_boston(return_X_y=True)
    model = TorchEstimator(net_ctor, loss, opt_ctor='Adam', lr=1e-3)
    model.fit(x, y, epochs=5)
    model.predict(x)
    model.score(x, y)

    # Comparable to sklearn linear regression
    theirs = SGDRegressor(max_iter=5, eta0=1e-3)
    theirs.fit(x, y)
    h_theirs = theirs.predict(x)
    h_ours = model.predict(x)
    mse_theirs = mean_squared_error(y, h_theirs)
    mse_ours = mean_squared_error(y, h_ours)
    assert mse_ours < mse_theirs  # torch is better than sklearn by a lot


def test_segmentation():
    torch.set_default_tensor_type('torch.DoubleTensor')

    net_ctor = lambda: UNet(5, 2, depth=3, size=32)
    loss = F.cross_entropy

    x, y, meta = load_dataset('./data', subset=['01.00'], n=5, stop=1)
    x = [a.compute() for a in x]
    model = TorchSegmenter(net_ctor, loss)
    model.fit(x, y, epochs=1)
    model.predict(x)
    model.score(x, y)
