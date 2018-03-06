from mahoney.nmf_skl import NMF_dcomp and NMF_extract
from mahoney import preprocess
from mahoney import io
from scipy.misc import imread
from glob import glob
import numpy as np
from sklearn import GridSearchCV

model = NMF_extract(k=10)

TEST_SET = [
    '00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test',
    '02.01.test', '03.00.test', '04.00.test', '04.01.test'
]

for vid in TEST_SET:
    path = "../data/neurofinder{vid}"
