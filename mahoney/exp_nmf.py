import mahoney.nmf as nmf
import mahoney.io as io
import mahoney.preprocess as preprocess
import mahoney.data as data
from scipy.misc import imread
from glob import glob
import numpy as np
import json

TEST_SET = [
    '00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test',
    '02.01.test', '03.00.test', '04.00.test', '04.01.test'
]

def std_nmf(cloud=False):
    output = []
    for vid in TEST_SET:
        tempdict = {}
        video, y, meta = data.load_dataset(base_path='./data', subset=[vid], preprocess=preprocess.ed_open, frames=1000)
        rois = [{"coordinates": [list(k) for k in i]} for i in nmf.nmf_extraction(np.dstack(video[0]), k=10)]
        tempdict["dataset"]=vid
        tempdict["regions"]=rois
        print(tempdict["regions"][0])
        output.append(tempdict)
    str_out = json.dumps(output, default=lambda x: int(x))
    print(str_out)

def property_nmf():
    output = []
    for vid in [TEST_SET]:
        tempdict = {}
        video, y, meta = data.load_dataset(base_path='./data', subset=[vid], preprocess=preprocess.ed_open, frames=1000)
        rois = [{"coordinates": [list(k) for k in i]} for i in nmf.nmf_extraction(np.dstack(video[0]), k=10)]
        tempdict["dataset"]=vid
        tempdict["regions"]=rois
        output.append(tempdict)
    str_out = json.dumps(output, default=lambda x: int(x))
    print(str_out)
