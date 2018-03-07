import mahoney.nmf as nmf
import mahoney.io as io
import mahoney.preprocess as preprocess
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
        path = f'./data/neurofinder.{vid}'
        video = io.load_video(path=path, preprocess=preprocess.normalize)
        rois = [{"coordinates": [list(k) for k in i]} for i in nmf.nmf_extraction(video, k=5)]
        tempdict["dataset"]=vid
        tempdict["regions"]=rois
        print(tempdict["regions"][0])
        output.append(tempdict)
    json.dumps(output)

def property_nmf():
    output = []
    for vid in TEST_SET:
        tempdict = {}
        path = f'./data/neurofinder.{vid}'
        video = io.load_video(path=path, imread=opencv, preprocess=preprocess.ed_open)
        rois = [{"coordinates": [list(k) for k in i]} for i in nmf.nmf_extraction(video, k=5)]
        tempdict["dataset"]=vid
        tempdict["regions"]=rois
        print(tempdict["regions"][0])
        output.append(tempdict)
    json.dumps(output)
