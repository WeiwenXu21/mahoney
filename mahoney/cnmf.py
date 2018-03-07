#!/usr/bin/env python
# -*- coding: utf-8 -*-
#ipython demo_caiman_basic.py
"""
Modified version of demo for running the CNMF source extraction algorithm with CaImAn and
evaluation the components.

For original version, please visit https://github.com/flatironinstitute/CaImAn/blob/master/demos_detailed/demo_caiman_basic.py

Credits to:
    Data courtesy of W. Yang, D. Peterka and R. Yuste (Columbia University)

    @authors: @agiovann and @epnev

"""


import numpy as np
import glob
import matplotlib.pyplot as plt
import caiman as cm
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.source_extraction.cnmf import cnmf as cnmf
from scipy.sparse import csc_matrix
import os
from mahoney import caiman_support as csp



rf = None           # setting these parameters to None
stride = None       # will run CNMF on the whole FOV
K = 500             # number of neurons expected (in the whole FOV)
gSig = [5, 5]       # expected half size of neurons
merge_thresh = 0.80 # merging threshold, max correlation allowed
p = 2               # order of the autoregressive system
gnb = 2             # global background order
fr = 10             # approximate frame rate of data
decay_time = 5.0    # length of transient
min_SNR = 2.5       # peak SNR for accepted components (if above this, acept)
rval_thr = 0.90     # space correlation threshold (if above this, accept)
use_cnn = True      # use the CNN classifier
min_cnn_thr = 0.10  # if cnn classifier predicts below this value, reject



def memmap_file(fname, video, dview):
    '''This is for changing data format into caiman data and saving into several memmap files.
        
        Args:
            fname:
                Name of the video (This is only used for working with caiman support)
                
            video:
                The input video
                shape: (frames, dims, dims)
                
            dview:
                Direct View object for parallelization pruposes when using ipyparallel
                
        Returns:
            Yr:
                Memory mapped variable
            
            dims:
                Dimension of one frame
                shape: (521,521)
                
            T:
                Number of frames
        '''
    video = np.squeeze(video)
    loaded = cm.movie(video,fr=30,start_time=0,file_name=fname[-1],meta_data=None)
    add_to_movie = -np.min(loaded).astype(float)
    add_to_movie = np.maximum(add_to_movie, 0)
    
    base_name = 'Yr'
    name_new = csp.save_memmap_each(video,fname, dview=dview, base_name=base_name,add_to_movie=add_to_movie)
    name_new.sort()
    
    fname_new = cm.save_memmap_join(name_new, base_name='Yr', dview=dview)
    Yr, dims, T = cm.load_memmap(fname_new)
    
    return Yr, dims, T

def correlation_image(images):
    '''Compute the correlation image for the input video
        
        Args:
            images:
                The input video
                shape: (frames, dims, dims)
        
        Returns:
            Cn:
                The correlation image
        '''
    
    Cn = cm.movie(images).local_correlations(swap_dim=False)
    return Cn

def components_eval(images,cnm,dview,dims):
    '''Evaluating the components being found
        
        Args:
            images:
                The input video
                shape: (frames, dims, dims)
            
            cnm:
                Object obtained from CNMF algorithm with C,A,S,b,f
            
            dview:
                Direct View object for parallelization pruposes when using ipyparallel
            
            dims:
                Dimension of each frame
                shape: (521, 521)
        
        Returns:
            idx_components:
                list of ids of components that are considered to be a neuron
            
        '''

    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = estimate_components_quality_auto(images, cnm.A, cnm.C,cnm.b, cnm.f,cnm.YrA, fr, decay_time, gSig, dims,dview=dview, min_SNR=min_SNR,r_values_min=rval_thr, use_cnn=use_cnn,thresh_cnn_lowest=min_cnn_thr)

    return idx_components



def compute_cnmf(n_processes,dview,images,dims):
    '''Applying CNMF algorithm on the video
        
        Args:
            n_processes:
                Number of parallel processing
                
            dview:
                Direct View object for parallelization pruposes when using ipyparallel
                
            images:
                The input video
                shape: (frames, dims, dims)
            
            dims:
                Dimension of each frame
                shape: (521, 521)
                
        Returns:
            cnm:
                Object obtained from CNMF algorithm with C,A,S,b,f
        
        '''
    cnm = cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig,
                merge_thresh=merge_thresh, p=p, dview=dview, gnb=gnb,
                rf=rf, stride=stride, rolling_sum=False)
    cnm = cnm.fit(images)

    return cnm

def caiman_cnmf(fname, video, test=False):
    '''Main console for the whole process of CNMF
        
        Args:
            fname:
                Name of the video (This is only used for working with caiman support)
            
            video:
                The input video
                
        Return:
            A:
                The list of neuron masks
                Each column is a mask of one neuron
                with pixels that belong to a neuron being non-zero
                and pixels that does not belong to a neuron being zero
                shape: (512*521, number of neurons being found)
        '''
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,single_thread=False)
    
    Yr, dims, T = memmap_file(fname, video, dview)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    if test:
        cm.movie(images).play(fr=50, gain=3.)

    Cn = correlation_image(images)

    if test:
        plt.imshow(Cn, cmap='gray')
        plt.title('Correlation Image')
        plt.show(block=True)

    cnm = compute_cnmf(n_processes,dview,images,dims)

    if test:
        plt.figure()
        crd = cm.utils.visualization.plot_contours(cnm.A, Cn, thr=0.9)
        plt.title('Contour plots of components')
        plt.show(block=True)

    idx_components = components_eval(images,cnm,dview,dims)

    if test:
        plt.figure()
        plt.title('Selected Components')
        cm.utils.visualization.view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components],
                                        cnm.C[idx_components, :], cnm.b, cnm.f,
                                        dims[0], dims[1],
                                        YrA=cnm.YrA[idx_components, :], img=Cn)
        plt.show(block=True)

    A = cnm.A.tocsc()[:, idx_components]
    A = csc_matrix(A).toarray()
    cm.stop_server()

    log_files = glob.glob('Yr*')
    for log_file in log_files:
        os.remove(log_file)
    return A













