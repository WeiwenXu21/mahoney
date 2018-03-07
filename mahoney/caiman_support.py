#!/usr/bin/env python
# -*- coding: utf-8 -*-
#ipython demo_caiman_basic.py
"""
Modified version of some caiman API(s).
Previously only used for loading from video. We modified it to work directedly with ndarray.


Credits to: @agiovann and @epnev
For original version,
please visit: https://github.com/flatironinstitute/CaImAn/blob/master/caiman/mmapping.py
"""

import numpy as np
import caiman as cm
import os


def save_memmap_each(array,fnames, dview=None, base_name=None, resize_fact=(1, 1, 1), remove_init=0, idx_xy=None, xy_shifts=None, add_to_movie=0, border_to_0=0, order = 'C'):
    """
        Create several memory mapped files using parallel processing
        
        Parameters:
        -----------
        fnames: list of str
        list of path to the filenames
        
        dview: ipyparallel dview
        used to perform computation in parallel. If none it will be signle thread
        
        base_name str
        BaseName for the file to be creates. If not given the file itself is used
        
        resize_fact: tuple
        resampling factors for each dimension x,y,time. .1 = downsample 10X
        
        remove_init: int
        number of samples to remove from the beginning of each chunk
        
        idx_xy: slice operator
        used to perform slicing of the movie (to select a subportion of the movie)
        
        xy_shifts: list
        x and y shifts computed by a motion correction algorithm to be applied before memory mapping
        
        add_to_movie: float
        if movie too negative will make it positive
        
        border_to_0: int
        number of pixels on the border to set to the minimum of the movie
        
        Returns:
        --------
        fnames_tot: list
        paths to the created memory map files
        
        """
    pars = []
    if xy_shifts is None:
        xy_shifts = [None] * len(fnames)
    
    if type(resize_fact)is not list:
        resize_fact = [resize_fact] * len(fnames)
    
    for idx, f in enumerate(fnames):
        if base_name is not None:
            pars.append([array, f, base_name + '{:04d}'.format(idx), resize_fact[idx], remove_init,
                         idx_xy, order, xy_shifts[idx], add_to_movie, border_to_0])
        else:
            pars.append([f, os.path.splitext(f)[0], resize_fact[idx], remove_init, idx_xy, order,
                         xy_shifts[idx], add_to_movie, border_to_0])
                                                    
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            fnames_new = dview.map_async(save_place_holder, pars).get(4294967)
        else:
            fnames_new = dview.map_sync(save_place_holder, pars)
    else:
        fnames_new = list(map(save_place_holder, pars))
                                                                    
    return fnames_new

def save_place_holder(pars):
    """ To use map reduce
        """
    # todo: todocument
    
    (array,f, base_name, resize_fact, remove_init, idx_xy, order,
     xy_shifts, add_to_movie, border_to_0) = pars
    return save_memmap_new(array,[f], base_name=base_name, resize_fact=resize_fact, remove_init=remove_init,
                        idx_xy=idx_xy, order=order, xy_shifts=xy_shifts,
                        add_to_movie=add_to_movie, border_to_0=border_to_0)

def save_memmap_new(array,filenames, base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None,
                order='F', xy_shifts=None, is_3D=False, add_to_movie=0, border_to_0=0, dview=None,
                n_chunks=20,  async=False):
    
    """ Efficiently write data from a list of tif files into a memory mappable file
        
        Parameters:
        ----------
        filenames: list
        list of tif files or list of numpy arrays
        
        base_name: str
        the base used to build the file name. IT MUST NOT CONTAIN "_"
        
        resize_fact: tuple
        x,y, and z downsampling factors (0.5 means downsampled by a factor 2)
        
        remove_init: int
        number of frames to remove at the begining of each tif file
        (used for resonant scanning images if laser in rutned on trial by trial)
        
        idx_xy: tuple size 2 [or 3 for 3D data]
        for selecting slices of the original FOV, for instance
        idx_xy = (slice(150,350,None), slice(150,350,None))
        
        order: string
        whether to save the file in 'C' or 'F' order
        
        xy_shifts: list
        x and y shifts computed by a motion correction algorithm to be applied before memory mapping
        
        is_3D: boolean
        whether it is 3D data
        add_to_movie: floating-point
        value to add to each image point, typically to keep negative values out.
        Returns:
        -------
        fname_new: the name of the mapped file, the format is such that
        the name will contain the frame dimensions and the number of f
        
        """
    if type(filenames) is not list:
        raise Exception('input should be a list of filenames')
    
    if len(filenames)>1:
        is_inconsistent_order = False
        for file__ in filenames:
            if 'order_' + order not in file__:
                is_inconsistent_order = True
    
        if is_inconsistent_order:
            fname_new = cm.save_memmap_each(filenames,
                                            base_name=base_name,
                                            order=order,
                                            border_to_0=border_to_0,
                                            dview=dview,
                                            resize_fact=resize_fact,
                                            remove_init=remove_init,
                                            idx_xy=idx_xy,
                                            xy_shifts=xy_shifts,
                                            add_to_movie = add_to_movie)
        
        
        
        
        fname_new = cm.save_memmap_join(fname_new, base_name=base_name, dview=dview, n_chunks=n_chunks,  async=async)

    else:
        # TODO: can be done online
        Ttot = 0
        for idx, f in enumerate(filenames):
            if isinstance(f, str):
                print(f)
        
            if is_3D:
                #import tifffile
                #            print("Using tifffile library instead of skimage because of  3D")
                
                Yr = f if not(isinstance(f, basestring)) else tifffile.imread(f)
                if idx_xy is None:
                    Yr = Yr[remove_init:]
                elif len(idx_xy) == 2:
                    Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                else:
                    Yr = Yr[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]
            
            else:
                Yr = cm.movie(array,fr=30,start_time=0,file_name=f,meta_data=None)

                if xy_shifts is not None:
                    Yr = Yr.apply_shifts(xy_shifts, interpolation='cubic', remove_blanks=False)
                
                if idx_xy is None:
                    if remove_init > 0:
                        Yr = Yr[remove_init:]
                elif len(idx_xy) == 2:
                    Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                else:
                    raise Exception('You need to set is_3D=True for 3D data)')
                    Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]
                    
            if border_to_0 > 0:
                    
                min_mov = Yr.calc_min()
                Yr[:, :border_to_0, :] = min_mov
                Yr[:, :, :border_to_0] = min_mov
                Yr[:, :, -border_to_0:] = min_mov
                Yr[:, -border_to_0:, :] = min_mov
                    
            fx, fy, fz = resize_fact
            if fx != 1 or fy != 1 or fz != 1:
                        
                if 'movie' not in str(type(Yr)):
                    Yr = cm.movie(Yr, fr=1)
                                
                    Yr = Yr.resize(fx=fx, fy=fy, fz=fz)
                            
            T, dims = Yr.shape[0], Yr.shape[1:]
            Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
            Yr = np.reshape(Yr, (np.prod(dims), T), order='F')

            if idx == 0:
                fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(1 if len(dims) == 2 else dims[2]) + '_order_' + str(order)
                if isinstance(f, str):
                    fname_tot = os.path.join(os.path.split(f)[0], fname_tot)
                big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,shape=(np.prod(dims), T), order=order)
            else:
                big_mov = np.memmap(fname_tot, dtype=np.float32, mode='r+',shape=(np.prod(dims), Ttot + T), order=order)
                                                                                                             
            big_mov[:, Ttot:Ttot + T] = np.asarray(Yr, dtype=np.float32) + 1e-10 + add_to_movie
            big_mov.flush()
            del big_mov
            Ttot = Ttot + T

        fname_new = fname_tot + '_frames_' + str(Ttot) + '_.mmap'
        try:
            # need to explicitly remove destination on windows
            os.unlink(fname_new)
        except OSError:
            pass
        os.rename(fname_tot, fname_new)

    return fname_new











