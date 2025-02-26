import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import h5py
import pandas as pd
import datetime
import warnings
import pickle
import os
from databroker import db

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _load_scan(scan_id, fill_events=False):
    '''Load scan from databroker by scan id'''

    #if scan_id > 0 and scan_id in data_cache:
    #    df = data_cache[scan_id]
    #else:
    #    hdr = db[scan_id]
    #    scan_id = hdr['start']['scan_id']
    #    if scan_id not in data_cache:
    #        data_cache[scan_id] = db.get_table(hdr, fill=fill_events)
    #    df = data_cache[scan_id]
    hdr = db[scan_id]
    scan_id = hdr['start']['scan_id']
    df = db.get_table(hdr,fill=fill_events)

    return scan_id, df


def get_xrf_data(scan_id):

    x=None
    y=None

    h = db[int(scan_id)]
    start_doc = db[scan_id]['start']
    mots = h.start['motors']
    df = h.table()

    if len(mots) == 2:
        dim1,dim2 = h.start['num1'], h.start['num2']

        if x is None:
            x = hdr['motor1']
            #x = hdr['motors'][0]
        x_data = np.asarray(df[x])

        if y is None:
            y = hdr['motor2']
            #y = hdr['motors'][1]
        y_data = np.asarray(df[y])

        extent = (np.nanmin(x_data), np.nanmax(x_data),
            np.nanmax(y_data), np.nanmin(y_data))

    elif len(mots) == 1:
        #dim1 = h.start['num1']

        if x is None:
            x = hdr['motor']
            #x = hdr['motors'][0]
        x_data = np.asarray(df[x])
        #extent = (np.nanmin(x_data), np.nanmax(x_data))

    xrf_cols_1 = sorted([col for col in df.columns if col.startswith('Det1')])
    xrf_cols_2 = sorted([col for col in df.columns if col.startswith('Det2')])
    xrf_cols_3 = sorted([col for col in df.columns if col.startswith('Det3')])
    
    xrfs_rois = df[xrf_cols_1].to_numpy()+df[xrf_cols_2].to_numpy()+df[xrf_cols_3].to_numpy()
    xrfs_rois_2d = xrfs_rois.reshape(dim1,dim2, -1).transpose(2,0,1)
    xrf_col_elem_names =[col_name[5:] for col_name in xrf_cols_1] 

    
    return xrf_col_elem_names, xrfs_rois_2d

