
import os
import warnings
import glob
import h5py
import pandas as pd
import datetime
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import tifffile as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


db = None
if  os.getlogin().startswith("xf03"):

    sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
    from hxntools.CompositeBroker import db
    from hxntools.scan_info import get_scan_positions


det_params = {'merlin1':55, "merlin2":55, "eiger2_images":75}


def get_path(scan_id, key_name='merlin1', db=db):
    """Return file path with given scan id and keyname.
    """
    
    h = db[int(scan_id)]
    e = list(db.get_events(h, fields=[key_name]))
    #id_list = [v.data[key_name] for v in e]
    id_list = [v['data'][key_name] for v in e]
    rootpath = db.reg.resource_given_datum_id(id_list[0])['root']
    flist = [db.reg.resource_given_datum_id(idv)['resource_path'] for idv in id_list]
    flist = set(flist)
    fpath = [os.path.join(rootpath, file_path) for file_path in flist]
    return fpath

def get_flyscan_dimensions(hdr):
    
    if 'dimensions' in hdr.start:
        return hdr.start['dimensions']
    else:
        return hdr.start['shape']

def get_all_scalar_data(hdr):

    keys = list(hdr.table().keys())
    scalar_keys = [k for k in keys if k.startswith('sclr1') ]
    print(f"{scalar_keys = }")
    scan_dim = get_flyscan_dimensions(hdr)
    scalar_stack_list = []

    for sclr in sorted(scalar_keys):
        
        scalar = np.array(list(hdr.data(sclr))).squeeze()
        sclr_img = scalar.reshape(scan_dim)
        scalar_stack_list.append(sclr_img)

    # Stack all the 2D images along a new axis (axis=0).
    scalar_stack = np.stack(scalar_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)

    return  scalar_stack, sorted(scalar_keys),

def get_all_xrf_roi_data(hdr):


    channels = [1, 2, 3]
    keys = list(hdr.table().keys())
    roi_keys = [k for k in keys if k.startswith('Det')]
    det1_keys = [k for k in keys if k.startswith('Det1')]
    elem_list = [k.replace("Det1_", "") for k in det1_keys]

    print(f"{elem_list = }")

    scan_dim = get_flyscan_dimensions(hdr)
    xrf_stack_list = []

    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        xrf_img = spectrum.reshape(scan_dim)
        xrf_stack_list.append(xrf_img)

    # Stack all the 2D images along a new axis (axis=0).
    xrf_stack = np.stack(xrf_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)
    return xrf_stack, sorted(elem_list)

def get_sid_list(str_list, interval):
    num_elem = np.size(str_list)
    for i in range(num_elem):
        str_elem = str_list[i].split('-')
        if i == 0:
            if np.size(str_elem) == 1:
                tmp = int(str_elem[0])
            else:
                tmp = np.arange(int(str_elem[0]),int(str_elem[1])+1,interval)
            sid_list = np.reshape(tmp,(-1,))
        else:
            if np.size(str_elem) == 1:
                tmp = int(str_elem[0])
            else:
                tmp = np.arange(int(str_elem[0]),int(str_elem[1])+1,interval)
            tmp = np.reshape(tmp,(-1,))
            sid_list = np.concatenate((sid_list,tmp))
    return sid_list

def get_scan_details(sid = -1):
    param_dict = {"scan_id":int(sid)}
    h = db[int(sid)]
    df = db.get_table(h,stream_name = "baseline")
    start_doc = h.start
    mots = start_doc['motors']

    # Create a datetime object from the Unix time.
    datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
    formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
    param_dict["time"] = formatted_time
    param_dict["motors"] = start_doc["motors"]
    if "detectors" in start_doc.keys():
        param_dict["detectors"] = start_doc["detectors"]
        param_dict["scan_start1"] = start_doc["scan_start1"]
        param_dict["num1"] = start_doc["num1"]
        param_dict["scan_end1"] = start_doc["scan_end1"]
        
        if len(mots)==2:

            param_dict["scan_start2"] = start_doc["scan_start2"]
            param_dict["scan_end2"] = start_doc["scan_end2"]
            param_dict["num2"] = start_doc["num2"]
        param_dict["exposure_time"] = start_doc["exposure_time"]

    elif "scan" in start_doc.keys():
        param_dict["scan"] = start_doc["scan"]
    
    param_dict["zp_theta"] = np.round(df.zpsth.iloc[0],3)
    param_dict["mll_theta"] = np.round(df.dsth.iloc[0],3)
    param_dict["energy"] = np.round(df.energy.iloc[0],3)
    return param_dict

def export_scan_details(sid_list, wd):

    for sid in tqdm(sid_list):
        export_scan_metadata(sid, wd)

def get_scan_metadata(sid):
    
    output = db.get_table(db[int(sid)],stream_name = "baseline")
    df_dictionary = pd.DataFrame([get_scan_details(sid = int(sid))])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    return output


def export_scan_metadata(sid, wd):
    output = db.get_table(db[int(sid)],stream_name = "baseline")
    df_dictionary = pd.DataFrame([get_scan_details(sid = int(sid))])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    sid_ = df_dictionary['scan_id']
    save_as = os.path.join(wd,f"{sid_}_metadata.csv")
    output.to_csv(save_as,index=False)
    print(f"{save_as = }")
    

def load_ims(file_list):
    # stacking is along the first axis
    num_ims = np.size(file_list)
    for i in tqdm(range(num_ims),desc="Progress"):
        file_name = file_list[i]
        im = tf.imread(file_name)
        im_row, im_col = np.shape(im)
        if i == 0:
            im_stack = np.reshape(im,(1,im_row,im_col))
        else:
            #im_stack_num = i 
            im_stack_num, im_stack_row,im_stack_col = np.shape(im_stack)
            row = np.maximum(im_row,im_stack_row)
            col = np.maximum(im_col,im_stack_col)
            if im_row < im_stack_row:
                r_s = np.round((im_stack_row-im_row)/2)
            else:
                r_s = 0
            if im_col < im_stack_col:
                c_s = np.round((im_stack_col-im_col)/2)
            else:
                c_s = 0
            im_stack_tmp = np.zeros((im_stack_num+1,row,col))
            im_stack_tmp[0:im_stack_num,0:im_stack_row,0:im_stack_col] = im_stack
            
            im_stack_tmp[im_stack_num,r_s:im_row+r_s,c_s:im_col+c_s] = im
            im_stack = im_stack_tmp
    return im_stack

def load_txts(file_list):
    # stacking is along the first axis
    num_ims = np.size(file_list)
    for i in tqdm(range(num_ims),desc="Progress"):
        file_name = file_list[i]
        im = np.loadtxt(file_name)
        im_row, im_col = np.shape(im)
        if i == 0:
            im_stack = np.reshape(im,(1,im_row,im_col))
        else:
            #im_stack_num = i 
            im_stack_num, im_stack_row,im_stack_col = np.shape(im_stack)
            row = np.maximum(im_row,im_stack_row)
            col = np.maximum(im_col,im_stack_col)
            if im_row < im_stack_row:
                r_s = np.round((im_stack_row-im_row)/2)
            else:
                r_s = 0
            if im_col < im_stack_col:
                c_s = np.round((im_stack_col-im_col)/2)
            else:
                c_s = 0
            im_stack_tmp = np.zeros((im_stack_num+1,row,col))
            im_stack_tmp[0:im_stack_num,0:im_stack_row,0:im_stack_col] = im_stack
            
            im_stack_tmp[im_stack_num,r_s:im_row+r_s,c_s:im_col+c_s] = im
            im_stack = im_stack_tmp
    return im_stack

def create_file_list(data_path, prefix, postfix, sid_list):
    num = np.size(sid_list)
    file_list = []
    for sid in sid_list:
        tmp = ''.join([data_path, prefix,'{}'.format(sid),postfix])
        file_list.append(tmp)
    return file_list



def load_h5_data(file_list, roi, mask):
    # load a list of scans, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'
    
    num_scans = np.size(file_list)
    det = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        file_name = file_list[i]
        f = h5py.File(file_name,'r')       
        if mask is None:
            data = f[det]
        else:
            data = f[det]*mask
        if roi is None:
            data = np.flip(data[:,:,:],axis = 1)
        else:
            data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
        if i == 0:
            raw_size = np.shape(f[det])
            print("Total scan points: {}; raw image row: {}; raw image col: {}".format(raw_size[0],raw_size[1],raw_size[2]))
            data_size = np.shape(data)
            print("Total scan points: {}; data image row: {}; data image col: {}".format(data_size[0],data_size[1],data_size[2]))
            diff_data = np.zeros(np.append(num_scans,np.shape(data)),dtype=data_type)
        sz = diff_data.shape    
        diff_data[i] = np.resize(data,(sz[1],sz[2],sz[3])) # in case there are lost frames
    if  num_scans == 1: # assume it is a rocking curve scan
        diff_data = np.swapaxes(diff_data,0,1) # move angle to the first axis
        print("Assume it is a rocking curve scan; number of angles = {}".format(diff_data.shape[0]))
    return diff_data  


def return_diff_array(sid, det="eiger2_image", mon="sclr1_ch4", threshold=None):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    
    data_type = 'float32'
    data_name = '/entry/instrument/detector/data'

    #skip 1d

    hdr = db[int(sid)]
    start_doc = hdr["start"]
    if not start_doc["plan_type"] in ("FlyPlan1D",):

        file_name = get_path(sid,det)
        print(file_name)
        num_subscan = len(file_name)
        
        if num_subscan == 1:
            f = h5py.File(file_name[0],'r') 
            data = np.asarray(f[data_name],dtype=data_type)
            #data = np.asarray(f[data_name])
            print(data.shape)
        else:
            sorted_files = sort_files_by_creation_time(file_name)
            ind = 0
            for name in sorted_files:
                f = h5py.File(name,'r')
                if ind == 0:
                    data = np.asarray(f[data_name],dtype=data_type)
                else:   
                    data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                ind = ind + 1
                print(data.shape)

        raw_size = np.shape(data)
        if threshold is not None:
            data[data<threshold[0]] = 0
            data[data>threshold[1]] = 0
        
        if mon is not None:
            mon_data = db[sid].table()[mon]
            ln = np.size(mon_data)
            mon_array = np.zeros(ln,dtype=data_type)
            for n in range(1,ln):
                mon_array[n] = mon_data[n] 
            avg = np.mean(mon_array[mon_array != 0])
            mon_array[mon_array == 0] = avg
                
            #misssing frame issue

            if len(mon_array) != data.shape[0]:
                if len(mon_array) > data.shape[0]:
                    last_data_point = data[-1]  # Last data point along the first dimension
                    last_data_point = last_data_point[np.newaxis, :,:]  
                    data = np.concatenate((data, last_data_point), axis=0)
                else:
                    last_mon_array_element = mon_array[-1]
                    mon_array = np.concatenate((mon_array, last_mon_array_element), axis=0)            

            data = data/mon_array[:,np.newaxis,np.newaxis]
            

    return data

def export_diff_data_as_tiff(first_sid,last_sid, det="eiger2_image", mon="sclr1_ch4", roi=None, mask=None, threshold=None, wd = '.', norm_with_ic = True):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    
    sid_list = np.arange(first_sid,last_sid+1)
    
    data_type = 'float32'
  
    num_scans = np.size(sid_list)
    data_name = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        sid = int(sid_list[i])
        print(f"{sid = }")

        #skip 1d

        hdr = db[int(sid)]
        start_doc = hdr["start"]
        scan_table = hdr.table()
        if not start_doc["plan_type"] in ("FlyPlan1D",):

            file_name = get_path(sid,det)
            print(file_name)
            num_subscan = len(file_name)
            
            if num_subscan == 1:
                f = h5py.File(file_name[0],'r') 
                data = np.asarray(f[data_name],dtype=data_type)
                #data = np.asarray(f[data_name])
                print(data.shape)
            else:
                sorted_files = sort_files_by_creation_time(file_name)
                ind = 0
                for name in sorted_files:
                    f = h5py.File(name,'r')
                    if ind == 0:
                        data = np.asarray(f[data_name],dtype=data_type)
                    else:   
                        data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                    ind = ind + 1
                    print(data.shape)
                #data = list(db[sid].data(det))
                #data = np.asarray(np.squeeze(data),dtype=data_type)
            raw_size = np.shape(data)
            if threshold is not None:
                data[data<threshold[0]] = 0
                data[data>threshold[1]] = 0
            if mon is not None:
                mon_data = db[sid].table()[mon]
                ln = np.size(mon_data)
                mon_array = np.zeros(ln,dtype=data_type)
                for n in range(1,ln):
                    mon_array[n] = mon_data[n] 
                avg = np.mean(mon_array[mon_array != 0])
                mon_array[mon_array == 0] = avg
                
            #misssing frame issue

            if len(mon_array) != data.shape[0]:
                if len(mon_array) > data.shape[0]:
                    last_data_point = data[-1]  # Last data point along the first dimension
                    last_data_point = last_data_point[np.newaxis, :,:]  
                    data = np.concatenate((data, last_data_point), axis=0)
                else:
                    last_mon_array_element = mon_array[-1]
                    mon_array = np.concatenate((mon_array, last_mon_array_element), axis=0)            
            if norm_with_ic:
                data = data/mon_array[:,np.newaxis,np.newaxis]
            
            if mask is not None:     
                #sz = data.shape
                data = data*mask
            if roi is None:
                data = np.flip(data[:,:,:],axis = 1)
            else:
                data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
                
            print(f"data size = {data.size/1_073_741_824 :.2f} GB")
            save_folder =  os.path.join(wd,f"{sid}_diff_data")   

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                

            saved_as = os.path.join(save_folder,f"{sid}_diff_{det}.tiff")
            tf.imwrite(saved_as, data, dtype = np.float32)
            export_scan_metadata(sid,save_folder)
            scan_table.to_csv(os.path.join(save_folder,f"{sid}_scan_table.csv"))
            print(f"{saved_as =}")
            
def export_diff_data_as_h5(sid_list, 
                           det="eiger2_image", 
                           wd = '.', 
                           mon = 'sclr1_ch4',
                           compression = 'gzip'):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    
    data_type = 'float32'
    if isinstance(sid_list, (int, float)):
        sid_list = [sid_list]

  
    num_scans = np.size(sid_list)
    data_name = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        sid = int(sid_list[i])
        print(f"{sid = }")

        #skip 1d

        hdr = db[int(sid)]
        start_doc = hdr["start"]
        sid = start_doc["scan_id"]
        
        if 'num1' and 'num2' in start_doc:
            dim1,dim2 = start_doc['num1'],start_doc['num2']
        elif 'shape' in start_doc:
            dim1,dim2 = start_doc.shape
        try:
            xy_scan_positions = list(np.array(df[mots[0]]),np.array(df[mots[1]]))
        except:
            xy_scan_positions = list(get_scan_positions(hdr))

        scan_table = get_scan_metadata(int(sid))
        if not start_doc["plan_type"] in ("FlyPlan1D",):

            file_name = get_path(sid,det)
            num_subscan = len(file_name)
            
            if num_subscan == 1:
                f = h5py.File(file_name[0],'r') 
                data = np.asarray(f[data_name],dtype=data_type)
            else:
                sorted_files = sort_files_by_creation_time(file_name)
                ind = 0
                for name in sorted_files:
                    f = h5py.File(name,'r')
                    if ind == 0:
                        data = np.asarray(f[data_name],dtype=data_type)
                    else:   
                        data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                    ind = ind + 1
                #data = list(db[sid].data(det))
                #data = np.asarray(np.squeeze(data),dtype=data_type)
            _, roi1,roi2 = np.shape(data)
            if mon != None:
                mon_array = np.array(list(hdr.data(str(mon)))).squeeze()
            data = np.flip(data[:,:,:],axis = 1)

                
            print(f"data size = {data.size/1_073_741_824 :.2f} GB")

            #save_folder =  os.path.join(wd,f"{sid}_diff_data")   

            #if not os.path.exists(save_folder):
                #os.makedirs(save_folder)

            if wd:
                save_folder = wd
                
            saved_as = os.path.join(save_folder,f"scan_{sid}_{det}")

            f.close()

            
            with h5py.File(saved_as+'.h5','w') as f:

                # Create (or get) the group for the detector raw data
                det_group = f.require_group(f"/diff_data/{det}/")
                det_group.create_dataset(
                    "raw_data",
                    data=data.reshape(dim1, dim2, roi1, roi2),
                    compression=compression
                )
                if mon != None:
                    det_group.create_dataset(
                        "Io",
                        data=mon_array.reshape(dim1, dim2)
                    )
                
                # Create (or get) the group for scan data
                scan_group = f.require_group("scan/")
                scan_group.create_dataset(
                    "scan_positions",
                    data=xy_scan_positions
                )
                # Store scan_parameters dictionary
                scan_params = get_scan_details
                for key, value in scan_params.items():
                    if isinstance(value, dict):  # Handle nested dictionaries
                        # Create a subgroup for the nested dictionary
                        param_group = scan_group.create_group(key)
                        for nested_key, nested_value in value.items():
                            param_group.create_dataset(nested_key, data=nested_value)
                    else:
                        # Store each parameter as a dataset
                        scan_group.create_dataset(key, data=value)

                scan_table.to_csv(saved_as+'_meta_data.csv')

                xrf_group = f.require_group('/xrf_roi_data/')
                xrf_stack, elem_list  = get_all_xrf_roi_data(hdr)
                scalar_stack, scalar_keys  = get_all_scalar_data(hdr)
                xrf_group.create_dataset('xrf_roi_array', data = xrf_stack)
                xrf_group.create_dataset('xrf_elem_names', data = elem_list)

                scalar_group = f.require_group('/scalar_data/')
                scalar_group.create_dataset('scalar_array', data = scalar_stack)
                scalar_group.create_dataset('scalar_array_names', data = scalar_keys)

            f.close()
            print(f"{saved_as =}")



def unpack_diff_h5(filename, det):
    """
    Unpack raw_data, Io, and scan_positions from an HDF5 file.

    Parameters:
        filename (str): Path to the HDF5 file.
        det (str): Detector name used in the group path.

    Returns:
        tuple: (raw_data, Io, scan_positions)
    """
    with h5py.File(filename, "r") as f:
        # Read data from the detector group
        det_group = f[f"/diff_data/{det}/"]
        raw_data = det_group["raw_data"][()]
        Io = det_group["Io"][()]
        
        # Read scan positions
        scan_positions = f["/scan/scan_positions"][()]

        xrf_stack = f["/xrf_roi_data/xrf_roi_array"][()]
        xrf_elem_list = [s.decode('utf-8') for s in f["/xrf_roi_data/xrf_elem_names"][()]]

    
    return raw_data, Io, scan_positions,xrf_stack, xrf_elem_list

def export_diff_h5_log_file(logfile, diff_detector = 'merlin1',compression = None):

    df = pd.read_csv(logfile)
    sid_list = df['scan_id'].to_numpy(dtype = 'int')
    angles = df['angle'].to_numpy()
    print(sid_list)

    dir_ = os.path.abspath(os.path.dirname(logfile))
    folder_name = os.path.basename(logfile).split('.')[0]
    save_folder =  os.path.join(dir_,folder_name+"_diff_data")
    data_path = save_folder
    os.makedirs(save_folder, exist_ok = True)

    print(f"h5 files will be saved to {save_folder}")
    
    export_diff_data_as_h5(sid_list, 
                           det=diff_detector,
                           wd = save_folder, 
                           compression = compression)
    
    print(f"All scans from {logfile} is exported to {save_folder}")



def export_single_diff_data(param_dict):
    
    '''
    load diffraction data of a single scan through databroker
    roi[row_start,col_start,row_size,col_size]
    mask has to be the same size of the image data, which corresponds to the last two axes
    
    param_dict = {wd:'.', 
                 "sid":-1, 
                 "det":"merlin1", 
                 "mon":"sclr1_ch4", 
                 "roi":None, 
                 "mask":None, 
                 "threshold":None}
    '''

    det=param_dict["det"]
    mon=param_dict["mon"]
    roi=param_dict["roi"]
    mask=param_dict["mask"]
    threshold=param_dict["threshold"]
    wd = param_dict["wd"]


    data_type = 'float32'
    data_name = '/entry/instrument/detector/data'
    sid = param_dict["sid"]
    start_doc = db[int(sid)].start
    sid = start_doc["scan_id"]
    param_dict["sid"] = sid
    file_name = get_path(sid,det)
    num_subscan = len(file_name)
    scan_table = db[sid].table()

    #print(f"Loading{sid} please wait...")
        

    hdr = db[int(sid)]
    start_doc = hdr["start"]
    sid = start_doc["scan_id"]
    
    if 'num1' and 'num2' in start_doc:
        dim1,dim2 = start_doc['num1'],start_doc['num2']
    elif 'shape' in start_doc:
        dim1,dim2 = start_doc.shape
    try:
        xy_scan_positions = list(np.array(df[mots[0]]),np.array(df[mots[1]]))
    except:
        xy_scan_positions = list(get_scan_positions(hdr))

    scan_table = get_scan_metadata(int(sid))
    if not start_doc["plan_type"] in ("FlyPlan1D",):

        file_name = get_path(sid,det)
        num_subscan = len(file_name)
        
        if num_subscan == 1:
            f = h5py.File(file_name[0],'r') 
            data = np.asarray(f[data_name],dtype=data_type)
        else:
            sorted_files = sort_files_by_creation_time(file_name)
            ind = 0
            for name in sorted_files:
                f = h5py.File(name,'r')
                if ind == 0:
                    data = np.asarray(f[data_name],dtype=data_type)
                else:   
                    data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                ind = ind + 1
            #data = list(db[sid].data(det))
            #data = np.asarray(np.squeeze(data),dtype=data_type)
        _, roi1,roi2 = np.shape(data)

        if threshold is not None:
            data[data<threshold[0]] = 0
            data[data>threshold[1]] = 0

        norm_with = mon

        if norm_with is not None:
            #mon_array = np.stack(hdr.table(fill=True)[norm_with])
            mon_array = np.array(list(hdr.data(str(norm_with)))).squeeze()
            norm_data = data/mon_array[:,np.newaxis,np.newaxis]
            print(f"data normalized with {norm_with} ")

        
        if mask is not None:     
            #sz = data.shape
            data = data*mask
        if roi is None:
            data = np.flip(data[:,:,:],axis = 1)
        else:
            data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
            
        print(f"data size = {data.size/1_073_741_824 :.2f} GB")

        #save_folder =  os.path.join(wd,f"{sid}_diff_data")   

        #if not os.path.exists(save_folder):
            #os.makedirs(save_folder)

        if wd:
            save_folder = wd
            
        saved_as = os.path.join(save_folder,f"scan_{sid}_{det}")

        f.close()

    print(f"data reshaped to {data.shape}")

    save_folder =  os.path.join(wd,f"{sid}_diff_data")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    saved_as = os.path.join(save_folder,f"{sid}_diff_{det}.tiff")
    if mon is not None:
        tf.imwrite(saved_as, data.reshape(dim1,dim2,roi1,roi2), dtype = np.float32)

    else:
        tf.imwrite(saved_as, data.reshape(dim1,dim2,roi1,roi2).astype('uint16'), imagej=True)

    export_scan_metadata(sid,save_folder)
    scan_table.to_csv(os.path.join(save_folder,f"{sid}_scan_table.csv"))
    print(f"{saved_as =}")

def get_file_creation_time(file_path):
    try:
        return os.path.getctime(file_path)
    except OSError:
        # If there is an error (e.g., file not found), return 0
        return 0

def sort_files_by_creation_time(file_list):
    # Sort the file list based on their creation time
    return sorted(file_list, key=lambda file: get_file_creation_time(file))

def parse_scan_range(str_scan_range):
    scanNumbers = []
    slist = str_scan_range.split(",")
    #print(slist)
    for item in slist:
        if "-" in item:
            slist_s, slist_e = item.split("-")
            print(slist_s, slist_e)
            scanNumbers.extend(list(np.linspace(int(slist_s.strip()), 
                                            int(slist_e.strip()), 
                                            int(slist_e.strip())-int(slist_s.strip())+1)))
        else:
            scanNumbers.append(int(item.strip()))
    
    return np.int_(sorted(scanNumbers))