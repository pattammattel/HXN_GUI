
import os
import warnings
import h5py
import pandas as pd
import datetime
import warnings
import sys
import numpy as np
import shutil
import tifffile as tf
from tqdm import tqdm
import pyqtgraph as pg
import matplotlib.pyplot as plt
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions

pg.setConfigOption('imageAxisOrder', 'row-major') # best performance

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if  os.getlogin().startswith("xf03") or os.getlogin().startswith("pattam"):

    #sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
    from hxntools.CompositeBroker import db
    from hxntools.scan_info import get_scan_positions

else: 
    db = None
    print("Offline analysis; No BL data available") 



det_params = {'merlin1':55, "merlin2":55, "eiger2_images":75}


def get_path(scan_id, key_name='merlin1'):
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
    #print(f"{scalar_keys = }")
    print(f"fetching scalar data")
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

    #print(f"{elem_list = }")
    print(f"fetching XRF ROIs")
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

def get_scan_metadata(sid):
    
    output = db.get_table(db[int(sid)],stream_name = "baseline")
    df_dictionary = pd.DataFrame([get_scan_details(sid = int(sid))])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    return output


def save_dict_to_h5(group, dictionary):
    """Recursively store a dictionary into HDF5 format"""
    for key, value in dictionary.items():
        if isinstance(value, dict):  # If it's a nested dictionary, create a subgroup
            subgroup = group.create_group(key)
            save_dict_to_h5(subgroup, value)
        else:  # If it's a simple type, create a dataset
            group.create_dataset(key, data=value)
            

def read_dict_from_h5(group):
    """
    Recursively read a dictionary from HDF5 format, decoding any byte-strings
    into Python str.
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = read_dict_from_h5(item)
        else:
            raw = item[()]
            result[key] = _decode_bytes(raw)
    return result

           
def _load_scan_common(hdr, mon, data_type='float32'):
    """
    Load everything *except* detector stacks (i.e. scan positions, xrf, scalar, scan params).
    """
    # 1) Monitor (Io) if requested
    sd = hdr.start
    plan   = hdr.start["scan"].get('type')
    
    if plan == "2D_FLY_PANDA":
        dim1, dim2 = (sd['num1'], sd['num2']) if 'num1' in sd and 'num2' in sd else sd.shape
        Io = None
        if mon:
            Io = np.array(list(hdr.data(str(mon))), dtype=data_type).squeeze().reshape(dim1, dim2)

        # 2) Scan positions
        try:
            xy = list(get_scan_positions(hdr))
        except:
            xy = [np.array(v) for v in df[mots]]  # fallback

        # 3) XRF & scalar
        xrf_stack, xrf_names = get_all_xrf_roi_data(hdr)
        scalar_stack, scalar_names = get_all_scalar_data(hdr)

        # 4) Scan parameters & metadata table
        scan_params = get_scan_details(hdr.start["scan_id"])
        scan_table  = get_scan_metadata(hdr.start["scan_id"])

        return {
            "Io": Io,
            "dim1": dim1,
            "dim2": dim2,
            "scan_positions": np.array(xy),
            "xrf_stack": xrf_stack,
            "xrf_names": xrf_names,
            "scalar_stack": scalar_stack,
            "scalar_names": scalar_names,
            "scan_params": scan_params,
            "scan_table": scan_table,
        }
    else: pass


def _load_detector_stack(hdr, det, data_type='float32'):
    """
    Load & reshape detector data more efficiently.
    """

    print("loading diff data")
    data_name = '/entry/instrument/detector/data'
    files = get_path(hdr.start["scan_id"], det)
    files = sorted(files, key=os.path.getctime)

    # Use memory mapping if possible for large datasets
    def read_file(fn):
        with h5py.File(fn, 'r') as f:
            return np.array(f[data_name], dtype=data_type)  # np.array slightly faster than np.asarray for HDF5

    # Option 1: Multithreading for I/O-bound task (if on fast shared FS)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
        arrays = list(executor.map(read_file, files))

    # Concatenate if needed
    data = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    return np.flip(data, axis=1)


def _read_group_as_dict(group):
    out = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _read_group_as_dict(item)
        else:
            val = item[()]
            if isinstance(val, (bytes, bytearray)):
                val = val.decode('utf-8')
            elif isinstance(val, np.ndarray) and val.dtype.kind in ('S', 'O', 'a'):
                val = [v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v for v in val]
            out[key] = val
    return out

def strip_and_rename_entry_data(h5_path, det="merlin1", compression="gzip"):
    """
    Strip all HDF5 groups except /entry/data/data and move it to /diff_data/{det}/det_images
    """
    with h5py.File(h5_path, 'r+') as f:
        # Step 1: Reference original dataset without loading
        if "/entry/data/data" not in f:
            raise KeyError("'/entry/data/data' not found in the file")
        dset_ref = f["/entry/data/data"]

        # Step 2: Create target group and copy dataset (fast internal copy)
        grp = f.require_group(f"/diff_data/{det}")
        f.copy(dset_ref, grp, name="det_images")

        # Step 3: Delete everything *except* /diff_data/{det}
        to_delete = [k for k in f.keys() if k == "entry"]
        for k in to_delete:
            del f[k]

        # Optional: delete extra groups inside /diff_data if needed
        for k in list(f["diff_data"].keys()):
            if k != det:
                del f["diff_data"][k]

def export_diff_data_as_h5_single(
    sid,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    save_to_disk=True,
    copy_if_possible=True,
    save_and_return=False
):
    """
    Export one scan (sid) & one detector (det) to HDF5 without any reshaping.
    - If a single raw file exists & copy_if_possible=True, copies it.
    - Otherwise loads raw via _load_detector_stack and writes it directly.
    Always writes:
      /diff_data/<det>/det_images  (raw shape: n_steps, ry, rx)
      /diff_data/<det>/Io          (2D monitor, if requested)
      scan_positions/positions
      scan_params/*
      xrf_roi_data/*
      scalar_data/*
    If save_and_return=True, returns metadata plus a lazy loader for the raw detector stack.
    """
    hdr    = db[int(sid)]
    plan   = hdr.start["scan"].get('type')
    if plan != "2D_FLY_PANDA":
        raise ValueError(f"Scan {sid} uses plan '{plan}', not 'fly2dpd'")
    if det not in hdr.start["scan"].get("detectors", []):
        raise ValueError(f"Detector '{det}' not in scan {sid}")

    common = _load_scan_common(hdr, mon)
    out_fn = os.path.join(wd, f"scan_{sid}_{det}.h5")
    copied = False

    raw_files = get_path(sid, det)
    if copy_if_possible and len(raw_files) == 1:
        shutil.copy2(raw_files[0], out_fn)
        strip_and_rename_entry_data(out_fn, det=det)
        copied = True

    if save_to_disk:
        mode = "a" if copied else "w"
        with h5py.File(out_fn, mode) as f:
            grp = f.require_group(f"/diff_data/{det}")
            if not copied:
                raw = _load_detector_stack(hdr, det)  # shape (n_steps, ry, rx)
                grp.create_dataset(
                    "det_images",
                    data=raw,
                    compression=compression
                )
            if common["Io"] is not None:
                grp.create_dataset("Io", data=common["Io"])

            sp = f.require_group("scan_positions")
            sp.create_dataset("positions", data=common["scan_positions"])

            pp = f.require_group("scan_params")
            save_dict_to_h5(pp, common["scan_params"])

            xg = f.require_group("xrf_roi_data")
            xg.create_dataset("xrf_roi_array",  data=common["xrf_stack"])
            xg.create_dataset("xrf_elem_names", data=common["xrf_names"])

            sg2 = f.require_group("scalar_data")
            sg2.create_dataset("scalar_array",       data=common["scalar_stack"])
            sg2.create_dataset("scalar_array_names", data=common["scalar_names"])

    result = {
        "filename": out_fn if save_to_disk else None,
        "copied":   copied,
    }

    if save_and_return:
        
        def load_det():
            with h5py.File(out_fn, "r") as f:
                print(f"data is flipped along y axis when returned")
                return np.flip(f[f"/diff_data/{det}/det_images"][()], 1)
                
        result.update({
            "det_images":             load_det(),
            "Io":              common["Io"],
            "scan_positions":  common["scan_positions"],
            "scan_params":     common["scan_params"],
            "xrf_array":       common["xrf_stack"],
            "xrf_names":       common["xrf_names"],
            "scalar_array":    common["scalar_stack"],
            "scalar_names":    common["scalar_names"],
        })

    return result

def export_diff_data_as_h5_batch(
    sid_list,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    copy_if_possible=True
):
    """
    Batch‐export one detector from each scan in sid_list:
      • Calls export_diff_data_as_h5_single(...) with save_to_disk=True,
        copy_if_possible as given, and save_and_return=False.
      • Prints a warning if a scan is skipped.
      • Returns None.
    """
    # normalize to list
    if isinstance(sid_list, (int, float)):
        sid_list = [int(sid_list)]

    for sid in tqdm(sid_list, desc="Batch exporting scans"):
        try:
            export_diff_data_as_h5_single(
                sid,
                det=det,
                wd=wd,
                mon=mon,
                compression=compression,
                save_to_disk=True,
                copy_if_possible=copy_if_possible,
                save_and_return=False
            )
        except ValueError as e:
            print(f"Skipping scan {sid!r}: {e}")


def unpack_diff_h5(filename, det="merlin1"):
    """
    Unpack a single‐detector HDF5 file with this structure:

      diff_data/<det>/{Io, det_images}
      scalar_data/{scalar_array, scalar_array_names}
      scan_positions/positions
      scan_params/...         (possibly nested)
      xrf_roi_data/{xrf_roi_array, xrf_elem_names}

    Returns a dict with keys:
      det_images, Io,
      scalar_array, scalar_names,
      scan_positions,
      scan_params,
      xrf_array, xrf_names
    """
    def _decode_list(arr):
        return [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            for x in arr
        ]

    result = {}
    with h5py.File(filename, "r") as f:
        # 1) diff_data
        dd = f["diff_data"][det]
        result["det_images"] = dd["det_images"][()]
        result["Io"] = dd["Io"][()] if "Io" in dd else None

        # 2) scalar_data
        sd = f["scalar_data"]
        result["scalar_array"] = sd["scalar_array"][()]
        raw_sn = sd["scalar_array_names"][()]
        result["scalar_names"] = _decode_list(raw_sn.tolist())

        # 3) scan_positions
        sp = f["scan_positions"]
        result["scan_positions"] = sp["positions"][()]

        # 4) scan_params (recursive)
        result["scan_params"] = _read_group_as_dict(f["scan_params"])

        # 5) xrf_roi_data
        if "xrf_roi_data" in f:
            xg = f["xrf_roi_data"]
            result["xrf_array"] = xg["xrf_roi_array"][()]
            raw_xn = xg["xrf_elem_names"][()]
            result["xrf_names"] = _decode_list(raw_xn.tolist())
        else:
            result["xrf_array"] = None
            result["xrf_names"] = []

    return result

if __name__ == "__main__" or "get_ipython" in globals():
    print("\n✅ Diffraction data I/O module loaded.")
    
    print("\n#####📘 For export only use this ######:\n") 
    print("▶ export_diff_data_as_h5_batch(sid_list, det, wd, mon, compression, save_to_disk, copy_if_possible)")
    print("   → Fast bulk exporter. If only 1 raw HDF5 file exists, it will copy instead of re-saving.")
    
    print("\n#####📘 For export and return/visualize data ######:\n") 
    print("▶ export_diff_data_as_h5_single(sid_list, det, wd, mon, compression, save_to_disk, return_data)")
    print("   → Saves or returns data for one or more scan IDs.")
    
    print("\n#####📘 To read the h5 saved using export_diff_data_as_h5 function ######:\n") 
    print("▶ unpack_diff_h5(filename, det)")
    print("   → Reads saved HDF5 back into a dictionary (diff, scan, XRF, scalar).")
    print("----------------------------------------------------------")