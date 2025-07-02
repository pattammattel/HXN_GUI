
import os
import warnings
import h5py
import pandas as pd
import datetime
import warnings
import sys
import numpy as np
import tifffile as tf
from tqdm import tqdm


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
    if hdr.start['plan_name'] == 'fly2dpd':
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


def export_diff_data_as_h5(
    sid_list,
    det="eiger2_image",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    save_to_disk=True,
    return_data=False
):
    """
    Export diffraction scan(s) to HDF5. Optionally return in-memory data.

    Parameters
    ----------
    sid_list : int or list of int
        List of scan IDs.
    det : str
        Detector name.
    wd : str
        Directory to save the HDF5 files.
    mon : str
        Monitor signal for normalization.
    compression : str
        Compression method for HDF5 datasets.
    save_to_disk : bool
        If True, write output to .h5 files.
    return_data : bool
        If True, return in-memory data as list of dicts.

    Returns
    -------
    list of dicts if return_data is True, otherwise None.
    """
    if isinstance(sid_list, (int, float)):
        sid_list = [int(sid_list)]

    results = []

    for sid in sid_list:
        hdr = db[int(sid)]
        common = _load_scan_common(hdr, mon)

        dim1, dim2 = common["dim1"], common["dim2"]
        raw = _load_detector_stack(hdr, det)
        _, roi_y, roi_x = raw.shape

        out_fn = os.path.join(wd, f"scan_{sid}_{det}.h5")

        if save_to_disk:
            with h5py.File(out_fn, 'w') as f:
                grp = f.require_group(f"/diff_data/{det}")
                grp.create_dataset("det_images", data=raw.reshape(dim1, dim2, roi_y, roi_x), compression=compression)
                if common["Io"] is not None:
                    grp.create_dataset("Io", data=common["Io"])

                sg = f.require_group("scan")
                sg.create_dataset("scan_positions", data=common["scan_positions"])
                save_dict_to_h5(sg, common["scan_params"])
                common["scan_table"].to_csv(out_fn.replace(".h5", "_meta_data.csv"))

                xg = f.require_group("xrf_roi_data")
                xg.create_dataset("xrf_roi_array", data=common["xrf_stack"])
                xg.create_dataset("xrf_elem_names", data=common["xrf_names"])

                sg2 = f.require_group("scalar_data")
                sg2.create_dataset("scalar_array", data=common["scalar_stack"])
                sg2.create_dataset("scalar_array_names", data=common["scalar_names"])

        if return_data:
            results.append({
                "Io": common["Io"],
                "scan_positions": common["scan_positions"],
                "xrf_stack": common["xrf_stack"],
                "xrf_names": common["xrf_names"],
                "scalar_stack": common["scalar_stack"],
                "scalar_names": common["scalar_names"],
                "det_images": raw,
                "filename": out_fn if save_to_disk else None
            })

    return results if return_data else None

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


def unpack_diff_h5(filename, det="eiger2_image"):
    result = {}
    with h5py.File(filename, "r") as f:
        grp = f["/diff_data"][det]
        result["det_images"] = grp["det_images"][()]
        result["Io"] = grp.get("Io")[()] if "Io" in grp else None

        result["scan"] = _read_group_as_dict(f["/scan"])

        if "xrf_roi_data" in f:
            xrf = f["xrf_roi_data"]
            result["xrf_array"] = xrf["xrf_roi_array"][()]
            result["xrf_names"] = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else n for n in xrf["xrf_elem_names"][()]]
        else:
            result["xrf_array"] = None
            result["xrf_names"] = []

        if "scalar_data" in f:
            scalar = f["scalar_data"]
            result["scalar_array"] = scalar["scalar_array"][()]
            result["scalar_names"] = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else s for s in scalar["scalar_array_names"][()]]
        else:
            result["scalar_array"] = None
            result["scalar_names"] = []

    return result


import shutil

def export_diff_data_as_h5_batch(
    sid_list,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    save_to_disk=True,
    copy_if_possible=True
):
    if isinstance(sid_list, (int, float)):
        sid_list = [int(sid_list)]

    results = []

    for sid in sid_list:
        hdr = db[int(sid)]
        scan_plan = hdr.start['plan_name']
        print (f" {scan_plan = }")
        if scan_plan == 'fly2dpd':
            dets_used = hdr.start['scan']['detectors']
            if det in dets_used:
                scan_id = hdr.start["scan_id"]
                print(f"{scan_id = }")
                common = _load_scan_common(hdr, mon)
                dim1, dim2 = common["dim1"], common["dim2"]

                raw_files = get_path(scan_id, det)
                out_fn = os.path.join(wd, f"scan_{scan_id}_{det}.h5")

                copied = False
                if copy_if_possible and len(raw_files) == 1:
                    shutil.copy2(raw_files[0], out_fn)
                    strip_and_rename_entry_data(out_fn, det=det)  # remove unwanted
                    copied = True
                    print(f"Copied detector data from {raw_files[0]} to {out_fn}")

                if save_to_disk:
                    mode = 'a' if copied else 'w'
                    with h5py.File(out_fn, mode) as f:
                        if not copied:
                            # load raw if not copied
                            raw = _load_detector_stack(hdr, det)
                            _, roi_y, roi_x = raw.shape
                            grp = f.require_group(f"/diff_data/{det}")
                            grp.create_dataset(
                                "det_images",
                                data=raw.reshape(dim1, dim2, roi_y, roi_x),
                                compression=compression
                            )
                            if common["Io"] is not None:
                                grp.create_dataset("Io", data=common["Io"])
                        else:
                            # detector already in file; skip writing
                            if common["Io"] is not None:
                                grp = f.require_group(f"/diff_data/{det}")
                                grp.create_dataset("Io", data=common["Io"])

                        # Add scan info
                        sg = f.require_group("scan")
                        sg.create_dataset("scan_positions", data=common["scan_positions"])
                        save_dict_to_h5(sg, common["scan_params"])
                        common["scan_table"].to_csv(out_fn.replace(".h5", "_meta_data.csv"))

                        # Add XRF
                        xg = f.require_group("xrf_roi_data")
                        xg.create_dataset("xrf_roi_array", data=common["xrf_stack"])
                        xg.create_dataset("xrf_elem_names", data=common["xrf_names"])

                        # Add scalars
                        sg2 = f.require_group("scalar_data")
                        sg2.create_dataset("scalar_array", data=common["scalar_stack"])
                        sg2.create_dataset("scalar_array_names", data=common["scalar_names"])

                # optionally build return dict
                size_gb = os.path.getsize(out_fn) / (1024 **3)
                print(f"Saved: {out_fn} ({size_gb:.3f} GB)")
                results.append({
                    "filename": out_fn if save_to_disk else None,
                    "copied": copied,
                    "size_gb": round(size_gb, 3)
                })
                return results
            else: 
                print(f"{det} is not found in the {sid = }; skipped")
                pass
                        
        else: 
            print(f"{sid = } is not a fly2d plan skipped")
            pass 
    
        


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


if __name__ == "__main__" or "get_ipython" in globals():
    print("\n✅ Diffraction data I/O module loaded.")
    
    print("\n#####📘 For export only use this ######:\n") 
    print("▶ export_diff_data_as_h5_batch(sid_list, det, wd, mon, compression, save_to_disk, copy_if_possible)")
    print("   → Fast bulk exporter. If only 1 raw HDF5 file exists, it will copy instead of re-saving.")
    
    print("\n#####📘 For export and return/visualize data ######:\n") 
    print("▶ export_diff_data_as_h5(sid_list, det, wd, mon, compression, save_to_disk, return_data)")
    print("   → Saves or returns data for one or more scan IDs.")
    
    print("\n#####📘 To read the h5 saved using export_diff_data_as_h5 function ######:\n") 
    print("▶ unpack_diff_h5(filename, det)")
    print("   → Reads saved HDF5 back into a dictionary (diff, scan, XRF, scalar).")
    print("----------------------------------------------------------")