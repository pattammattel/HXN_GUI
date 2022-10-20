

from databroker import Broker
from hxntools.handlers import register
from hxntools.scan_info import ScanInfo
#from probe_propagation.prop_probe_v2 import *

import sys
import os
import json
import collections
import ast
import h5py
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import tifffile as tf

try:
    # new mongo database
    hxn_db = Broker.named('hxn')
    register(hxn_db)
except FileNotFoundError:
    print("hxn.yml not found. Unable to access HXN's database.", file=sys.stderr)
    hxn_db = None

'''
cmap_names = ['CET-L13', 'CET-L14', 'CET-L15']
cmap_label1 = ['red', 'green', 'blue']
cmap_dict = {}
for i, name in zip(cmap_names, cmap_label1):
    cmap_dict[name] = pg.colormap.get(i).getLookupTable(alpha=True)
'''


# API; later move to a file

def calculate_res_and_dof(energy, det_distance_m, det_pixel_um, img_size):
    lambda_nm = 1.2398 / energy
    pixel_size = lambda_nm * 1.e-9 * det_distance_m / (img_size * det_pixel_um * 1e-6)
    depth_of_field = lambda_nm * 1.e-9 / (img_size / 2 * det_pixel_um * 1.e-6 / det_distance_m) ** 2

    return pixel_size, depth_of_field


def calculate_img_size(energy, det_distance_m, det_pixel_um, target_nm):
    lambda_nm = 1.2398 / energy

    img_size = lambda_nm * 1.e-9 * det_distance_m / (target_nm * 1.e-9 * det_pixel_um * 1e-6)

    if img_size % 2 == 1:
        img_size += 1

    return int(img_size)


def get_single_image(db, frame_num, mds_table):
    length = (mds_table.shape)[0]
    if frame_num >= length:
        message = "[ERROR] The {0}-th frame doesn't exist. "
        message += "Available frames for the chosen scan: [0, {1}]."
        raise ValueError(message.format(frame_num, length - 1))

    img = db.reg.retrieve(mds_table.iat[frame_num])[0]
    return np.nan_to_num(img,nan=0.0, posinf=0.0, neginf=0.0)

def check_info(sid):
    header = hxn_db[sid]
    plan_args = header.start['plan_args']
    scan_type = header.start['plan_name']
    bl = hxn_db.get_table(header, stream_name='baseline')
    energy = bl.energy.iloc[0]


def replacePixelValues(image, list_pixels, setToZero=False):
    "replace 2D image  pixels in in the list with neighbor average or zero"

    if list_pixels:

        # replace the pixel with neighbor average
        for pixel in list_pixels:

            if setToZero:
                image[pixel[1], pixel[0]] = 0
                print(f"{pixel[1], pixel[0]} = 0")

            else:
                replaceWith_m = image[pixel[1] - 1:pixel[1] + 1, pixel[0] - 1:pixel[0] + 1]
                replaceWith_m[1, 1] = 0
                replaceWith = np.mean(replaceWith_m)
                image[pixel[1], pixel[0]] = int(replaceWith)

                print(f"{pixel[1], pixel[0]} = {int(replaceWith)}")

        return image

    else:
        # print("No Pixel correction")
        # print(list_pixels)
        pass


def replacePixelValues3D(image_stk, pixels_to_nbr_avg, pixels_to_zero):
    mod_img_stk = np.zeros_like(image_stk)
    for n in range(image_stk.shape[0]):
        image = image_stk[n]
        replacePixelValues(image, pixels_to_zero, setToZero=True)
        replacePixelValues(image, pixels_to_nbr_avg, setToZero=False)
        mod_img_stk[n] = image
    return mod_img_stk


def cropToROI(img_stk, dims: tuple):
    xpos, ypos, xsize, ysize = dims

    if dims != (0, 0, 0, 0):
        crop_img_stk = img_stk[:, ypos:ypos + ysize, xpos:xpos + xsize]

    return crop_img_stk


def get_detector_images(db_, scan_num, det_name, norm='sclr1_ch4'):
    ''' load detector data as 3D array; this will take a while'''

    header = db_[scan_num]
    motors = header.start['motors']
    items = [det_name, 'sclr1_ch2', 'sclr1_ch4'] + motors
    # create dataframe with items
    df = db_.get_table(header, fields=items, fill=False)

    raw_images = np.squeeze(list(header.data(det_name)))
    # todo make scalar selection option in the gui
    ic = np.asfarray(df[norm])
    ic = np.where(ic == 0, np.nanmean(ic), ic)
    ic_ = ic[0] / ic
    ic_norm = np.ones_like(raw_images) * ic_[:, np.newaxis, np.newaxis]

    norm_images = raw_images * ic_norm
    # images = np.fliplr(norm_images).transpose(0, 2, 1)
    # print(np.shape(images))

    return norm_images


def parse_scan_range(str_scan_range):
    scanNumbers = []
    slist = str_scan_range.split(",")
    # print(slist)
    for item in slist:
        if "-" in item:
            slist_s, slist_e = item.split("-")
            print(slist_s, slist_e)
            scanNumbers.extend(list(np.linspace(int(slist_s.strip()),
                                                int(slist_e.strip()),
                                                int(slist_e.strip()) - int(slist_s.strip()) + 1)))
        else:
            scanNumbers.append(int(item.strip()))

    return np.int_(scanNumbers)


def save_ptycho_h5(config, mesh_flag, fly_flag):
    '''

    Sample Config file;
    config = {
                "wd": os.getcwd(),
                "scan_num":'',
                "detector": "merlin1",
                "crop_roi": (0,0,0,0),
                "hot_pixels": [],
                "outl_pixels" : [],
                "switchXY":False,
                "det_dist":0.5,
                "energy":12,
                "db":hxn_db
                }



    '''

    # TODO think if keeping scan number outside the config file is better

    # os make /h5_data/ dir , if not exist
    if not os.path.exists(os.path.join(config["wd"], 'h5_data')):
        os.makedirs(os.path.join(config["wd"], 'h5_data'))

    db = config["db"]
    header = db[config["scan_num"]]
    start_doc = header["start"]

    if not start_doc["plan_type"] in ("FlyPlan1D",):

        plan_args = header.start['plan_args']
        scan_type = header.start['plan_name']
        motors = header.start['motors']
        bl = db.get_table(header, stream_name='baseline')
        items = [config["detector"], 'sclr1_ch3', 'sclr1_ch4'] + motors
        df = db.get_table(header, fields=items, fill=False)

        try:
            angle = bl.zpsth[1]
        except:
            angle = 0

        # dcm_th = bl.dcm_th[1]
        # energy_kev = 12.39842 / (2. * 3.1355893 * np.sin(dcm_th * np.pi / 180.))
        config["energy"] = bl.energy.iloc[0]  # replace?
        lambda_nm = 1.2398 / config["energy"]

        if mesh_flag:
            if fly_flag:
                x_range = plan_args['scan_end1'] - plan_args['scan_start1']
                y_range = plan_args['scan_end2'] - plan_args['scan_start2']
                x_num = plan_args['num1']
                y_num = plan_args['num2']
            else:
                x_range = plan_args['args'][2] - plan_args['args'][1]
                y_range = plan_args['args'][6] - plan_args['args'][5]
                x_num = plan_args['args'][3]
                y_num = plan_args['args'][7]
            dr_x = 1. * x_range / x_num
            dr_y = 1. * y_range / y_num
            x_range = x_range - dr_x
            y_range = y_range - dr_y
        else:
            x_range = plan_args['x_range']
            y_range = plan_args['y_range']
            dr_x = plan_args['dr']
            dr_y = 0

        if config["switchXY"]:

            y = np.array(df[motors[0]])
            x = np.array(df[motors[1]])

        else:
            x = np.array(df[motors[0]])
            y = np.array(df[motors[1]])

        points = np.vstack([x, y])

        # cx, cy = center of the roi
        n, nn = int(config["crop_roi"][-2]), int(config["crop_roi"][-1])
        cx, cy = int(config["crop_roi"][0]), int(config["crop_roi"][1])

        # remove bad pixels
        # det_images = self.load_detector_images_gui()
        det_images = get_detector_images(config["db"], config["scan_num"], config["detector"])
        mod_image = replacePixelValues3D(det_images, config["hot_pixels"], config["outl_pixels"])
        print(f"raw data shape: {np.shape(det_images)}")

        tmptmp = mod_image[:, cy:nn + cy, cx:n + cx]

        # image flipping and rotation
        tmptmp = np.fliplr(tmptmp).transpose(0, 2, 1)

        print(f"crop data shape: {np.shape(tmptmp)}")

        data = np.fft.fftshift(tmptmp, axes=[1, 2])

        threshold = 1.
        data = data - threshold
        data[data < 0.] = 0.
        data = np.sqrt(data)

        det_pixel_um = 55.
        det_distance_m = config["det_dist"]

        pixel_size, depth_of_field = calculate_res_and_dof(config["energy"], det_distance_m, det_pixel_um, n)
        print('pixel num, pixel size, depth of field: ', n, pixel_size, depth_of_field)

        print("creating h5")
        with h5py.File(config["wd"] + '/h5_data/scan_' + str(config["scan_num"]) + '.h5', 'w') as hf:
            # with h5py.File(config["wd"] + '/scan_' + str(config["scan_num"]) + '.h5', 'w') as hf:
            dset = hf.create_dataset('diffamp', data=data)
            dset = hf.create_dataset('points', data=points)
            dset = hf.create_dataset('x_range', data=x_range)
            dset = hf.create_dataset('y_range', data=y_range)
            dset = hf.create_dataset('dr_x', data=dr_x)
            dset = hf.create_dataset('dr_y', data=dr_y)
            dset = hf.create_dataset('z_m', data=det_distance_m)
            dset = hf.create_dataset('lambda_nm', data=lambda_nm)
            dset = hf.create_dataset('ccd_pixel_um', data=det_pixel_um)
            dset = hf.create_dataset('angle', data=angle)
            # dset = hf.create_dataset('Ni_xrf',data=Ni_xrf)
            # dset = hf.create_dataset('Au_xrf',data=Au_xrf)

        # symlink
        src = f'{config["wd"]}/h5_data/scan_{config["scan_num"]}.h5'
        dest = f'{config["wd"]}/scan_{config["scan_num"]}.h5'
        os.symlink(src, dest)

    else:
        print(f'{config["scan_num"]} is a 1D scan; skipped')
        return
