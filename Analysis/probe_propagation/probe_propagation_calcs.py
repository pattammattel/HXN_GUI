import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def calculate_res_and_dof(energy, det_distance_m, det_pixel_um, img_size):
    lambda_nm = 1.2398 / energy
    pixel_size = lambda_nm * 1.e-9 * det_distance_m / (img_size * det_pixel_um * 1e-6)
    depth_of_field = lambda_nm * 1.e-9 / (img_size / 2 * det_pixel_um * 1.e-6 / det_distance_m) ** 2

    return pixel_size, depth_of_field

def gauss_linear(x,p0,p1,p2,p3,p4):
    return p0 + p1 * x + p2*np.exp(-(x-p3)**2/(2.*p4**2))

def gaussian_fit(x,y,para):

    para = np.array(para)
    popt, pcov = curve_fit(gauss_linear,x,y,para)
    y_fit = gauss_linear(x,popt[0],popt[1],popt[2],popt[3],popt[4])

    return popt, pcov,y_fit

def propagate(probe_np_array,energy,dist,dx, dy):

    """"dist,dx,dy in microns"""

    wavelength_m = 12.398*1.e-4/energy

    k = 2. * np.pi / wavelength_m
    nx, ny = np.shape(probe_np_array)
    spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(probe_np_array)))

    dkx = 2. * np.pi / (nx * dx)
    dky = 2. * np.pi / (ny * dy)

    skx = dkx * nx / 2
    sky = dky * ny / 2

    kproj_x = np.linspace(-skx, skx - dkx, nx)
    kproj_y = np.linspace(-sky, sky - dky, ny)
    kx, ky = np.meshgrid(kproj_x, kproj_y)

    phase = np.sqrt(k ** 2 - kx ** 2 - ky ** 2) * dist

    spectrum *= np.exp(1j * phase)
    array_prop = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(spectrum)))

    return array_prop


def propagate_probe(probe_file,det_distance_m=0.5,energy=12,det_pixel_size = 55, start_um=-50,end_um=50,step_size_um=1):


    # load the probe file
    prb_ini = np.load(probe_file)
    nx, ny = np.shape(prb_ini)
    # pixel size , calculation is ;
    nx_size_m,_ = calculate_res_and_dof(energy,
                                        det_distance_m,
                                        det_pixel_size,
                                        nx)
    ny_size_m, _ = calculate_res_and_dof(energy,
                                         det_distance_m,
                                         det_pixel_size,
                                         ny)

    # propagation distance and steps
    num_steps = int((end_um - start_um) / step_size_um) + 1
    projection_points = np.linspace(start_um,
                                    end_um,
                                    num_steps)

    print(projection_points)

    #for sigma
    sigma = np.zeros((3, num_steps))

    #for fitted data
    xfits = np.zeros((num_steps, 2, nx))
    yfits = np.zeros((num_steps, 2, ny))

    #initial probe
    prb = propagate(prb_ini,
                    energy,0,
                    nx_size_m*10**6,
                    nx_size_m*10**6)

    # create an image stack with probe at each propogated distance
    prop_data = np.zeros((nx, ny, num_steps)).astype(complex)


    for i, distance in enumerate(projection_points):

        print(i)
        tmp = propagate(prb,
                        energy,
                        distance,
                        nx_size_m*10**6,
                        nx_size_m*10**6)

        prop_data[:, :, i] = tmp

        if i == 0:
            sig_x, sig_y,data_x,data_y = probe_img_to_linefit(tmp,
                                                              gaussian_sig_init = 0.8)

        else:
            sig_x, sig_y,data_x,data_y  = probe_img_to_linefit(tmp,
                                                               gaussian_sig_init=sigma[1,i-1])

        sigma[0,i] = distance
        sigma[1, i] = sig_x
        sigma[2, i] = sig_y

        xfits[i] = data_x
        yfits[i] = data_y

    return prop_data, sigma, xfits, yfits

def probe_img_to_linefit(prb_image, gaussian_sig_init = 0.8):

    nx, ny = np.shape(prb_image)
    #axis for projection 1D
    proj_x = np.arange(nx, dtype = np.float64)
    proj_y = np.arange(ny,dtype = np.float64)

    x_fit_data = np.zeros((2,nx))
    y_fit_data = np.zeros((2, ny))

    # find the max points in the image and get the line profile at that point
    ix, iy = np.where(np.abs(prb_image) == np.nanmax(np.abs(prb_image)))
    prb_intensity = (np.abs(prb_image)) ** 2
    line_tmp_x = np.squeeze(prb_intensity.sum(0))
    line_tmp_y = np.squeeze(prb_intensity.sum(1))

    x_fit_data[0] = line_tmp_x
    y_fit_data[0] = line_tmp_y

    gaussian_params_y = [0., 0., 1, iy[0], gaussian_sig_init]
    gaussian_params_x = [0., 0., 1, ix[0], gaussian_sig_init]

    popt, pcov, y_fit = gaussian_fit(proj_y,
                                     line_tmp_y / np.max(line_tmp_y),
                                     gaussian_params_y)
    sigma_y = popt[4]
    popt, pcov, x_fit = gaussian_fit(proj_x,
                                     line_tmp_x / np.max(line_tmp_x),
                                     gaussian_params_x)
    sigma_x = popt[4]

    x_fit_data[1] = x_fit
    y_fit_data[1] = y_fit

    return sigma_x, sigma_y, x_fit_data, y_fit_data