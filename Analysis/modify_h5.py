import h5py
import numpy as np

def rebin(a,shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def swapPtychoScanAxis(sid = 120000):
    fn = f'scan_{sid}.h5'

    f = h5py.File('./h5_data/'+fn, 'r')
    data = np.array(f['diffamp'])
    points = np.array(f['points'])
    x_range = np.array(f['x_range'])
    y_range = np.array(f['y_range'])
    dr_x = np.array(f['dr_x'])
    dr_y = np.array(f['dr_y'])
    det_distance_m = np.array(f['z_m'])
    lambda_nm = np.array(f['lambda_nm'])
    det_pixel_um = np.array(f['ccd_pixel_um'])
    angle = np.array(f['angle'])

    f.close()
    points = points[::-1]

    with h5py.File('./h5_data/'+fn, 'w') as hf:
        dset = hf.create_dataset('diffamp',data=data)
        dset = hf.create_dataset('points',data=points)
        dset = hf.create_dataset('x_range',data=x_range)
        dset = hf.create_dataset('y_range',data=y_range)
        dset = hf.create_dataset('dr_x',data=dr_x)
        dset = hf.create_dataset('dr_y',data=dr_y)
        dset = hf.create_dataset('z_m',data=det_distance_m)
        dset = hf.create_dataset('lambda_nm',data=lambda_nm)
        dset = hf.create_dataset('ccd_pixel_um',data=det_pixel_um)
        dset = hf.create_dataset('angle',data=angle)
        dset = hf.create_dataset('bragg_theta',data=angle)
        
    print(f'{sid} done!')

    
