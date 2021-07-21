import numpy as np
import h5py
import tifffile as tf
import os
import matplotlib.pyplot as plt



def getEnergyNScalar(h='h5file'):
	""" 
	Function retrieve the ion chamber readings and 
	mono energy from an h5 file created within pyxrf at HXN
	
	input: h5 file path
	output1: normalized IC3 reading ("float")
	output2: mono energy ("float")
	
	"""
	#open the h5
    f = h5py.File(h, 'r') 
	# get Io and IC3,  edges are removeed to exclude nany dropped frame or delayed reading
    Io = np.array(f['xrfmap/scalers/val'])[1:-1, 1:-1, 0].mean() 
    I = np.array(f['xrfmap/scalers/val'])[1:-1, 1:-1, 2].mean()
	# get monoe
    mono_e = f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy']
    
	#return values
	return I/Io, mono_e
    
def getCalibSpectrum(path_):
    """
	
	Get the I/Io and enegry value from all the h5 files in the given folder
	
	input: path to folder (string)
	output: calibration array containing energy in column and log(I/Io) in the other (np.array)
	
	
	"""
	
	#get the all the files in the directory
    fileList = os.listdir(path_)
    
	#empty list to add values
    spectrum = []
    energyList = []
    
    for file in sorted(fileList):
        if file.endswith('.h5'): #filer for h5
            IbyIo, mono_e = getEnergyNScalar(h=file) 
            energyList.append(mono_e)
            spectrum.append(IbyIo)
    
	#get the output in two column format
    calib_spectrum = np.column_stack([energyList,(-1*np.log10(spectrum))])
	
	#save as txt to the parent folder
	np.savetxt('calibration_spectrum.txt',calib_spectrum)
	
	#plot results
    plt.plot(calib_spectrum[:,0],calib_spectrum[:,1])
	plt.ioff()
    plt.gcf().show()
                
   
            
