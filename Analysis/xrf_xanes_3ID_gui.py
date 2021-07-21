#conda activate analysis-2019-3.0-hxn-clone2

import sys, os, time, subprocess, logging,gc,h5py,traceback
import matplotlib.pyplot as plt
import numpy as np
import time

from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from pyxrf.api import *
from epics import caget

logger = logging.getLogger()


def getEnergyNScalar(h='h5file'):
    """
    Function retrieve the ion chamber readings and
    mono energy from an h5 file created within pyxrf at HXN

    input: h5 file path
    output1: normalized IC3 reading ("float")
    output2: mono energy ("float")

    """
    # open the h5


    f = h5py.File(h, 'r')
    # get Io and IC3,  edges are removeed to exclude nany dropped frame or delayed reading
    Io = np.array(f['xrfmap/scalers/val'])[1:-1, 1:-1, 0].mean()
    I = np.array(f['xrfmap/scalers/val'])[1:-1, 1:-1, 2].mean()
    # get monoe
    mono_e = f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy']

    # return values
    return I / Io, mono_e


def getCalibSpectrum(path_ = os.getcwd()):
    """

	Get the I/Io and enegry value from all the h5 files in the given folder

	input: path to folder (string), if none, use current working directory
	output: calibration array containing energy in column and log(I/Io) in the other (np.array)


	"""

    # get the all the files in the directory
    fileList = os.listdir(path_)

    # empty list to add values
    spectrum = []
    energyList = []

    for file in sorted(fileList):
        if file.endswith('.h5'):  # filer for h5
            IbyIo, mono_e = getEnergyNScalar(h=file)
            energyList.append(mono_e)
            spectrum.append(IbyIo)

    # get the output in two column format
    calib_spectrum = np.column_stack([energyList, (-1 * np.log10(spectrum))])
    #sort by energy
    calib_spectrum = calib_spectrum[np.argsort(calib_spectrum[:, 0])]
    # save as txt to the parent folder
    np.savetxt('calibration_spectrum.txt', calib_spectrum)
    logger.info("calibration spectrum saved in : {path_} ")

    # plot results
    plt.plot(calib_spectrum[:, 0], calib_spectrum[:, 1])
    plt.ioff()
    plt.gcf().show()



def hxn_auto_loader(wd, param_file_name, scaler_name, buffer_time = 100, hdf = True, xrf_fit = True):
    sid = 100
    printed = False

    while True:
    
        gc.collect()

        while caget('XF:03IDC-ES{Sclr:2}_cts1.B') < 5000:
            logger.info('beam is not available: waiting for shutter to open')
            time.sleep(60)

        while caget('XF:03IDC-ES{Status}ScanRunning-I') == 1 and not printed:
            print('\n**waitng for scan to complete**\n')
            printed = True

        while caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and int(caget('XF:03IDC-ES{Status}ScanID-I')) != sid:

            sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
            print(f'calling scan {sid} from data broaker')
            hdr = db[(sid)]

            if bool(hdr.stop):
                try:
                
                    if hdf: 
                        make_hdf(sid, wd=wd, file_overwrite_existing=True)
                        
                    if xrf_fit: 
                        pyxrf_batch(sid, sid, wd=wd, param_file_name=param_file_name, scaler_name=scaler_name)

                    time.sleep(buffer_time)

                except:
                    pass

    print('\n**waitng for next scan to complete**\n')

    def


class xrf_3ID(QtWidgets.QMainWindow):
    def __init__(self):
        super(xrf_3ID, self).__init__()
        uic.loadUi("/GPFS/XF03ID1/home/xf03id/user_macros/HXN_GUI/Analysis/xrf_xanes_gui_debug.ui", self)

        self.pb_wd.clicked.connect(self.get_wd)
        self.pb_param.clicked.connect(self.get_param)
        self.pb_ref.clicked.connect(self.get_ref_file)
        self.pb_start.clicked.connect(self.create_xanes_macro)

        self.pb_xrf_start.clicked.connect(lambda: self.threadMaker(self.create_pyxrf_batch_macro))
        self.pb_live.clicked.connect(self.start_auto)
        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        self.pb_close_plots.clicked.connect(self.close_all_plots)
        
        self.pb_scan_meta.clicked.connect(self.print_metadata)
        self.pb_scan_dets.clicked.connect(self.print_dets)

        self.show()

    def get_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(folder_path))
    
    def get_param(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'json file (*.json)')
        self.le_param.setText(str(file_name[0]))
        
    def get_ref_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'ref file (*.txt, *.nor, *.csv)')
        self.le_ref.setText(str(file_name[0]))
        
    
    def create_pyxrf_batch_macro(self):
    
        cwd = self.le_wd.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        norm = self.le_sclr_2.text()
        all_sid = np.arange(first_sid, last_sid+1)
        logger.info(f'scans to process {all_sid}')

        self.pte_status.clear()
        self.pte_status.appendPlainText("All the h5 files will be created first followed by XRF Fitting")
        
        for sids in all_sid:
            if self.rb_make_hdf.isChecked():
                make_hdf(int(sids),int(sids), wd = cwd, file_overwrite_existing=True)
                
            if self.rb_xrf_fit.isChecked():
                param = self.le_param.text()
                pyxrf_batch(int(sids), int(sids), wd = cwd, param_file_name =param, scaler_name=norm, save_tiff = True)
             
        
                                   
    def create_xanes_macro(self):

        gc.collect()
        cwd = self.le_wd.text()
        param = self.le_param.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        ref = self.le_ref.text()
        fit_method = self.cb_fittin_method.currentText()
        elem = self.xanes_elem.text()
        align_elem = self.alignment_elem.text()
        e_shift = float(self.energy_shift.text())
        admm_lambda = int(self.nnls_lamda.text())
        work_flow = self.cb_process.currentText()
        norm = self.le_sclr.text()
        save_all = self.ch_b_save_all_tiffs.isChecked()
        pre_edge = self.ch_b_baseline.isChecked()
        align = self.cb_align.isChecked()

        build_xanes_map_param = {}
        build_xanes_map_param["cwd"] = self.le_wd.text()
        build_xanes_map_param["param"] = self.le_param.text()
        build_xanes_map_param["last_sid"] = int(self.le_lastid.text())
        build_xanes_map_param["first_sid"] = int(self.le_startid.text())
        build_xanes_map_param["ref"] = self.le_ref.text()
        build_xanes_map_param["fit_method"] = self.cb_fittin_method.currentText()
        build_xanes_map_param["elem"] = self.xanes_elem.text()
        build_xanes_map_param["align_elem"] = self.alignment_elem.text()
        build_xanes_map_param["e_shift"] = float(self.energy_shift.text())
        build_xanes_map_param["admm_lambda"] = int(self.nnls_lamda.text())
        build_xanes_map_param["work_flow"] = self.cb_process.currentText()
        build_xanes_map_param["norm"] = self.le_sclr.text()
        build_xanes_map_param["save_all"] = self.ch_b_save_all_tiffs.isChecked()
        build_xanes_map_param["pre_edge"] = self.ch_b_baseline.isChecked()
        build_xanes_map_param["align"] = self.cb_align.isChecked()
		
        self.pte_status.appendPlainText(str(build_xanes_map_param))
        '''
        build_xanes_map(first_sid, last_sid, wd = cwd,xrf_subdir = cwd, xrf_fitting_param_fln=param,
                        scaler_name=norm,sequence=work_flow,
                        ref_file_name=ref, fitting_method=fit_method,
                        emission_line=elem, emission_line_alignment=align_elem,
                        incident_energy_shift_keV=(e_shift*0.001), subtract_pre_edge_baseline = pre_edge,
                        alignment_enable = align, output_save_all = save_all, use_incident_energy_from_param_file = True)
		'''
		
        build_xanes_map(first_sid, last_sid, wd = cwd,xrf_subdir = cwd, xrf_fitting_param_fln=param,
                        scaler_name=norm,sequence=work_flow,
                        ref_file_name=ref, fitting_method=fit_method,
                        emission_line=elem, emission_line_alignment=align_elem,
                        incident_energy_shift_keV=(e_shift*0.001), subtract_pre_edge_baseline = pre_edge,
                        alignment_enable = align, output_save_all = save_all, use_incident_energy_from_param_file = True)
                                                                             
                    
    def start_auto(self):
        self.pte_status.clear()
        self.pte_status.appendPlainText("live started")
        print('live started')
        cwd = self.le_wd.text()
        param = self.le_param.text()
        norm = self.le_sclr_2.text()
        sid_display = int(caget('XF:03IDC-ES{Status}ScanID-I'))
        self.pte_status.appendPlainText(f'current scan; {sid_display}')
        print(f'current scan; {sid_display}')
        QThread.sleep(5)
        self.pb_live.setStyleSheet("background-color: rgb(0, 93, 29)")
        QThread.sleep(10)
        hxn_auto_loader(wd = cwd,param_file_name = param,scaler_name=norm, 
                buffer_time = self.sb_buffer_time.value(), hdf = self.rb_make_hdf.isChecked(), 
                xrf_fit = self.rb_xrf_fit.isChecked())
        
        QThread.sleep(2)
                    
    def open_pyxrf(self):
           subprocess.Popen(['pyxrf'])

    def close_all_plots(self):
            return plt.close('all')
            
            
    def print_metadata(self):
            sid = int(self.le_sid_meta.text())
            h = db[sid]
            self.pte_status.clear()
            self.pte_status.appendPlainText(str(h.start))
            
    def print_dets(self):
            sid = int(self.le_sid_meta.text())
            h = db[sid]
            self.pte_status.clear()
            self.pte_status.appendPlainText(str(h.start['detectors']))

    def threadMaker(self, funct):
        # Pass the function to execute
        worker = Worker(funct)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # Execute
        self.threadpool.start(worker)


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error:`tuple` (exctype, value, traceback.format_exc() )
    - result: `object` data returned from processing, anything
    - progress: `tuple` indicating progress metadata
    '''
    start = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        self.signals.start.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


if __name__ == "__main__":

    

    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if (logger.hasHandlers()):
      logger.handlers.clear()
    logger.addHandler(stream_handler)
    
    app = QtWidgets.QApplication(sys.argv)
    window = xrf_3ID()
    window.show()
    sys.exit(app.exec_())
