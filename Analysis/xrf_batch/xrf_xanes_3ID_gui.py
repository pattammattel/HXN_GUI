# conda activate analysis-2019-3.0-hxn-clone2

import sys, os, time, subprocess, logging, h5py, traceback, json
import matplotlib.pyplot as plt
import numpy as np
import time
import pyqtgraph as pg

from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets, uic, QtTest, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from pyxrf.api import *
from epics import caget
from calcs import *

logger = logging.getLogger()
ui_path = os.path.dirname(os.path.abspath(__file__))


class xrf_3ID(QtWidgets.QMainWindow):
    def __init__(self):
        super(xrf_3ID, self).__init__()
        uic.loadUi(os.path.join(ui_path, "xrf_xanes_3ID_gui.ui"), self)

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

        self.pyxrf_subprocess = None

        self.pb_wd.clicked.connect(self.get_wd)
        self.pb_param.clicked.connect(self.get_param)
        self.pb_ref.clicked.connect(self.get_ref_file)
        self.batchJob = {}

        self.pb_start.clicked.connect(self.runSingleXANESJob)
        self.pb_xrf_start.clicked.connect(self.create_pyxrf_batch_macro)
        self.pb_live.clicked.connect(self.autoXRFThread)
        self.pb_stop_live.clicked.connect(self.stopAuto)
        self.pb_xanes_calib.clicked.connect(self.getCalibrationData)
        self.pb_plot_calib.clicked.connect(self.plotCalibration)
        self.pb_save_calib.clicked.connect(self.saveCalibration)

        #batchfiles
        self.pb_addTobBatch.clicked.connect(self.addToXANESBatchJob)
        self.pb_runBatch.clicked.connect(self.runBatchFile)
        self.pb_showBatch.clicked.connect(lambda: self.pte_status.append(str(self.batchJob)))
        self.pb_clear_batch.clicked.connect(lambda: self.batchJob.clear())

        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        self.pb_close_plots.clicked.connect(self.close_all_plots)

        self.pb_scan_meta.clicked.connect(self.print_metadata)
        self.pb_scan_dets.clicked.connect(self.print_dets)
        
        
        #load previous config

        line_edits =  [self.le_XRFBatchSID,
                        self.le_wd,
                        self.le_param,
                        self.le_startid,
                        self.le_lastid,
                        self.xanes_elem,
                        self.alignment_elem]


        for le in line_edits:
            le.textEdited.connect(self.save_config)

        self.load_config(os.path.join(ui_path,"config_file.json"))

        
        #threds
        self.scan_thread = QThread()
        self.scan_sts_thread = QThread()
        self.xanes_thread = QThread()
        self.h5thread = QThread()

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum  {self.threadpool.maxThreadCount()} threads")


        #liveupdates
        self.startScanStatusThread()
        self.last_sid = 100

        self.show()

    def __del__(self):
        import sys
        # Restore sys.stdout
        sys.stdout = sys.__stdout__


    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()


    def errorOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()


    def get_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(folder_path))

    def get_param(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'json file (*.json)')
        self.le_param.setText(str(file_name[0]))

    def get_ref_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'ref file (*.txt, *.nor, *.csv)')
        self.le_ref.setText(str(file_name[0]))


    def save_config(self):

        self.config = {"xrf_scan_range":self.le_XRFBatchSID.text(),
                       "wd":self.le_wd.text(),
                       "param_file":self.le_param.text(),
                       "xanes_start_id":self.le_startid.text(),
                       "xanes_end_id":self.le_lastid.text(),
                       "xanes_elem":self.xanes_elem.text(),
                       "alignment_elem":self.alignment_elem.text()
                       }

        with open(os.path.join(ui_path, "config_file.json"), "w") as fp:

            json.dump(self.config, fp, indent = 4)


    def load_config(self, json_file):

        if json_file:

            with open(json_file, 'r') as fp:
                self.config = json.load(fp)
            
            try:
                self.le_XRFBatchSID.setText(self.config["xrf_scan_range"]),
                self.le_wd.setText(self.config["wd"]),
                self.le_param.setText(self.config["param_file"]),
                self.le_startid.setText(self.config["xanes_start_id"]),
                self.le_lastid.setText(self.config["xanes_end_id"]),
                self.xanes_elem.setText(self.config["xanes_elem"]),
                self.alignment_elem.setText(self.config["alignment_elem"])

            except:
                pass


        else:
            pass

    
    def parseScanRange(self,str_scan_range):
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
        
        return np.int_(scanNumbers)
    
    def create_pyxrf_batch_macro(self):

        cwd = self.le_wd.text()
        all_sid = self.parseScanRange(self.le_XRFBatchSID.text())
        QtTest.QTest.qWait(500)
        self.pte_status.append(f"scans to process {all_sid}")
        QtTest.QTest.qWait(500)

        h5Param = {'sidList':all_sid,
                   'wd':cwd,
                   'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
                    "xrfParam":self.le_param.text(),
                    "norm" :self.le_sclr_2.text(),
                    "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
                    "XRFfit":self.rb_xrf_fit.isChecked()
                  }


        if self.rb_make_hdf.isChecked():
            self.h5thread = loadh5(h5Param)
            self.h5thread.start()

        elif not self.rb_make_hdf.isChecked() and self.rb_xrf_fit.isChecked():
            
            xrf_batch_param_dict = {"sid_i":all_sid[0],
                                    "sid_f":all_sid[-1],
                                    "wd":cwd,
                                    "xrfParam":self.le_param.text(),
                                     "norm" :self.le_sclr_2.text(),
                                     "saveXRFTiff": self.rb_saveXRFTiff.isChecked()}
            
            self.pyxrfBatchThread = xrfBatchThread(xrf_batch_param_dict)
            self.pyxrfBatchThread.start()

            '''
            for sid in all_sid:
                fname = f"scan2D_{int(sid)}.h5"
                if os.path.isfile(os.path.join(cwd,fname)):
                    self.xrfThread(sid)
                else:
                    print(f"{fname} not found")
            '''
        else:
            pass

    def stopXRFBatch(self):
        self.h5thread.quit()

    def xrfThread(self,sid):

        xrfParam = {
            "sid":sid, 
            "wd":self.le_wd.text(),
            "xrfParam":self.le_param.text(),
            "norm":self.le_sclr_2.text(),
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked()}

        self.xrf_thread = XRFFitThread(xrfParam)
        self.xrf_thread.start()
        

    def createParamDictXANES(self):

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

        self.pte_status.append(str(build_xanes_map_param))

        return build_xanes_map_param

    def addToXANESBatchJob(self):
        self.batchJob[f"job_{len(self.batchJob)+1}"] = self.createParamDictXANES()
        self.pte_status.append(str(self.batchJob))

    def runBatchFile(self):
        if self.batchJob:
            for value in self.batchJob.values():
                #plt.close('all')
                self.create_xanes_macro(value)

    def create_xanes_macro(self,param_dict):

        self.xanes_thread = XANESProcessing(param_dict)
        self.xanes_thread.start()
        
        '''
        build_xanes_map(param_dict["first_sid"], param_dict["last_sid"], wd=param_dict["cwd"],
                        xrf_subdir=param_dict["cwd"], xrf_fitting_param_fln=param_dict["param"],
                        scaler_name=param_dict["norm"], sequence=param_dict["work_flow"],
                        ref_file_name=param_dict["ref"], fitting_method=param_dict["fit_method"],
                        emission_line=param_dict["elem"], emission_line_alignment=param_dict["align_elem"],
                        incident_energy_shift_keV=(param_dict["e_shift"] * 0.001),
                        subtract_pre_edge_baseline = param_dict["pre_edge"],
                        alignment_enable = param_dict["align"], output_save_all=param_dict["save_all"],
                        use_incident_energy_from_param_file=True )
        '''
    
    def runSingleXANESJob(self):
        params = self.createParamDictXANES()
        self.create_xanes_macro(params)

    def getCalibrationData(self):

        cwd = self.le_wd.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        if self.rb_loadh5.isChecked():
            make_hdf(first_sid, last_sid, wd=cwd, file_overwrite_existing=self.rb_h5Overwrite.isChecked())
        else:
            QtTest.QTest.qWait(0.1)
            self.pte_status.append("Loading h5 From DataBroker is skipped ")
        #worker2 = Worker(getCalibSpectrum, path_ = cwd)
        #worker2.signals.result.connect(self.print_output)
        #list(map(worker2.signals.finished.connect, [self.thread_complete, self.plotCalibration]))
        self.calib_spec = getCalibSpectrum(path_= cwd)
        QtTest.QTest.qWait(0.1)
        self.pte_status.append(str("calibration spec available"))
        #np.savetxt(os.path.join(self.le_wd.text(), "calibration_spec.txt"), self.calib_spec)
        self.plotCalibration()

    def plotCalibration(self):
        if self.rb_calib_derivative.isChecked():
            pg.plot(self.calib_spec[:, 0], np.gradient(self.calib_spec[:, 1]),
                    pen = pg.mkPen(pg.mkColor(0,0,255,255), width=3),
                    symbol='o',symbolSize = 6,symbolBrush = 'r', title = "Calibration Spectrum")
        else:
            pg.plot(self.calib_spec[:, 0], self.calib_spec[:, 1],
                    pen = pg.mkPen(pg.mkColor(0,0,255,255), width=3),
                    symbol='o',symbolSize = 6,symbolBrush = 'r', title = "Calibration Spectrum")

    def saveCalibration(self):
        file_name = QFileDialog().getSaveFileName(self, "Save Calibration", '', 'txt file (*.txt)')
        if file_name[0]:
            np.savetxt(file_name[0], self.calib_spec, fmt = '%.5f')
        else:
            pass

    def stopAuto(self):
        self.scan_thread.quit()
        self.pte_status.clear()
        self.lbl_live_sts_msg.setText("  Live processing is OFF  ")
        self.lbl_live_sts_msg.setStyleSheet("background-color: yellow")
        self.pb_live.setEnabled(True)
        print(f"Thread Running: {self.scan_thread.isRunning()}")

    def liveButtonSts(self, sts):
        self.pb_live.setEnabled(sts)
        self.lbl_live_sts_msg.setText("  Live processing is ON  ")
        self.lbl_live_sts_msg.setStyleSheet("background-color: lightgreen")

    def scanStatusUpdate(self,sts):
        
        if sts == 1:
            self.lbl_scan_status.setText("    Scan in Progress    ")
            self.lbl_scan_status.setStyleSheet("background-color: lightgreen")

        else:
            self.lbl_scan_status.setText("    Run Engine is Idle    ")
            self.lbl_scan_status.setStyleSheet("background-color: yellow")
         
    def startScanStatusThread(self):

        self.scan_sts_thread = scanStatus()
        self.scan_sts_thread.scan_sts.connect(self.scanStatusUpdate)
        self.scan_sts_thread.scan_num.connect(self.sb_scan_number.setValue)
        self.scan_sts_thread.start()

    def autoXRFThread(self):

        self.scan_thread = liveData(self.sb_buffer_time.value())
        #self.scan_thread.scan_num.connect(self.pyxrf_live_process)
        self.scan_thread.scan_num.connect(self.pyxrf_live_process) #thread this
        self.scan_thread.enableLiveButton.connect(self.liveButtonSts)
        self.scan_thread.start()
        print(f"Auto XRF Thread Running: {self.scan_thread.isRunning()}")

    def pyxrf_live_process(self, sid):
        invalidData = False
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        print(f" Waiting for {self.sb_buffer_time.value()/2} seconds")
        QtTest.QTest.qWait((self.sb_buffer_time.value()/2)*1000)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()
        hdr = db[(sid)]

        if self.rb_make_hdf.isChecked() and bool(hdr.stop) and sid != self.last_sid:
            attempt = 0
            while attempt<20:
                try:
                    make_hdf(int(sid), 
                            wd=cwd, 
                            file_overwrite_existing=True
                            )
                #except KeyError:

                except UnboundLocalError:
                    print("Invalid Datatype")
                    invalidData = True
                    pass

                except RuntimeError:
                    print("Invalid Datatype")
                    invalidData = True
                    pass
                
                except:
                    attempt+=1
                    print(f"Data Not found attempt # {attempt}/20")
                    QtTest.QTest.qWait(6*1000)
                    continue
                break

            self.pte_status.append(f"{sid} h5 loaded")
            QtTest.QTest.qWait(50)
                
        if self.rb_xrf_fit.isChecked() and not invalidData and sid != self.last_sid:
            param = self.le_param.text()
            fname = f'scan2D_{int(sid)}.h5'
            fit_pixel_data_and_save(
                    cwd,
                    fname,
                    param_file_name = param,
                    scaler_name = norm,
                    save_tiff = self.rb_saveXRFTiff.isChecked()
                )

            self.last_sid = sid

            invalidData = False
            self.pte_status.append(f"{sid} Fitted")
            QtTest.QTest.qWait(50)

            #pyxrf_batch(sid, sid, wd=cwd, param_file_name=param, scaler_name=norm, save_tiff=True)
        else:
            
            self.pte_status.append(f"Error processing : {self.last_sid}")
            QtTest.QTest.qWait(500)
            pass
        
        self.sb_lastScanProcessed.setValue(sid)
        QtTest.QTest.qWait(500)


    def open_pyxrf(self):
        os.system('gnome-terminal --tab --command pyxrf --active')
        #self.pyxrf_subprocess = subprocess.Popen(['pyxrf'])

    def close_all_plots(self):
        return plt.close('all')

    def print_metadata(self):
        sid = int(self.le_sid_meta.text())
        h = db[sid]
        self.pte_status.clear()
        self.pte_status.append(str(h.start))

    def print_dets(self):
        sid = int(self.le_sid_meta.text())
        h = db[sid]
        self.pte_status.clear()
        self.pte_status.append(str(h.start['detectors']))

    # Thread Signals

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def threadMaker(self, funct):
        # Pass the function to execute
        worker = Worker(funct)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # Execute
        self.threadpool.start(worker)

    def closeEvent(self,event):

        for thrd in [self.scan_thread,self.scan_sts_thread,self.xanes_thread,self.h5thread,self.xrfThread]:
            if not thrd == None:
                if thrd.isRunning():
                    thrd.quit()
                    QtTest.QTest.qWait(500)
                    thrd.quit()
        if not self.pyxrf_subprocess == None:
            if self.pyxrf_subprocess.poll() is None:
                self.pyxrf_subprocess.kill()
        
        sys.exit()


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
        Initialize the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        self.signals.start.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class liveData(QThread):
    scan_num = pyqtSignal(int)
    enableLiveButton = pyqtSignal(bool)
    def __init__(self, buffertime):
        super().__init__()
        self.buffertime = buffertime
    
    def run(self):
        self.enableLiveButton.emit(False)
        while True:
            QtTest.QTest.qWait(333)
            while caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and caget('XF:03IDC-ES{Sclr:2}_cts1.D')>5000:

                self.scan_num.emit(int(caget('XF:03IDC-ES{Status}ScanID-I')))
                print("new scan signal sent")
                self.sleep(self.buffertime)
                #QtTest.QTest.qWait(self.buffertime*1000)
                
class scanStatus(QThread):
    scan_sts = pyqtSignal(int)
    scan_num = pyqtSignal(int)

    def run(self):
        while True:
            QtTest.QTest.qWait(500)
            
            self.scan_num.emit(int(caget('XF:03IDC-ES{Status}ScanID-I')))
            self.scan_sts.emit(caget('XF:03IDC-ES{Status}ScanRunning-I'))

class XANESProcessing(QThread):
    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict

    def run(self):

        build_xanes_map(
            self.paramDict["first_sid"], 
            self.paramDict["last_sid"], 
            wd=self.paramDict["cwd"],
            xrf_subdir=self.paramDict["cwd"], 
            xrf_fitting_param_fln=self.paramDict["param"],
            scaler_name=self.paramDict["norm"], 
            sequence=self.paramDict["work_flow"],
            ref_file_name=self.paramDict["ref"], 
            fitting_method=self.paramDict["fit_method"],
            emission_line=self.paramDict["elem"], 
            emission_line_alignment=self.paramDict["align_elem"],
            incident_energy_shift_keV=(self.paramDict["e_shift"] * 0.001),
            subtract_pre_edge_baseline = self.paramDict["pre_edge"],
            alignment_enable = self.paramDict["align"], 
            output_save_all=self.paramDict["save_all"],
            use_incident_energy_from_param_file=True ,
            skip_scan_types = ['FlyPlan1D']
            )

class loadh5(QThread):
    
    h5loaded = pyqtSignal(int)

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict


    def run(self):
        print("h5 thread started")
        
        for sid in self.paramDict["sidList"]: #filter for 1d
            

            #try to find the data in the db, except pass
            try:
                hdr = db[int(sid)]
                start_doc = hdr["start"]

                #skip 1D scans
                if not start_doc["plan_type"] in ("FlyPlan1D",):
                    
                    
                    #try to make h5, bypass any errors
                    try: 

                        make_hdf(
                            int(sid), 
                            wd = self.paramDict["wd"],
                            file_overwrite_existing = self.paramDict['file_overwrite_existing'],
                            create_each_det = True,
                            skip_scan_types = ['FlyPlan1D']
                            )
                    
                           
                        #self.h5loaded.emit(sid)
                        QtTest.QTest.qWait(50)
                        #if h5 available tracker to do xrf fitting
                        h5DataAvailable = True

                    except:
                        print(f" Failed to load {sid}")
                        h5DataAvailable = False
                        pass


                    if self.paramDict["XRFfit"] and h5DataAvailable:

                        try:
                            fname = f"scan2D_{int(sid)}.h5"


                            fit_pixel_data_and_save(
                                    self.paramDict["wd"],
                                    fname,
                                    param_file_name = self.paramDict["xrfParam"],
                                    scaler_name = self.paramDict["norm"],
                                    save_tiff = self.paramDict["saveXRFTiff"]
                                    )

                        except:
                            print("XRF Fitting Unsuccessful")
                            pass

                    else:
                        pass
            except:
                print("Trouble finding data from data broker")
                pass

class XRFFitThread(QThread):
    
    def __init__(self, paramDict):
        super().__init__()
        
        self.paramDict = paramDict

    def run(self):
        sid = self.paramDict["sid"]

        fname = f"scan2D_{int(sid)}.h5"

        fit_pixel_data_and_save(
                self.paramDict["wd"],
                fname,
                param_file_name = self.paramDict["xrfParam"],
                scaler_name = self.paramDict["norm"],
                save_tiff = self.paramDict["saveXRFTiff"],
                save_txt = False
                )

        QtTest.QTest.qWait(500)
        self.quit()

        
class xrfBatchThread(QThread):
    def __init__(self, paramDict):
        super().__init__()
            
        self.paramDict = paramDict

    def run(self):
        sid_i = self.paramDict["sid_i"]
        sid_f = self.paramDict["sid_f"]


        pyxrf_batch(self.paramDict["sid_i"], 
            self.paramDict["sid_f"], 
            wd=self.paramDict["wd"], 
            param_file_name=self.paramDict["xrfParam"], 
            scaler_name=self.paramDict["norm"], 
            save_tiff=self.paramDict["saveXRFTiff"],
            save_txt = False,
            ignore_datafile_metadata = True
            )

class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

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
