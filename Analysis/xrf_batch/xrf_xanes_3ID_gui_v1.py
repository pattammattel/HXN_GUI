# conda activate analysis-2019-3.0-hxn-clone2

import sys, os, time, subprocess, logging, h5py, traceback, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pyqtgraph as pg
from glob import glob
from functools import wraps
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets, uic, QtTest, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from pyxrf.api import *
from epics import caget
from calcs import *

logger = logging.getLogger()
ui_path = os.path.dirname(os.path.abspath(__file__))


def show_error_message_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(None, "Error", error_message)
            pass
    return wrapper

def try_except_pass(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            pass
    return wrapper

@try_except_pass
def run_build_xanes_dict(param_dict):

    save_to_folder = os.path.join(param_dict["cwd"], f'{param_dict["first_sid"]}-{param_dict["last_sid"]}')
    if not os.path.exists(save_to_folder): 
        os.makedirs(save_to_folder)
        
    build_xanes_map(param_dict["first_sid"], 
                    param_dict["last_sid"], 
                    wd=save_to_folder,
                    xrf_subdir=save_to_folder, 
                    xrf_fitting_param_fln=param_dict["param"],
                    scaler_name=param_dict["norm"], 
                    sequence=param_dict["work_flow"],
                    ref_file_name=param_dict["ref"], 
                    fitting_method=param_dict["fit_method"],
                    emission_line=param_dict["elem"], 
                    emission_line_alignment=param_dict["align_elem"],
                    incident_energy_shift_keV=(param_dict["e_shift"] * 0.001),
                    subtract_pre_edge_baseline = param_dict["pre_edge"],
                    alignment_enable = param_dict["align"], 
                    output_save_all=param_dict["save_all"],
                    use_incident_energy_from_param_file=True,
                    skip_scan_types = ['FlyPlan1D'])

    plt.close()

    if param_dict["align"]:
        build_xanes_map(param_dict["first_sid"], 
                        param_dict["last_sid"], 
                        wd=save_to_folder,
                        xrf_subdir=save_to_folder, 
                        xrf_fitting_param_fln=param_dict["param"],
                        scaler_name=param_dict["norm"], 
                        sequence="build_xanes_map",
                        ref_file_name=param_dict["ref"], 
                        fitting_method=param_dict["fit_method"],
                        emission_line=param_dict["elem"], 
                        emission_line_alignment=param_dict["align_elem"],
                        incident_energy_shift_keV=(param_dict["e_shift"] * 0.001),
                        subtract_pre_edge_baseline = param_dict["pre_edge"],
                        alignment_enable = False, 
                        output_save_all=param_dict["save_all"],
                        use_incident_energy_from_param_file=True, 
                        skip_scan_types = ['FlyPlan1D'] )
        
        plt.close()


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
        self.pb_xrf_start.clicked.connect(lambda:self.create_pyxrf_batch_macro())
        #self.pb_live.clicked.connect(self.autoXRFThread)
        self.pb_live.clicked.connect(self.autoXRFThreadChunkMode)
        self.pb_stop_live.clicked.connect(self.stopAuto)
        self.pb_xanes_calib.clicked.connect(self.getCalibrationData)
        self.pb_plot_calib.clicked.connect(self.plotCalibration)
        self.pb_save_calib.clicked.connect(self.saveCalibration)

        #batchfiles
        self.pb_addTobBatch.clicked.connect(self.addToXANESBatchJob)
        self.pb_runBatch.clicked.connect(lambda:self.runBatchFile(
            os.path.join(ui_path,'xanes_batch_params.json'))
            )
        
        self.pb_create_batch_from_log.clicked.connect(
            lambda:self.run_xanes_batch_job_from_logfiles(
            file_filter_key = "nanoXANES",
            file_extention = "csv")
            )
        self.pb_showBatch.clicked.connect(lambda: self.pte_status.append(str(self.batchJob)))
        self.pb_clear_batch.clicked.connect(lambda: self.batchJob.clear())
        self.pb_stop_xanes_batch.clicked.connect(lambda:self.batch_xanes_thread.quit())

        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        self.pb_close_plots.clicked.connect(self.close_all_plots)

        self.pb_scan_meta.clicked.connect(self.print_metadata)
        self.pb_scan_dets.clicked.connect(self.print_dets)

        self.pb_select_calib_file.clicked.connect(self.get_calib_file)
        
        
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

    def get_calib_file(self):
        calib_fname = QFileDialog().getOpenFileName(self, "Open file", '', 'json file (*.json)')
        if calib_fname[0]:
            self.le_quant_calib_file.setText(calib_fname[0])

    def get_wd(self):
        dirname = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(dirname))

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

    @show_error_message_box
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

    @show_error_message_box
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
        
        return np.int_(sorted(scanNumbers))
    
    @show_error_message_box
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
                    "XRFfit":self.rb_xrf_fit.isChecked(),
                    "quant_calib_file":self.le_quant_calib_file.text(),
                    "quant_calib_elem":self.le_qunat_ref_elem.text()
                  }


        if self.rb_make_hdf.isChecked():
            self.h5thread = Loadh5AndFit(h5Param)
            self.h5thread.start()

        elif not self.rb_make_hdf.isChecked() and self.rb_xrf_fit.isChecked():
            
            xrf_batch_param_dict = {"sid_i":all_sid[0],
                                    "sid_f":all_sid[-1],
                                    "wd":cwd,
                                    "xrfParam":self.le_param.text(),
                                     "norm" :self.le_sclr_2.text(),
                                     "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
                                     "quant_calib_file":self.le_quant_calib_file.text(),
                                     "quant_calib_elem":self.le_qunat_ref_elem.text()}
            
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
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "quant_calib_file":self.le_quant_calib_file.text(),
            "quant_calib_elem":self.le_qunat_ref_elem.text()}

        self.xrf_thread = XRFFitThread(xrfParam)
        self.xrf_thread.start()
        
    @show_error_message_box
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
        out_file_ = os.path.join(ui_path,'xanes_batch_params.json')
        with open(out_file_, 'w') as outfile:
            json.dump(self.batchJob, outfile, indent=6)

        outfile.close()
        #self.pte_status.append(str(self.batchJob))

    def show_in_pte(self,str_to_show):
        self.pte_status.append(str(str_to_show))

    def runBatchFile(self, param_file):
        with open(param_file,'r') as infile:
            batch_job = json.load(infile)
        infile.close()

        if batch_job:
            self.xanes_batch_progress.setRange(0,0)

            self.batch_xanes_thread = XANESBatchProcessing(batch_job)
            self.batch_xanes_thread.current_process.connect(self.show_in_pte)
            self.batch_xanes_thread.finished.connect(lambda:self.xanes_batch_progress.setRange(0,100))
            self.batch_xanes_thread.finished.connect(lambda:self.xanes_batch_progress.setValue(100))
            self.batch_xanes_thread.start()


    def export_xanes_batch_param_file(self):
        out_file_path = os.path.join(self.le_wd.text(),"batch_xanes_params.json")
        export_file_name = QFileDialog.getSaveFileName(self,
                                "export_xanes_params",
                                out_file_path,
                                "All Files (*)")
        if export_file_name[0]:
            with open(out_file_path, 'w') as outfile:
                json.dump(self.batchJob, outfile, indent=6)
        else:
            return
        
    def run_xanes_batch_job_from_logfiles(self, file_filter_key = "nanoXANES",
                                      file_extention = "csv"):
        

        dirname = QFileDialog.getExistingDirectory(self, 
                                                   "Select Folder", 
                                                   self.le_wd.text(), )
        print(dirname)

        if dirname:
            logfiles = glob(os.path.join(dirname,
                                         f"*{file_filter_key}*.{file_extention}"))

        else:
            return 

        choice = QMessageBox.question(None,'Files Found',
                                      f"Files found are {logfiles}. \n Proceed?", 
                                      QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        
        if choice == QMessageBox.Yes:
            new_batch_job = {}
            for fname in sorted(logfiles):
                df = pd.read_csv(fname)
                sid_list = df["Scan ID"].dropna().to_numpy(dtype = int)

                temp_batch_file = self.createParamDictXANES()
                temp_batch_file['last_sid'] = int(sid_list[-1])
                temp_batch_file['first_sid'] = int(sid_list[0])
                new_batch_job[f"job_{len(new_batch_job)+1}"] = temp_batch_file

        else:
            return

        choice = QMessageBox.question(None,'Scans to process',
                                      f"The batch job is {new_batch_job}. \n Proceed?", 
                                      QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        
        if choice == QMessageBox.Yes:
            if new_batch_job:
                self.xanes_batch_progress.setRange(0,0)
                self.batch_xanes_thread = XANESBatchProcessing(new_batch_job)
                self.batch_xanes_thread.current_process.connect(self.show_in_pte)
                self.batch_xanes_thread.finished.connect(
                    lambda:self.xanes_batch_progress.setRange(0,100)
                    )
                self.batch_xanes_thread.finished.connect(
                    lambda:self.xanes_batch_progress.setValue(100)
                    )
                self.batch_xanes_thread.start()
        else:
            return

    def runSingleXANESJob(self):
        params = self.createParamDictXANES()
        self.xanes_thread = XANESProcessing(params)
        self.xanes_thread.start()

    def getCalibrationData(self):

        cwd = self.le_wd.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        if self.rb_Loadh5AndFit.isChecked():
            make_hdf(first_sid, last_sid, wd=cwd, 
                     file_overwrite_existing=self.rb_h5Overwrite.isChecked())
        else:
            QtTest.QTest.qWait(1)
            self.pte_status.append("Loading h5 From DataBroker is skipped ")
        #worker2 = Worker(getCalibSpectrum, path_ = cwd)
        #worker2.signals.result.connect(self.print_output)
        #list(map(worker2.signals.finished.connect, [self.thread_complete, self.plotCalibration]))
        self.calib_spec = getCalibSpectrum(path_= cwd)
        QtTest.QTest.qWait(1)
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
        self.scan_thread.requestInterruption()
        #self.scan_thread.wait()
        self.scan_thread.quit()
        self.pte_status.clear()
        QtTest.QTest.qWait(int(1000))
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

        self.scan_thread = ScanNumberStream(self.sb_chunk_size.value())
        #self.scan_thread.scan_num.connect(self.pyxrf_live_process)
        self.scan_thread.scan_num.connect(self.pyxrf_live_process) #thread this
        #self.scan_thread.scan_num.connect(self.pyxrf_live_process_batch) #thread this
        self.scan_thread.enableLiveButton.connect(self.liveButtonSts)
        self.scan_thread.start()
        print(f"Auto XRF Thread Running: {self.scan_thread.isRunning()}")

    def autoXRFThreadChunkMode(self):

        self.scan_thread = ScanListStream(self.sb_chunk_size.value())
        self.scan_thread.scan_list.connect(self.pyxrf_live_process_batch_for_live) #thread this
        self.scan_thread.enableLiveButton.connect(self.liveButtonSts)
        self.scan_thread.start()
        print(f"Auto XRF Thread Running: {self.scan_thread.isRunning()}")

    def make_hdf_live(self,wd,sid, skip1d = True):

        if not os.path.exists(os.path.join(wd,f"scan2D_{sid}.h5")):

            hdr = db[(sid)]
            if bool(hdr.stop):
                start_doc = hdr["start"]
                if not start_doc["plan_type"] in ("FlyPlan1D",):
                    attempt = 0
                    while attempt<3 and not os.path.exists(os.path.join(wd,f"scan2D_{sid}.h5")):
                        if skip1d:
                            make_hdf(int(sid), 
                                    wd=wd, 
                                    file_overwrite_existing=False,
                                    skip_scan_types=['FlyPlan1D']
                                    )

                        else:
                            make_hdf(int(sid), 
                                    wd=wd, 
                                    file_overwrite_existing=False,
                                    skip_scan_types=[]
                                    )

                        attempt+=1
                        logger.info(f"Data Not found attempt # {attempt}/3")
                        QtTest.QTest.qWait(2000)

            else: print("waiting for scan to complete")

    def pyxrf_live_process(self, sid):
        print(f"live process started : {sid}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        logger.info(f" Waiting for {self.sb_chunk_size.value()/2} seconds")
        QtTest.QTest.qWait(int(self.sb_chunk_size.value()/2)*1000)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()

        # for sid_ in (np.arange(sid-4,sid+1)):

        #TODO dont pass the scan if 1D checked unnecessry checkings otherwise

        if self.rb_make_hdf.isChecked():
            self.make_hdf_live(cwd,int(sid), skip1d=self.rb_skip1d.isChecked())
                
        if self.rb_xrf_fit.isChecked():
            param = self.le_param.text()
            fname = f'scan2D_{int(sid)}.h5'
            
            if os.path.exists(os.path.join(cwd,fname)):
                if not os.path.exists(os.path.join(cwd,f"output_tiff_scan2D_{sid}")):
                    fit_pixel_data_and_save(
                            cwd,
                            fname,
                            param_file_name = param,
                            scaler_name = norm,
                            save_tiff = self.rb_saveXRFTiff.isChecked(),
                            incident_energy = None,
                            ignore_datafile_metadata = True,
                            

                        )
                else:
                    pass

                self.pte_status.append(f"{sid} Fitted")
                QtTest.QTest.qWait(50)

            else:
                print(f"{fname} not found")
                pass
        else:
            
            pass
        
        #self.sb_lastScanProcessed.setValue(sid)
        QtTest.QTest.qWait(2000)
    
    def pyxrf_live_process_batch(self, sid_list):
        print(f"live process started : {sid_list}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()

        h5Param = {'sidList':sid_list,
            'wd':cwd,
            'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
            "xrfParam":self.le_param.text(),
            "norm" :norm,
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "XRFfit":self.rb_xrf_fit.isChecked()
            }

        if self.rb_make_hdf.isChecked():
                self.h5thread = Loadh5AndFit(h5Param)
                self.h5thread.start()
                #self.h5thread.last_processed.connect(self.sb_lastScanProcessed.setValue)

        elif not self.rb_make_hdf.isChecked() and self.rb_xrf_fit.isChecked():
            
            xrf_batch_param_dict = {"sid_i":sid_list[0],
                                    "sid_f":sid_list[-1],
                                    "wd":cwd,
                                    "xrfParam":self.le_param.text(),
                                    "norm" :self.le_sclr_2.text(),
                                    "saveXRFTiff": self.rb_saveXRFTiff.isChecked()}
            
            xrf_batch_param_dict = {"sid_i":all_sid[0],
                                    "sid_f":all_sid[-1],
                                    "wd":cwd,
                                    "xrfParam":self.le_param.text(),
                                     "norm" :self.le_sclr_2.text(),
                                     "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
                                     "quant_calib_file":self.le_quant_calib_file.text(),
                                     "quant_calib_elem":self.le_qunat_ref_elem.text()}
            
            self.pyxrfBatchThread = xrfBatchThread(xrf_batch_param_dict)
            self.pyxrfBatchThread.start()

        else:
            pass


    def pyxrf_live_process_batch_for_live(self, sid_list):
        print(f"live process started : {sid_list}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()

        h5Param = {'sidList':sid_list,
            'wd':cwd,
            'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
            "xrfParam":self.le_param.text(),
            "norm" :norm,
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "XRFfit":self.rb_xrf_fit.isChecked()
            }
        
        # xrf_batch_param_dict = {"sid_i":sid_list[0],
        #                 "sid_f":sid_list[-1],
        #                 "wd":cwd,
        #                 "xrfParam":self.le_param.text(),
        #                 "norm" :self.le_sclr_2.text(),
        #                 "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
        #                 "quant_calib_file":self.le_quant_calib_file.text(),
        #                 "quant_calib_elem":self.le_qunat_ref_elem.text()}
            
        self.xrf_batch_thread = Loadh5AndFit(h5Param)
        self.xrf_batch_thread.start()
        self.xrf_batch_thread.last_processed.connect(self.sb_last_sid_processed.setValue)
        # self.h5thread = loadh5Thread(h5Param)
        # self.h5thread.start()
        # self.h5thread.last_processed.connect(self.sb_lastScanProcessed.setValue)
        # self.h5thread.finished.connect(self.xrf_batch_thread.start) 



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
                    #thrd.wait()
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


class ScanListStream(QThread):
    scan_list = pyqtSignal(list)
    enableLiveButton = pyqtSignal(bool)
    def __init__(self, chunk_length):
        super().__init__()
        self.chunk_length = chunk_length
        self.scans_to_process = []

    def run(self):
        self.enableLiveButton.emit(False)
        timeout = time.time() + 60*60   # 60 minute intervals
        timeout_for_live = time.time() + 60*60*24*5 # max 5 days active loop 
        previous_list = [100,200]
        while True:
            QtTest.QTest.qWait(500)
            sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))

            if caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and not sid in self.scans_to_process and sid not in previous_list:
                self.scans_to_process.append(sid)
                logger.info(f"new scan added: {sid}; scan list to process{self.scans_to_process}")
                print(f"new scan added: {sid}; scan list to process = {self.scans_to_process}")
            if len(self.scans_to_process) == self.chunk_length or (time.time() > timeout and self.scan_list):
                self.scan_list.emit(self.scans_to_process)
                previous_list = self.scans_to_process
                self.scans_to_process = []
                timeout = time.time()+ 60*60 
                #QtTest.QTest.qWait(1000)

            if time.time() > timeout_for_live:
                break
                


class ScanNumberStream(QThread):
    scan_num = pyqtSignal(int)
    enableLiveButton = pyqtSignal(bool)
    def __init__(self, buffertime):
        super().__init__()
        self.buffertime = buffertime
    
    def run(self):
        self.enableLiveButton.emit(False)
        sid_sent = 100
        while True:
            QtTest.QTest.qWait(2000)
            sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
            hdr = db[int(sid)]
            start_doc = hdr["start"]
            if not start_doc["plan_type"] in ("FlyPlan1D",):
                if caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and not sid==sid_sent:
                
                    self.scan_num.emit(sid)
                    sid_sent = sid
                    logger.info(f"new scan signal sent: {sid}")
                    print(f"new scan signal sent: {sid}")
                    self.sleep(self.buffertime)
                    #QtTest.QTest.qWait(5000)
                
class scanStatus(QThread):
    scan_sts = pyqtSignal(int)
    scan_num = pyqtSignal(int)

    def run(self):
        while True:
            QtTest.QTest.qWait(2000)
            
            self.scan_num.emit(int(caget('XF:03IDC-ES{Status}ScanID-I')))
            self.scan_sts.emit(caget('XF:03IDC-ES{Status}ScanRunning-I'))

class XANESProcessing(QThread):
    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict

    def run(self):
        run_build_xanes_dict(self.paramDict)

class XANESBatchProcessing(QThread):

    current_process = pyqtSignal(dict)
    current_iter = pyqtSignal(int)

    def __init__(self, batch_job_dict):
        super().__init__()
        self.batch_job_dict = batch_job_dict
        self.paramDict = {}

    def run(self):
        n = 0 
        for key, value in self.batch_job_dict.items():

            save_dict = self.paramDict
            self.paramDict = value
            self.current_process.emit(self.paramDict)
            n = +1
            self.current_iter.emit(n)

            try:
                run_build_xanes_dict(self.paramDict)
                
                h = db[int(save_dict["first_sid"])]
                start_doc = h["start"]

                save_dict["n_points"] = (start_doc["num1"],start_doc["num2"])
                save_dict["exposure_time_sec"] = start_doc["exposure_time"]
                save_dict["step_size_um"] = start_doc["per_points"]

                outfile = os.path.join(f'{save_dict["cwd"]}/{save_dict["first_sid"]}-{save_dict["last_sid"]}',f'{save_dict["first_sid"]}_{save_dict["last_sid"]}.json')
                with open(outfile, "w") as fp:
                    json.dump(save_dict,fp, indent=6)
            except:
                pass

class loadh5Thread(QThread):
    h5loaded = pyqtSignal(int)
    last_processed = pyqtSignal(int)

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict
        self.missed_scans = []

    def run(self):
        logger.info("h5 thread started")
        QtTest.QTest.qWait(5000)
        for sid in self.paramDict["sidList"]: #filter for 1d

            if not os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5")):
                hdr = db[int(sid)]
                start_doc = hdr["start"]
                if not start_doc["plan_type"] in ("FlyPlan1D",):

                    make_hdf(
                        int(sid), 
                        wd = self.paramDict["wd"],
                        file_overwrite_existing = self.paramDict['file_overwrite_existing'],
                        create_each_det = True,
                        skip_scan_types = ['FlyPlan1D']
                        )




class Loadh5AndFit(QThread):
    
    h5loaded = pyqtSignal(int)
    last_processed = pyqtSignal(int)

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict
        self.missed_scans = []


    def run(self):
        logger.info("h5 thread started")
        QtTest.QTest.qWait(500)
        for sid in self.paramDict["sidList"]: #filter for 1d

            try:

                if not os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5")):
                    hdr = db[int(sid)]
                    start_doc = hdr["start"]
                    if not start_doc["plan_type"] in ("FlyPlan1D",):

                        make_hdf(
                            int(sid), 
                            wd = self.paramDict["wd"],
                            file_overwrite_existing = self.paramDict['file_overwrite_existing'],
                            create_each_det = True,
                            skip_scan_types = ['FlyPlan1D']
                            )

                        QtTest.QTest.qWait(1000)
                        if self.paramDict["XRFfit"]:

                            fname = f"scan2D_{int(sid)}.h5"

                            if os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5")):

                                print("batch fitting try")

                                pyxrf_batch(int(sid), 
                                            int(sid), 
                                            wd=self.paramDict["wd"], 
                                            param_file_name=self.paramDict["xrfParam"], 
                                            scaler_name=self.paramDict["norm"], 
                                            save_tiff=self.paramDict["saveXRFTiff"],
                                            save_txt = False,
                                            ignore_datafile_metadata = True,
                                            fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
                                            quant_ref_eline = self.paramDict.get("quant_calib_elem",'')
                                            )
                            else:
                                raise FileNotFoundError(f"{fname} not found")
                            
                        self.last_processed.emit(sid)
                    else:
                        pass

                        

            except KeyError:
                print(" Data transfer is not completed; scan number added to next iteration")
                self.missed_scans.append(sid)
                pass

class XRFFitThread(QThread):
    
    def __init__(self, paramDict):
        super().__init__()
        
        self.paramDict = paramDict

    def run(self):
        sid = self.paramDict["sid"]

        fname = f"scan2D_{int(sid)}.h5"

        pyxrf_batch(int(sid), 
                    int(sid), 
                    wd=self.paramDict["wd"], 
                    param_file_name=self.paramDict["xrfParam"], 
                    scaler_name=self.paramDict["norm"], 
                    save_tiff=self.paramDict["saveXRFTiff"],
                    save_txt = False,
                    ignore_datafile_metadata = True,
                    fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
                    quant_ref_eline = self.paramDict.get("quant_calib_elem",'')
                    )

        QtTest.QTest.qWait(500)

        
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
            ignore_datafile_metadata = True,
            fln_quant_calib_data = self.paramDict.get("quant_calib_file", ''),
            quant_ref_eline = self.paramDict.get("quant_calib_elem", '')
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
