"""
 __Author__: Ajith Pattammattel
 Original Date:06-23-2020
 Last Update: 12-02-2022

 """

import os
import signal
import subprocess
import sys
import collections
import webbrowser
import pyqtgraph as pg
import json
import re
from functools import partial 
from scipy.ndimage import rotate
from epics import caget, caput


from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QLCDNumber, QLabel, QErrorMessage, QPushButton
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QDate, QTime

#import custom functions
from HXNSampleExchange import *
from exceptions import *
from element_lines import *
HXNSampleExchanger = SampleExchangeProtocol()

ui_path = os.path.dirname(os.path.abspath(__file__))

class Ui(QtWidgets.QMainWindow):
   
    def __init__(self):
        super(Ui, self).__init__()

        print("Loading UI... Please wait")
        
        uic.loadUi(os.path.join(ui_path,'hxn_gui_v5.ui'), self)
        print("UI File loaded")


        self.initParams()
        self.client = webbrowser.get('firefox')
        self.threadpool = QThreadPool()
        #self.tw_hxn_contact.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

        self.energies = []
        self.roiDict = {}

        #user setup
        self.xrf_combo_boxes = self.xrf_elem_cb_widget.findChildren(QtWidgets.QComboBox)

        for cb in self.xrf_combo_boxes:
            cb.addItems(all_elem_lines)
            cb.currentIndexChanged.connect(self.populate_elems_from_combobox)

        self.pb_create_user.clicked.connect(lambda:self.new_user_setup())
        self.pb_update_xrf_elems.clicked.connect(self.apply_xrf_elems)
        self.import_xrf_elem_list(auto = True)
        self.pb_import_xrf_elem_list.clicked.connect(lambda:self.import_xrf_elem_list(auto = False))
        self.pb_export_xrf_elem_list.clicked.connect(lambda:self.export_xrf_elem_list(auto = False))
        self.sb_live_elem_num.valueChanged.connect(self.populate_elems_from_combobox)

        #flyscan
        self.motor_list = {'zpssx': zpssx, 'zpssy': zpssy, 'zpssz': zpssz,
        'dssx': dssx, 'dssy': dssy, 'dssz': dssz} 

        self.dwell.valueChanged.connect(self.initParams)
        self.x_step.valueChanged.connect(self.initParams)
        self.y_step.valueChanged.connect(self.initParams)
        self.x_start.valueChanged.connect(self.initParams)
        self.y_start.valueChanged.connect(self.initParams)
        self.x_end.valueChanged.connect(self.initParams)
        self.y_end.valueChanged.connect(self.initParams)
        self.pb_dets.currentTextChanged.connect(self.initParams)        
        self.cb_motor1.currentTextChanged.connect(self.initParams)
        self.cb_motor2.currentTextChanged.connect(self.initParams)
        print("Fly Motors Connected")

        # logic control for 1d or 2d scan selection
        self.rb_1d.clicked.connect(self.disableMot2)
        self.rb_2d.clicked.connect(self.enableMot2)
        self.rb_1d.clicked.connect(self.initParams)
        self.rb_2d.clicked.connect(self.initParams)

        #Abort scan plan
        #self.pb_reqPause.clicked.connect(self.requestScanPause)
        #self.pb_REAbort.clicked.connect(self.scanAbort)
        #self.pb_REResume.clicked.connect(self.scanResume)

        #shutters
        self.pb_open_b_shutter.clicked.connect(lambda:caput("XF:03IDB-PPS{PSh}Cmd:Opn-Cmd", 1))
        self.pb_close_b_shutter.clicked.connect(lambda:caput("XF:03IDB-PPS{PSh}Cmd:Cls-Cmd", 1))
        self.pb_open_c_shutter.clicked.connect(lambda:caput("XF:03IDC-ES{Zeb:2}:SOFT_IN:B0", 1))
        self.pb_close_c_shutter.clicked.connect(lambda:caput("XF:03IDC-ES{Zeb:2}:SOFT_IN:B0", 0))


        # plotting controls
        self.pb_close_all_plot.clicked.connect(self.close_all_plots)
        self.pb_plot.clicked.connect(self.plot_me)
        self.pb_erf_fit.clicked.connect(self.plot_erf_fit)
        self.pb_plot_line_center.clicked.connect(self.plot_line_center)

        #energy change

        #print("energy_change")
        self.pb_move_energy.clicked.connect(lambda:self.change_energy(self.dsb_target_e.value()))
        self.dsb_target_e.valueChanged.connect(self.update_energy_calc)
        self.sb_harmonic.valueChanged.connect(self.update_energy_calc)


        # scans and motor motion
        #flyscan
        self.start.clicked.connect(lambda:self.initFlyScan())

        #mll navigation
        self.pb_move_dsx_pos.clicked.connect(lambda:self.move_dsx())
        self.pb_move_dsy_pos.clicked.connect(lambda:self.move_dsy())
        self.pb_move_dsz_pos.clicked.connect(lambda:self.move_dsz())
        self.pb_move_dth_pos.clicked.connect(lambda:self.move_dsth())
        self.pb_move_sbz_pos.clicked.connect(lambda:self.move_zpz1())

        self.pb_move_dsx_neg.clicked.connect(lambda: self.move_dsx(neg_=True))
        self.pb_move_dsy_neg.clicked.connect(lambda: self.move_dsy(neg_=True))
        self.pb_move_dsz_neg.clicked.connect(lambda: self.move_dsz(neg_=True))
        self.pb_move_dth_pos_neg.clicked.connect(lambda: self.move_dsth(neg_=True))
        self.pb_move_sbz_neg.clicked.connect(lambda: self.move_zpz1(neg_=True))

        #zp navigation
        self.pb_move_smarx_pos.clicked.connect(lambda:self.move_smarx())
        self.pb_move_smary_pos.clicked.connect(lambda:self.move_smary())
        self.pb_move_smarz_pos.clicked.connect(lambda:self.move_smarz())
        self.pb_move_zpsth_pos.clicked.connect(lambda:self.move_zpth())
        self.pb_move_zpz_pos.clicked.connect(lambda:self.move_zpz1())

        self.pb_move_smarx_neg.clicked.connect(lambda: self.move_smarx(neg_=True))
        self.pb_move_smary_neg.clicked.connect(lambda: self.move_smary(neg_=True))
        self.pb_move_smarz_neg.clicked.connect(lambda: self.move_smarz(neg_=True))
        self.pb_move_zpsth_pos_neg.clicked.connect(lambda: self.move_zpth(neg_=True))
        self.pb_move_zpz_neg.clicked.connect(lambda: self.move_zpz1(neg_=True))

        # Detector/Camera Motions

        #Merlin
        self.pb_merlinOUT.clicked.connect(lambda:self.merlinOUT())
        self.pb_merlinIN.clicked.connect(lambda:self.merlinIN())
        self.pb_stop_merlin.clicked.connect(lambda:diff.stop())
        #fluorescence det
        self.pb_vortexOUT.clicked.connect(lambda:self.vortexOUT())
        self.pb_vortexIN.clicked.connect(lambda:self.vortexIN())
        #self.pb_stop_fluor.clicked.connect(lambda:caput("XF:03IDC-ES{Det:Vort-Ax:X}Mtr.STOP",1))
        self.pb_stop_fluor.clicked.connect(lambda:fdet1.x.stop())
        #cam06
        self.pb_cam6IN.clicked.connect(lambda:self.cam6IN())
        self.pb_cam6OUT.clicked.connect(lambda:self.cam6OUT())
        self.pb_CAM6_IN.clicked.connect(lambda:self.cam6IN())
        self.pb_CAM6_OUT.clicked.connect(lambda:self.cam6OUT())
        #cam11
        self.pb_cam11IN.clicked.connect(lambda:self.cam11IN())
        self.pb_stop_cam11.clicked.connect(lambda:diff.stop())
        #dexela
        self.pb_dexela_IN.clicked.connect(lambda:self.dexela_motion(move_to = 0))
        self.pb_dexela_OUT.clicked.connect(lambda:self.dexela_motion(move_to = 400))
        self.pb_stop_dexela.clicked.connect(lambda:caput("XF:03IDC-ES{Stg:FPDet-Ax:Y}Mtr.STOP",1))

        #light
        self.pb_light_ON.clicked.connect(lambda:caput("XF:03IDC-ES{PDU:1}Cmd:Outlet8-Sel", 0))
        self.pb_light_OFF.clicked.connect(lambda:caput("XF:03IDC-ES{PDU:1}Cmd:Outlet8-Sel", 1))

        #sample exchange
        self.pb_start_vent.clicked.connect(lambda:self.vent_in_thread(750))
        self.pb_stop_vent.clicked.connect(HXNSampleExchanger.stop_vending)
        self.pb_start_pump.clicked.connect(lambda:self.start_pumping_in_thread(350))
        self.pb_stop_pump.clicked.connect(self.pumping_stop)
        self.pb_auto_he_fill.clicked.connect(self.he_backfill_in_thread)
        self.pb_open_fast_pumps.clicked.connect(HXNSampleExchanger.start_fast_pumps)

        #alignment
        #SSA2 motion
        self.pb_SSA2_Open.clicked.connect(lambda:self.SSA2_Pos(2.1, 2.1))
        self.pb_SSA2_Close.clicked.connect(lambda:self.SSA2_Pos(0.05, 0.03))
        self.pb_SSA2_Close.clicked.connect(lambda:self.SSA2_Pos(0.05, 0.03))
        self.pb_SSA2_HClose.clicked.connect(lambda:self.SSA2_Pos(0.1, 2.1))
        self.pb_SSA2_VClose.clicked.connect(lambda:self.SSA2_Pos(2.1, 0.1))
        self.pb_move_ssa2.clicked.connect(lambda:self.SSA2_Pos(self.db_ssa2_x_set.value()
                                                              ,self.db_ssa2_y_set.value()))

        #s5 slits
        self.pb_S5_Open.clicked.connect(lambda:self.S5_Pos(4,4))
        self.pb_S5_Close.clicked.connect(lambda:self.S5_Pos(0.3,0.3))
        self.pb_S5_HClose.clicked.connect(lambda:self.S5_Pos(0.1,0.3))
        self.pb_S5_VClose.clicked.connect(lambda:self.S5_Pos(0.3,0.1))
        self.pb_move_s5.clicked.connect(lambda:self.S5_Pos(self.db_s5_x_set.value()
                                                              ,self.db_s5_y_set.value()))

        #front end
        self.pb_FS_IN.clicked.connect(lambda:self.FS_IN())
        self.pb_FS_OUT.clicked.connect(lambda:self.FS_OUT())

        #OSA Y Pos
        self.pb_osa_out.clicked.connect(lambda:self.ZP_OSA_OUT())
        self.pb_osa_in.clicked.connect(lambda:self.ZP_OSA_IN())

        #alignment
        self.pb_ZPZFocusScanStart.clicked.connect(self.zpFocusScan)
        self.pb_MoveZPZ1AbsPos.clicked.connect(self.zpMoveAbs)


        # sample position
        self.pb_save_pos.clicked.connect(self.generatePositionDict)
        self.pb_roiList_import.clicked.connect(self.importROIDict)
        self.pb_roiList_export.clicked.connect(self.exportROIDict)
        self.pb_roiList_clear.clicked.connect(self.clearROIList)
        #self.sampleROI_List.itemClicked.connect(self.showROIPos)
        #self.sampleROI_List.itemClicked.connect(lambda: self.statusbar.showMessage(
            #(str(self.roiDict[self.sampleROI_List.currentItem().text()]))))

        self.sampleROI_List.itemClicked.connect(self.showROIPosition)

        self.pb_move_pos.clicked.connect(self.gotoROIPosition)
        self.pb_recover_scan_pos.clicked.connect(self.gotoPosSID)
        self.pb_show_scan_pos.clicked.connect(self.viewScanPosSID)
        self.pb_print_scan_meta.clicked.connect(self.viewScanMetaData)
        self.pb_copy_curr_pos.clicked.connect(self.copyPosition)

        # Quick fill scan Params
        self.pb_3030.clicked.connect(self.fill_common_scan_params)
        self.pb_2020.clicked.connect(self.fill_common_scan_params)
        self.pb_66.clicked.connect(self.fill_common_scan_params)
        self.pb_22.clicked.connect(self.fill_common_scan_params)

        #copy scan plan
        self.pb_scan_copy.clicked.connect(self.copyScanPlan)
        self.pb_batchscan_copy.clicked.connect(self.copyForBatch)

        '''
        # elog
        self.pb_pdf_wd.clicked.connect(self.select_pdf_wd)
        self.pb_pdf_image.clicked.connect(self.select_pdf_image)
        self.pb_save_pdf.clicked.connect(self.force_save_pdf)
        self.pb_createpdf.clicked.connect(self.generate_pdf)
        self.pb_fig_to_pdf.clicked.connect(self.InsertFigToPDF)
        self.dateEdit_elog.setDate(QDate.currentDate())
        '''

        # close the application
        self.actionClose_Application.triggered.connect(self.close_application)

        self.create_live_pv_dict()
        self.create_pump_pv_dict()
        

        self.liveUpdateThread()
        self.scanStatusThread()
        self.pump_update_thread()
        #self.disable_while_scanning()
        self.show()

    def reload_gui(self):
        """Restarts gui"""
        print(sys.executable)
        print(sys.argv)
        os.execl(sys.executable,sys.executable, '/nsls2/conda/envs/2023-1.0-py310-tiled/bin/ipython',
                                                '--profile=collection',
                                                '--IPCompleter.use_jedi=False',
                                                sys.argv[0])
  
    def create_live_pv_dict(self):
        '''
        generate a dictionary of slots and signals , 
        later change to a json for flexibity ,; like mll specific?
        '''

        self.live_PVs = {

            self.lcd_ic3:"XF:03IDC-ES{Sclr:2}_cts1.D",
            self.lcd_monoE:"XF:03ID{}Energy-I",
            self.lcdPressure:"XF:03IDC-VA{VT:Chm-CM:1}P-I",
            self.lcd_scanNumber:"XF:03IDC-ES{Status}ScanID-I",
            self.db_smarx:"XF:03IDC-ES{SPod:1-Ax:2}Pos-I",
            self.db_smary:"XF:03IDC-ES{SPod:1-Ax:3}Pos-I",
            self.db_smarz:"XF:03IDC-ES{SPod:1-Ax:1}Pos-I",
            self.db_zpsth:"XF:03IDC-ES{SC210:1-Ax:1}Mtr.RBV",
            self.db_zpz1:"XF:03IDC-ES{MCS:1-Ax:zpz1}Mtr.RBV",
            self.db_dsx:"XF:03IDC-ES{SPod:1-Ax:2}Pos-I",
            self.db_dsy:"XF:03IDC-ES{SPod:1-Ax:3}Pos-I",
            self.db_dsz:"XF:03IDC-ES{SPod:1-Ax:1}Pos-I",
            self.db_dsth:"XF:03IDC-ES{SC210:1-Ax:1}Mtr.RBV",
            self.db_sbz:"XF:03IDC-ES{MCS:1-Ax:zpz1}Mtr.RBV",
            self.db_ssa2_x:"XF:03IDC-OP{Slt:SSA2-Ax:XAp}Mtr.RBV",
            self.db_ssa2_y:"XF:03IDC-OP{Slt:SSA2-Ax:YAp}Mtr.RBV",
            self.db_fs:"XF:03IDA-OP{FS:1-Ax:Y}Mtr.RBV",
            self.db_cam6:"XF:03IDC-OP{Stg:CAM6-Ax:X}Mtr.RBV",
            self.db_fs_det:"XF:03IDC-ES{Det:Vort-Ax:X}Mtr.RBV",
            self.db_diffx:"XF:03IDC-ES{Diff-Ax:X}Mtr.RBV",
            self.db_cam06x:"XF:03IDC-OP{Stg:CAM6-Ax:X}Mtr.RBV",
            self.db_s5_x:"XF:03IDC-ES{Slt:5-Ax:Vgap}Mtr.RBV",
            self.db_s5_y:"XF:03IDC-ES{Slt:5-Ax:Hgap}Mtr.RBV",
            self.db_dexela:"XF:03IDC-ES{Stg:FPDet-Ax:Y}Mtr.RBV"

            }
        
    def create_pump_pv_dict(self):
        '''
        1--> valves closed, 0--> open
        could not find PVs
        self.rb_pumpA_on:
        self.rb_pumpB_on:
        self.rb_he_backfill_fail:
        '''


        self.pump_PVs = {

            self.rb_fast_vent:"XF:03IDC-VA{ES:1-FastVtVlv:Stg3}Sts:Cls-Sts",
            self.rb_pumpA_slow:"XF:03IDC-VA{ES:1-SlowFrVlv:A}Sts:Cls-Sts",
            self.rb_pumpB_slow:"XF:03IDC-VA{ES:1-SlowFrVlv:B}Sts:Cls-Sts",

            }

    def handle_value_signals(self,pv_val_list):
        #print ("updating live values")
        livePVs = {key:value for key, value in zip(self.live_PVs.keys(),pv_val_list)}
        for item in livePVs.items():
                item[0].setValue(item[1])

    def handle_bool_signals(self,pv_val_list):
    
        #self.pump_PVs.keys() = pv_val_list (works?)
        livePVs = {key:value for key, value in zip(self.pump_PVs.keys(),pv_val_list)}
        for item in livePVs.items():
                item[0].setChecked(not bool(item[1]))
                

    def scanStatus(self,sts):

        if sts==1:
            self.label_scanStatus.setText("        Scan in Progress       ")
            self.label_scanStatus.setStyleSheet("background-color:rgb(0, 255, 0);color:rgb(255,0, 0)")
        else:
            self.label_scanStatus.setText("         Idle        ")
            self.label_scanStatus.setStyleSheet("background-color:rgb(255, 255, 0);color:rgb(0, 255, 0)")

    def scanStatusThread(self):

        self.scanStatus_thread = liveStatus("XF:03IDC-ES{Status}ScanRunning-I")
        self.scanStatus_thread.current_sts.connect(self.scanStatus)
        self.scanStatus_thread.start()

    def liveUpdateThread(self):
        print("Thread Started")

        self.liveWorker = liveUpdate(self.live_PVs)
        self.liveWorker.current_positions.connect(self.handle_value_signals)
        self.liveWorker.start()

    def pump_update_thread(self):
        print("Pump Update Thread Started")
        self.pump_update_worker = liveUpdate(self.pump_PVs,update_interval_ms = 2000)
        self.pump_update_worker.current_positions.connect(self.handle_bool_signals)
        self.pump_update_worker.start()

    def scanStatusMonitor(self):
        scanStatus = caget("XF:03IDC-ES{Status}ScanRunning-I")
        if scanStatus == 1:
            self.label_scanStatus.setText("Scan in Progress")
            self.label_scanStatus.setStyleSheet("background-color:rgb(0, 255, 0);color:rgb(255,0, 0)")

        else:
            self.label_scanStatus.setText("Idle")
            self.label_scanStatus.setStyleSheet("background-color:rgb(255, 255, 0);color:rgb(0, 255, 0)")

    #setup user
    @show_error_message_box
    def new_user_setup(self):
        setup_new_user(name = self.le_username.text(),
                       experimenters = self.le_experimenters.text(), 
                       sample = self.le_sample_name.text(), 
                       sample_image = self.le_image_path.text()
                       )
    
    @show_error_message_box
    def import_xrf_elem_list(self, auto = False):
         
        if auto:
             json_param_file = "/nsls2/data/hxn/shared/config/bluesky/profile_collection/startup/plot_elems.json"

        else:
            # Open a file dialog to select a JSON file
            file_name, _ = QFileDialog.getOpenFileName(self, 'Open JSON File', '', 'JSON Files (*.json)')

            if file_name:
                json_param_file = file_name
            else:
                return

        with open(json_param_file, "r") as fp:
            xrf_elems = json.load(fp)
            fp.close()

        roi_elems = xrf_elems["roi_elems"]
        live_plot_elems  = xrf_elems["live_plot_elems"]
        line_plot_elem = xrf_elems["line_plot_elem"]

        self.le_roi_elems.setText(str(roi_elems)[1:-1])
        self.le_live_elems.setText(str(live_plot_elems)[1:-1])
        self.le_line_elem.setText(str(line_plot_elem)[1:-1])

        for elem,box in zip(roi_elems,self.xrf_combo_boxes):
            try:box.setCurrentText(elem)
            except:box.setCurrentText("Si")

    @show_error_message_box
    def export_xrf_elem_list(self, auto = False):
        
        xrf_elems = {}
        
        if auto:
             json_param_file = "/nsls2/data/hxn/shared/config/bluesky/profile_collection/startup/plot_elems.json"

        else:
            # Open a file dialog to select a JSON file
            file_name, _ = QFileDialog.getSaveFileName(self, 'Open JSON File', '', 'JSON Files (*.json)')

            if file_name:
                json_param_file = file_name

            else:
                return

        xrf_elems["roi_elems"] = [re.sub("[^A-Z,_]", '', i,0,re.IGNORECASE) for i in self.le_roi_elems.text().split(",")]
        xrf_elems["live_plot_elems"] = [re.sub("[^A-Z,_]", '', i,0,re.IGNORECASE) for i in self.le_live_elems.text().split(",")]
        xrf_elems["line_plot_elem"] = [re.sub("[^A-Z,_]", '', i,0,re.IGNORECASE) for i in self.le_line_elem.text().split(",")]
        
        with open(json_param_file, "w") as fp:
            json.dump(xrf_elems, fp, indent = 6)
            fp.close()

        

    def populate_elems_from_combobox(self):

        selected_elems = []
        for cb in self.xrf_combo_boxes:
            selected_elems.append(cb.currentText())

        live_plot_elems = selected_elems[0:self.sb_live_elem_num.value()]
        line_plot_elem = selected_elems[0]


        self.le_roi_elems.setText(str(selected_elems)[1:-1])
        self.le_live_elems.setText(str(live_plot_elems)[1:-1])
        self.le_line_elem.setText(str(line_plot_elem))

    def apply_xrf_elems(self):

        choice = QMessageBox.question(self, 'Info',
            f"Update the list of elements? The program require reload to update the changes. Proceed?", 
            QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)
        if choice == QMessageBox.Yes:
            self.export_xrf_elem_list(auto = True)
            self.reload_gui()
        else:
            pass      
        
    #flyscan
    def getScanValues(self):
        self.det = self.pb_dets.currentText()

        self.mot1_s = self.x_start.value()
        self.mot1_e = self.x_end.value()
        self.mot1_steps = self.x_step.value()

        self.mot2_s = self.y_start.value()
        self.mot2_e = self.y_end.value()
        self.mot2_steps = self.y_step.value()

        self.dwell_t = self.dwell.value()

        self.motor1 = self.cb_motor1.currentText()
        self.motor2 = self.cb_motor2.currentText()


        self.det_list = {'dets1': dets1, 'dets2': dets2, 'dets3': dets3,
                         'dets4': dets4, 'dets_fs': dets_fs}

    def initParams(self):
        self.getScanValues()

        cal_res_x = abs(self.mot1_e - self.mot1_s) / self.mot1_steps
        cal_res_y = abs(self.mot2_e - self.mot2_s) / self.mot2_steps
        self.tot_t_2d = self.mot1_steps * self.mot2_steps * self.dwell_t / 60
        self.tot_t_1d = self.mot1_steps * self.dwell_t / 60

        if self.rb_1d.isChecked():
            self.label_scan_info_calc.setText(f'X: {(cal_res_x * 1000):.2f} nm, Y: {(cal_res_y * 1000):.2f} nm \n'
                                              f'{self.tot_t_1d:.2f} minutes + overhead')
            self.scan_plan = f'fly1d({self.det},{self.motor1}, {self.mot1_s},{self.mot1_e}, ' \
                        f'{self.mot1_steps}, {self.dwell_t:.3f})'



        else:
            self.label_scan_info_calc.setText(f'X: {(cal_res_x * 1000):.2f} nm, Y: {(cal_res_y * 1000):.2f} nm \n'
                                              f'{self.tot_t_2d:.2f} minutes + overhead')
            self.scan_plan = f'fly2d({self.det}, {self.motor1},{self.mot1_s}, {self.mot1_e}, {self.mot1_steps},' \
                        f'{self.motor2},{self.mot2_s},{self.mot2_e},{self.mot2_steps},{self.dwell_t:.3f})'

        self.text_scan_plan.setText(self.scan_plan)

    def copyForBatch(self):
        self.text_scan_plan.setText('yield from '+self.scan_plan)
        self.text_scan_plan.selectAll()
        self.text_scan_plan.copy()

    def copyScanPlan(self):
        self.text_scan_plan.setText('<'+self.scan_plan)
        self.text_scan_plan.selectAll()
        self.text_scan_plan.copy()
 
    @show_error_message_box
    def initFlyScan(self):
        self.getScanValues()

        if self.rb_1d.isChecked():
            
            if self.tot_t_1d>1.0:
                choice = QMessageBox.question(self, 'Warning',
                                f"Total scan time is {self.tot_t_1d :.2f}. Continue?", QMessageBox.Yes |
                                QMessageBox.No, QMessageBox.No)

                if choice == QMessageBox.Yes:

                    RE(fly1d(self.det_list[self.det], 
                             self.motor_list[self.motor1],
                             self.mot1_s, 
                             self.mot1_e, 
                             self.mot1_steps, 
                             self.dwell_t))

                else:
                    return

            else:

                RE(fly1d(self.det_list[self.det],
                         self.motor_list[self.motor1],
                         self.mot1_s, 
                         self.mot1_e, 
                         self.mot1_steps, 
                         self.dwell_t))

                    

        else:
            if self.motor_list[self.motor1] == self.motor_list[self.motor2]:
                msg = QErrorMessage(self)
                msg.setWindowTitle("Flyscan Motors are the same")
                msg.showMessage(f"Choose two different motors for 2D scan. You selected {self.motor_list[self.motor1].name}")
                return
            elif self.tot_t_2d>5.0:
                    choice = QMessageBox.question(self, 'Warning',
                                f"Total scan time is more than {self.tot_t_2d :.2f}. Continue?", 
                                QMessageBox.Yes |
                                QMessageBox.No, QMessageBox.No)

                    if choice == QMessageBox.Yes:


                        RE(fly2d(self.det_list[self.det], 
                                 self.motor_list[self.motor1], 
                                 self.mot1_s, self.mot1_e, 
                                 self.mot1_steps,
                                 self.motor_list[self.motor2], 
                                 self.mot2_s, 
                                 self.mot2_e, 
                                 self.mot2_steps, 
                                 self.dwell_t))

                    else:
                        return

            else:

                RE(fly2d(self.det_list[self.det], 
                         self.motor_list[self.motor1], 
                         self.mot1_s, self.mot1_e, 
                         self.mot1_steps,
                         self.motor_list[self.motor2], 
                         self.mot2_s, 
                         self.mot2_e, 
                         self.mot2_steps, 
                         self.dwell_t))


    def disableMot2(self):
        self.y_start.setEnabled(False)
        self.y_end.setEnabled(False)
        self.y_step.setEnabled(False)

    def enableMot2(self):
        self.y_start.setEnabled(True)
        self.y_end.setEnabled(True)
        self.y_step.setEnabled(True)


    def disable_while_scanning(self):

        while True:
            if caget("XF:03IDC-ES{Status}ScanRunning-I") == 1:
                self.HXN_GUI_tabs.setEnabled(False)
                QtTest.QTest.qWait(500)

            else:
                self.HXN_GUI_tabs.setEnabled(True)
                QtTest.QTest.qWait(500)

    def fill_common_scan_params(self):
        button_name = self.sender()
        button_names = {'pb_2020': (20, 20, 100, 100, 0.03),
                        'pb_3030': (30, 30, 30, 30, 0.03),
                        'pb_66': (6, 6, 100, 100, 0.05),
                        'pb_22': (2, 2, 100, 100, 0.03)
                        }
        if button_name.objectName() in button_names.keys():
            valsToFill = button_names[button_name.objectName()]
            self.x_start.setValue(valsToFill[0] / -2)
            self.x_end.setValue(valsToFill[0] / 2)
            self.y_start.setValue(valsToFill[1] / -2)
            self.y_end.setValue(valsToFill[1] / 2)
            self.x_step.setValue(valsToFill[2])
            self.y_step.setValue(valsToFill[3])
            self.dwell.setValue(valsToFill[4])

    def requestScanPause(self):
        RE.request_pause(True)
        self.pb_REAbort.setEnabled(True)
        self.pb_REResume.setEnabled(True)

    def scanAbort(self):
        RE.abort()
        self.pb_REAbort.setEnabled(False)
        self.pb_REResume.setEnabled(False)

    def scanResume():
        RE.resume()
        self.pb_REAbort.setEnabled(False)
        self.pb_REResume.setEnabled(False)
    
    #TODO change to PV based
    @show_error_message_box
    def moveAMotor(self, val_box, mot_name, unit_conv_factor= 1, neg=False):

        if neg:
            move_by = val_box.value() * -1
        else:
            move_by = val_box.value()

        RE(bps.movr(mot_name, move_by * unit_conv_factor))
        self.statusbar.showMessage(f'{mot_name.name} moved by {move_by} um ')

    @show_error_message_box
    def move_motor_with_pv(self, val_box, mot_name, unit_conv_factor= 1, neg=False):

        if mot_name.prefix != "":
            pv_name = mot_name.prefix
            position = mot_name.position

        else:
            pv_name = mot_name.report["pv"]
            position = mot_name.report["position"]

        if neg:
            move_by = val_box.value() * -1
        else:
            move_by = val_box.value()

        caput(pv_name, position+move_by * unit_conv_factor)
        self.statusbar.showMessage(f'{mot_name.name} moved by {move_by} um ')

    def move_smarx(self, neg_=False):
         self.moveAMotor(self.db_move_smarx, smarx, 0.001, neg=neg_)

    def move_smary(self, neg_=False):
         self.moveAMotor(self.db_move_smary, smary, 0.001, neg=neg_)

    def move_smarz(self, neg_=False):
         self.moveAMotor(self.db_move_smarz, smarz, 0.001, neg=neg_)

    def move_zpth(self, neg_=False):
        self.moveAMotor(db_move_zpsth, zpsth, neg=neg_)

    def move_zpz1(self, neg_=False):
        if neg_:

            RE(movr_zpz1(self.db_move_zpz.value() * 0.001 * -1))

        else:
            RE(movr_zpz1(self.db_move_zpz.value() * 0.001))

    #mll

    def move_dsx(self, neg_=False):
        self.move_motor_with_pv(self.db_move_dsx, dsx, 1, neg=neg_)

    def move_dsy(self, neg_=False):
        self.move_motor_with_pv(self.db_move_dsy, dsy, 1, neg=neg_)

    def move_dsz(self, neg_=False):
        self.move_motor_with_pv(self.db_move_dsz, dsz, 1, neg=neg_)

    def move_dsth(self, neg_=False):
        self.moveAMotor(self.db_move_dsth, dsth, neg=neg_)


    #Energy change ui

    def update_energy_calc(self, target_energy):

        print("calculating energy values")



        ugap,hrm,dcm_p,dcm_r,hfm_p,mirror_coating = Energy.get_values(target_energy)
        '''
        target_pitch = Energy.calculatePitch(target_energy)
        target_mirror_coating = Energy.findAMirror(target_energy)

        if self.sb_harmonic.value() == -1:
            target_ugap = Energy.gap(target_energy)
        else:
            target_ugap = Energy.gap(target_energy, self.sb_harmonic.value())

        print(f"{target_pitch = :.4f},{target_ugap[0] = :.1f},{target_mirror_coating} ")
        '''
        self.dsb_target_ugap.setValue(ugap)
        self.dsb_target_pitch.setValue(dcm_p)
        self.cb_target_coating.setCurrentText(mirror_coating)
        self.sb_calc_harmonic.setValue(hrm)
        self.dsb_target_roll.setValue(dcm_r)
        self.dsb_hfm_target_pitch.setValue(hfm_p)

        return ugap,hrm,dcm_p,dcm_r,hfm_p,mirror_coating


    @show_error_message_box
    def change_energy(self,target_energy):
    
        target_energy = self.dsb_target_e.value()
        if not caget('XF:03IDA-OP{FS:1-Ax:Y}Mtr.VAL')<-50: 

            fs_choice = QMessageBox.question(self, 'Info',
            f"Do you want to insert front end fluorescence camera?", 
            QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)
            if fs_choice == QMessageBox.Yes:

                caput('XF:03IDA-OP{FS:1-Ax:Y}Mtr.VAL', -57.)
                caput("XF:03IDA-BI{FS:1-CAM:1}cam1:Acquire",1)

            else:
                pass      

        choice = QMessageBox.question(self, 'Info',
        f"{target_energy = :.2f}, Continue?", 
        QMessageBox.Yes |
        QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:

            RE(Energy.move(target_energy, self.sb_harmonic.value()))

            if self.cb_beam_at_ssa2.isChecked():
                RE(find_beam_at_ssa2(max_iter = self.sb_max_iter.value()))

        else:
            return 

    @show_error_message_box
    def ZP_OSA_OUT(self):
        curr_pos = caget("XF:03IDC-ES{ANC350:5-Ax:1}Mtr.VAL")
        if curr_pos >2000:
            self.statusbar.showMessage('OSAY is out of IN range')
        else:
            caput("XF:03IDC-ES{ANC350:5-Ax:1}Mtr.VAL",curr_pos+2700)

        self.statusbar.showMessage('OSA Y moved OUT')
    
    @show_error_message_box
    def ZP_OSA_IN(self):
        curr_pos = caget("XF:03IDC-ES{ANC350:5-Ax:1}Mtr.VAL")

        if curr_pos > 2500:
            caput("XF:03IDC-ES{ANC350:5-Ax:1}Mtr.VAL",curr_pos-2700)
            self.statusbar.showMessage('OSA Y is IN')
        else:
            self.statusbar.showMessage('OSA Y is close to IN position')
            pass

    @show_error_message_box
    def merlinIN(self):
        self.client.open('http://10.66.17.43')
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            RE(go_det('merlin'))
        else:
            pass
    
    @show_error_message_box
    def merlinOUT(self):
        self.client.open('http://10.66.17.43')
        QtTest.QTest.qWait(2000)
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            RE(go_det("out"))
        else:
            pass
    
    @show_error_message_box
    def dexela_motion(self, move_to = 0):
        
        self.client.open('http://10.66.17.48')
        self.client.open('http://10.66.17.43')
        QtTest.QTest.qWait(2000)
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            caput("XF:03IDC-ES{Stg:FPDet-Ax:Y}Mtr.VAL",move_to)
        else:
            pass
    
    @show_error_message_box
    def vortexIN(self):
        #RE(bps.mov(fdet1.x, -7))
        caput("XF:03IDC-ES{Det:Vort-Ax:X}Mtr.VAL", self.dsb_flur_in_pos.value())
        self.statusbar.showMessage('FS det Moving')
    
    @show_error_message_box
    def vortexOUT(self):
        #RE(bps.mov(fdet1.x, -107))
        caput("XF:03IDC-ES{Det:Vort-Ax:X}Mtr.VAL", -107)
        self.statusbar.showMessage('FS det Moving')
    
    @show_error_message_box
    def cam11IN(self):
        self.client.open('http://10.66.17.43')
        QtTest.QTest.qWait(2000)
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            RE(go_det('cam11'))
            self.statusbar.showMessage('CAM11 is IN')
        else:
            pass

    @show_error_message_box
    def cam6IN(self):
        caput("XF:03IDC-ES{CAM:06}cam1:Acquire",1)
        caput('XF:03IDC-OP{Stg:CAM6-Ax:X}Mtr.VAL', 0)
        QtTest.QTest.qWait(1000)
        self.statusbar.showMessage('CAM6 Moving!')

    @show_error_message_box
    def cam6OUT(self):
        caput("XF:03IDC-ES{CAM:06}cam1:Acquire",0)
        caput('XF:03IDC-OP{Stg:CAM6-Ax:X}Mtr.VAL', -50)
        QtTest.QTest.qWait(1000)
        self.statusbar.showMessage('CAM6 Moving!')
    
    @show_error_message_box
    def FS_IN(self):
        caput('XF:03IDA-OP{FS:1-Ax:Y}Mtr.VAL', -57.)
        caput("XF:03IDA-BI{FS:1-CAM:1}cam1:Acquire",1)
        QtTest.QTest.qWait(20000)
        #self.statusbar.showMessage('FS Motion Done!')
    
    @show_error_message_box
    def FS_OUT(self):
        caput("XF:03IDA-BI{FS:1-CAM:1}cam1:Acquire",0)
        caput('XF:03IDA-OP{FS:1-Ax:Y}Mtr.VAL', -20.)
        
        QtTest.QTest.qWait(20000)
        #self.statusbar.showMessage('FS Motion Done!')
    
    @show_error_message_box
    def SSA2_Pos(self, x, y):
        caput('XF:03IDC-OP{Slt:SSA2-Ax:XAp}Mtr.VAL', x)
        caput('XF:03IDC-OP{Slt:SSA2-Ax:YAp}Mtr.VAL', y)
        QtTest.QTest.qWait(15000)
    
    @show_error_message_box
    def S5_Pos(self, x, y):
        caput('XF:03IDC-ES{Slt:5-Ax:Vgap}Mtr.VAL', x) #PV names seems flipped
        caput('XF:03IDC-ES{Slt:5-Ax:Hgap}Mtr.VAL', y)
        QtTest.QTest.qWait(15000)
    
    @show_error_message_box
    def plot_me(self):
        sd = self.pb_plot_sd.text()
        elem = self.pb_plot_elem.text()

        if ',' in sd:
            slist_s, slist_e = sd.split(",")

            f_sd = int(slist_s.strip())
            l_sd = int(slist_e.strip())
            space = abs(int(slist_e.strip())-int(slist_s.strip()))+1

            s_list = np.linspace(f_sd, l_sd, space)
            for sd_ in s_list:
                plot_data(int(sd_), elem, 'sclr1_ch4')

        else:
            plot_data(int(sd), elem, 'sclr1_ch4')
    
    @show_error_message_box
    def plot_erf_fit(self):
        sd = self.pb_plot_sd.text()
        elem = self.pb_plot_elem.text()
        erf_fit(int(sd), elem, linear_flag=self.cb_erf_linear_flag.isChecked())
    
    @show_error_message_box
    def plot_line_center(self):
        sd = self.pb_plot_sd.text()
        elem = self.pb_plot_elem.text()
        return_line_center(int(sd), elem, threshold=self.dsb_line_center_thre.value())

    def close_all_plots(self):
        plt.close('all')
    
    @show_error_message_box
    def zpFocusScan(self):
        zpStart = self.sb_ZPZ1RelativeStart.value()*0.001
        zpEnd = self.sb_ZPZ1RelativeEnd.value()*0.001
        zpSteps = self.sb_ZPZ1Steps.value()

        scanMotor = self.motor_list[self.cb_foucsScanMotor.currentText()]
        scanStart = self.sb_FocusScanMtrStart.value()
        scanEnd = self.sb_FocusScanMtrEnd.value()
        scanSteps = self.sb_FocusScanMtrStep.value()
        scanDwell = self.dsb_FocusScanDwell.value()

        fitElem = self.le_FocusingElem.text()
        linFlag = self.cb_linearFlag_zpFocus.isChecked()

        RE(zp_z_alignment(zpStart,zpEnd,zpSteps,scanMotor,scanStart,scanEnd,scanSteps,scanDwell,
                          elem= fitElem, linFlag = linFlag))

    def zpMoveAbs(self):
        zpTarget = self.dsb_ZPZ1TargetPos.value()
        choice = QMessageBox.question(self, "Zone Plate Z Motion",
                                      f"You're making an Absolute motion of ZP to {zpTarget}. \n Proceed?",
                                      QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        QtTest.QTest.qWait(500)
        if choice == QMessageBox.Yes:
            RE(mov_zpz1(zpTarget))

        else:
            pass

    def zpRotAlignment(self):
        pass

    def abort_scan(self):
        #signal.signal(signal.SIGINT, signal.SIG_DFL)
        RE.abort()

    #Sample Chamber
    def qMessageExcecute(self,funct):
        QtTest.QTest.qWait(500)
        choice = QMessageBox.question(self, 'Sample Chamber Operation Warning',
                                      "Make sure this action is safe. \n Proceed?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        QtTest.QTest.qWait(500)
        if choice == QMessageBox.Yes:
            RE(funct)

        else:
            pass

    # PDF Log

    def select_pdf_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_folder_log.setText(str(folder_path))

    def select_pdf_image(self):
        file_name = QFileDialog().getOpenFileName(self, "Select an Image")
        self.le_elog_image.setText(str(file_name[0]))

    def generate_pdf(self):
        dt = self.dateEdit_elog.date()
        tmp_date = dt.toString(self.dateEdit_elog.displayFormat())
        tmp_file = os.path.join(self.le_folder_log.text(), self.le_elog_name.text())
        tmp_sample = self.le_elog_sample.text()
        tmp_experimenter = self.le_elog_experimenters.text()
        tmp_pic = self.le_elog_image.text()

        setup_pdf_for_gui(tmp_file, tmp_date, tmp_sample, tmp_experimenter, tmp_pic)
        insertTitle_for_gui()
        self.statusbar.showMessage(f'pdf generated as {tmp_file}')

    def force_save_pdf(self):
        save_page_for_gui()

    def InsertFigToPDF(self):
        insertFig_for_gui(note=self.le_pdf_fig_note.text(),
                          title=self.le_pdf_fig_title.text())
        self.statusbar.showMessage("Figure added to the pdf")

    # Sample Stage Navigation
    def recordPositions(self):

        fx, fy, fz = zpssx.position, zpssy.position, zpssz.position
        cx, cy, cz = smarx.position, smary.position, smarz.position
        zpz1_pos = zp.zpz1.position
        zp_sx, zp_sz = zps.zpsx.position, zps.zpsz.position
        th = zpsth.position
        self.roi = {
            zpssx: fx, zpssy: fy, zpssz: fz,
            smarx: cx, smary: cy, smarz: cz,
            zp.zpz1: zpz1_pos, zpsth: th,
            zps.zpsx: zp_sx, zps.zpsz: zp_sz
        }

    def generatePositionDict(self):

        self.recordPositions()
        roi_name = 'ROI' + str(self.sampleROI_List.count())
        self.roiDict[roi_name] = self.roi
        self.sampleROI_List.addItem(roi_name)

        #make the item editable
        item = self.sampleROI_List.item(self.sampleROI_List.count()-1)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

    def applyDictWithLabel(self):
        label_ = {}
        for idx in range(self.sampleROI_List.count()):
            label = self.sampleROI_List.item(idx).text()
            label_[label] = idx
        self.roiDict['user_labels'] = label_

    def exportROIDict(self):
        self.applyDictWithLabel()
        file_name = QFileDialog().getSaveFileName(self, "Save Parameter File",
                                                  'hxn_zp_roi_list.json',
                                                  'json file(*json)')
        if file_name:

            with open(file_name[0], 'w') as fp:
                json.dump(self.roiDict, fp, indent=4)
        else:
            pass

    def importROIDict(self):

        file_name = QFileDialog().getOpenFileName(self, "Open Parameter File",
                                                  ' ', 'json file(*json)')
        if file_name:
            self.roiDict = {}
            with open(file_name[0], 'r') as fp:
                self.roiDict = json.load(fp)

            print(self.roiDict['user_labels'])

            self.sampleROI_List.clear()
            for num,items in enumerate(self.roiDict['user_labels']):
                self.sampleROI_List.addItem(items)
                item = self.sampleROI_List.item(num)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        else:
            pass

    def clearROIList(self):
        self.sampleROI_List.clear()

    def showROIPos(self,item):
        item_num = self.sampleROI_List.row(item)
        for key, value in self.roiDict[f'ROI{item_num}'].items():
            self.statusbar.showMessage(f'{key.name}:{value:.4f}')

    def gotoROIPosition(self):
        roi_num = self.sampleROI_List.currentRow()
        param_file = self.roiDict[f'ROI{roi_num}']
        for key, value in param_file.items():
            if not key == zp.zpz1:
                RE(bps.mov(key, value))
            elif key == zp.zpz1:
                RE(mov_zpz1(value))
            self.statusbar.showMessage(f'Sample moved to {key.name}:{value:.4f} ')

    def showROIPosition(self, item):
        item_num = self.sampleROI_List.row(item)
        param_file = self.roiDict[f'ROI{item_num}']
        self.statusbar.showMessage(('*' * 20))
        for key, value in param_file.items():
            self.statusbar.showMessage(f'{key.name}:{value:.4f}')

        #self.sampleROI_List.itemClicked.connect(lambda: self.statusbar.showMessage(
        # (self.roiDict[self.sampleROI_List.currentItem().text()])))

    def gotoPosSID(self):
        sd = self.le_sid_position.text()
        zp_flag = self.cb_sid_moveZPZ.isChecked()

        sdZPZ1 = (db[int(sd)].table("baseline")["zpz1"].values)[0]
        currentZPZ1 = zp.zpz1.position
        #current_energy = caget("XF:03ID{}Energy-I")
        zDiff = abs(sdZPZ1-currentZPZ1)

        if zDiff>1 and zp_flag:
                choice = QMessageBox.question(self, 'Warning',
                "You are recovering positions from a scan done at a different Focus."
                "The ZPZ1 motion could cause collision. Are you sure??",
                QMessageBox.Yes |QMessageBox.No, QMessageBox.No)

                if choice == QMessageBox.Yes:
                    RE(recover_zp_scan_pos(int(sd), zp_flag, 1))
                else:
                    return
        else:

            RE(recover_zp_scan_pos(int(sd), zp_flag, 1))

        self.statusbar.showMessage(f'Positions recovered from {sd}')

    def viewScanPosSID(self):
        sd = self.le_sid_position.text()
        data = db.get_table(db[int(sd)],stream_name='baseline')

        zpz1 = data.zpz1[1]
        zpx = data.zpx[1]
        zpy = data.zpy[1]
        smarx = data.smarx[1]
        smary = data.smary[1]
        smarz = data.smarz[1]
        zpsz = data.zpsz[1]

        info1 = f"scan id: {db[int(sd)].start['scan_id']}, zpz1:{zpz1 :.3f}, zpsz:{zpsz :.3f} \n"
        info2 = f"smarx: {smarx :.3f}, smary: {smary :.3f}, smarz: {smarz :.3f} \n"
        info3 = f"zpssx: {data.zpssx[1] :.3f}, zpssy: {data.zpssy[1] :.3f}, zpssz: {data.zpssz[1] :.3f} \n "


        self.statusbar.showMessage(str(info1+info2+info3))

    def viewScanMetaData(self):
        sd = self.le_sid_position.text()
        h = db[int(sd)]
        self.statusbar.showMessage(str(h.start))

    def copyPosition(self):

        self.recordPositions()

        stringToCopy = ' '
        for item in self.roi.items():
            stringToCopy += (f'{item[0].name}:{item[1]:.4f} \n')

        #stringToCopy = str([item[0].name, item[1] for item in self.roi.items()])

        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(stringToCopy, mode=cb.Clipboard)

    #exit gui

    def closeEvent(self,event):
        reply = QMessageBox.question(self, 'Quit GUI', "Are you sure you want to close the window?")
        if reply == QMessageBox.Yes:
            event.accept()
            self.scanStatus_thread.quit()
            self.liveWorker.quit()
            plt.close('all')
            #self.updateTimer.stop()
            QApplication.closeAllWindows()

        else:
            event.ignore()

    def close_application(self):

        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            plt.close('all')
            #stop the timers
            #self.updateTimer.stop()
            QApplication.closeAllWindows()

            print('quit application')
            sys.exit()
        else:
            pass


    #in use
    def start_pumping_in_thread(self, turbo_start=350):

        user_target_pressure = np.round(self.dsb_target_p.value(),1)


        choice = QMessageBox.question(self, 'Start Pumping',
                                      f"HXN pumping will be excecuted with target value of {user_target_pressure}"
                                      f"Turbo starts at ~ {turbo_start:.1f}"
                                      f" Please confirm.", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:

            # reset the progress bar 
            self.prb_sample_exchange.reset()

            #set min/max for the progress bar- by default the max-min>0 and pressure here is decreasing
            #tracking relative change
            self.prb_sample_exchange.setRange(0, 100)

            self.lb_sample_change_action.setText("Pumping in progress")

            self.pump_worker_thread = PumpingThread("XF:03IDC-VA{VT:Chm-CM:1}P-I", 
                                             turbo_start, 
                                             user_target_pressure
                                            )
            
            #thread connections
            self.pump_worker_thread.start_slow_pump.connect(HXNSampleExchanger.start_slow_pumps)
            self.pump_worker_thread.start_fast_pump.connect(HXNSampleExchanger.start_fast_pumps)
            self.pump_worker_thread.pressure_change.connect(self.prb_sample_exchange.setValue)
            self.pump_worker_thread.finished.connect(HXNSampleExchanger.stop_pumping)
            self.pump_worker_thread.finished.connect(lambda:self.prb_sample_exchange.setRange(0,100))
            self.pump_worker_thread.finished.connect(lambda:self.prb_sample_exchange.setValue(100))
            self.pump_worker_thread.finished.connect(self.pump_worker_thread.quit)
            self.pump_worker_thread.finished.connect(lambda: self.lb_sample_change_action.setText("Pumping finished"))
            self.pump_worker_thread.finished.connect(lambda:self.he_backfill_in_thread(target_pressure = 250))

            self.pump_worker_thread.start()

        else:
            return

   
    def vent_in_thread(self, target_pressure = 745):

        choice = QMessageBox.question(self, 'Start Venting',
                                      "Microscope chamber will be vented. Continue?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:

            # reset the progress bar 
            self.prb_sample_exchange.reset()

            self.lb_sample_change_action.setText("Venting in progress")

            #set min/max for the progress bar
            self.prb_sample_exchange.setRange(0,  int(target_pressure))

            self.vent_thread = VentingThread("XF:03IDC-VA{VT:Chm-CM:1}P-I", target_pressure)
            
            self.vent_thread.open_vent.connect(HXNSampleExchanger.start_venting)
            
            if self.cb_sample_ex_det_out.isChecked():
                self.vent_thread.started.connect(lambda:RE(diff_to_home(move_out_later = True)))
            
            self.vent_thread.pressure_change.connect(self.prb_sample_exchange.setValue)
            self.vent_thread.finished.connect(lambda:self.prb_sample_exchange.setRange(0,100))
            self.vent_thread.finished.connect(lambda:self.prb_sample_exchange.setValue(100))
            self.vent_thread.finished.connect(self.vent_thread.quit)
            self.vent_thread.finished.connect(lambda: self.lb_sample_change_action.setText("Venting finished"))
            self.vent_thread.start()
    
    def he_backfill_in_thread(self,target_pressure = 250):

        QtTest.QTest.qWait(10000)

        #open he valve
        caput('XF:03IDC-ES{IO:1}DO:4-Cmd',1)

        # reset the progress bar 
        self.prb_sample_exchange.reset()

        #set min/max for the progress bar
        self.prb_sample_exchange.setRange(0,  249)
        
        self.lb_sample_change_action.setText("He-backfill in progress")

        self.he_backfill_worker_thread = HeBackFillThread("XF:03IDC-VA{VT:Chm-CM:1}P-I",
                                                            target_pressure)

        #connections
        self.he_backfill_worker_thread.start_he_backfill.connect(HXNSampleExchanger.auto_he_backfill)
        self.he_backfill_worker_thread.pressure_change.connect(self.prb_sample_exchange.setValue)
        self.he_backfill_worker_thread.finished.connect(lambda:self.prb_sample_exchange.setRange(0,100))
        self.he_backfill_worker_thread.finished.connect(lambda:self.prb_sample_exchange.setValue(100))
        self.he_backfill_worker_thread.finished.connect(lambda: caput('XF:03IDC-ES{IO:1}DO:4-Cmd',0))
        self.he_backfill_worker_thread.finished.connect(lambda: self.lb_sample_change_action.setText("He-Backfill finished"))
        self.he_backfill_worker_thread.finished.connect(lambda: QMessageBox.about(self, "He Backfill Status", "Chamber is Ready"))
        self.he_backfill_worker_thread.finished.connect(self.he_backfill_worker_thread.quit)

        if self.cb_sample_ex_cam11_in.isChecked():

            if diff_status()=="diff_pos":
                logger.Warning("Detector stage probably in a diffraction position;Motion Aborted")
                QMessageBox.about(self, "Detector stage probably in a diffraction position", 
                                  "Motion Aborted")

            else:
                self.he_backfill_worker_thread.started.connect(lambda:RE(go_det("cam11")))

            #TODO need to verify if the detector stage is in  some diff position, check sum(which diff) not round(0)



        self.he_backfill_worker_thread.start()

    def pumping_stop(self):

        choice = QMessageBox.question(self, 'Stop Pumping',
                                      "Pumping will be stopped. Continue?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
        
            HXNSampleExchanger.stop_pumping()
            QtTest.QTest.qWait(3000)
            self.pump_worker_thread.quit()
        else:
            return




#from FXI--modified
class liveStatus(QThread):
    current_sts = pyqtSignal(int)
    def __init__(self, PV):
        super().__init__()
        self.PV = PV

    def run(self):

        while True:
            self.current_sts.emit(caget(self.PV))
            #print("New positions")
            QtTest.QTest.qWait(500)

class liveUpdate(QThread):
    
    current_positions = pyqtSignal(list)

    def __init__(self, pv_dict, update_interval_ms = 500):
        super().__init__()
        self.pv_dict = pv_dict
        self.update_interval_ms = update_interval_ms

    def return_values(self):
        readings = []
        for i in self.pv_dict.values():
            
            readings.append(caget(i))

        return readings

    #use moveToThread method later
    def run(self):

        while True:
            positions = self.return_values()
            #self.pv_list = list(self.pv_dict.values())
            self.current_positions.emit(positions)
            #print(list(self.pv_dict.values())[0])
            #print(positions[0])
            QtTest.QTest.qWait(self.update_interval_ms)

#in use
class PumpingThread(QThread):

    pressure_change = pyqtSignal(int)

    start_slow_pump = pyqtSignal()
    start_fast_pump = pyqtSignal()
    finished = pyqtSignal()


    def __init__(self, pressure_pv, turbo_start_p, target_p):
        
        super().__init__()
        self.pressure_pv = pressure_pv
        self.turbo_start_p = turbo_start_p
        self.target_p = target_p


    def run(self):

        self.start_slow_pump.emit()

        while caget(self.pressure_pv)>self.turbo_start_p:
            #keep emitting the pressure value for progress bar
            self.pressure_change.emit(int((760-caget(self.pressure_pv))/7.6))
            QtTest.QTest.qWait(5000)

        self.start_fast_pump.emit()
        print("fast_triggered")
        
        while caget(self.pressure_pv)>self.target_p:
            
            #if pressure is below threshold open fast
            self.pressure_change.emit(int((760-caget(self.pressure_pv))/7.6))
            QtTest.QTest.qWait(5000)

        self.finished.emit()


class VentingThread(QThread):
    
    pressure_change = pyqtSignal()
    open_vent = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, pressure_pv, target_p):

        #make edits to trigger fast vent open from here
        
        super().__init__()
        self.pressure_pv = pressure_pv
        self.target_p = target_p
    
    def run(self):

        self.open_vent.emit()
        
        while caget(self.pressure_pv)<self.target_p:
            self.pressure_change.emit(int(caget(self.pressure_pv)))
            QtTest.QTest.qWait(5000)

        self.finished.emit()


class HeBackFillThread(QThread):

    pressure_change = pyqtSignal(int)
    start_he_backfill = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, pressure_pv, target_p):
    
        super().__init__()
        self.pressure_pv = pressure_pv
        self.target_p = target_p

    def run(self):

        self.start_he_backfill.emit()

        while caget(self.pressure_pv)<248.0:
            QtTest.QTest.qWait(1000)
            self.pressure_change.emit(int(caget(self.pressure_pv)))

        self.finished.emit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
    app.deleteLater()


