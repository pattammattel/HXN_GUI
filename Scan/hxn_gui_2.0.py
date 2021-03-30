# -*- coding: utf-8 -*-

# Author: Ajith Pattammattel
# Original Date:06-23-2020

import os
import signal
import subprocess
import sys
import collections
import webbrowser
import pyqtgraph as pg
import json
from scipy.ndimage import rotate

from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool

from pdf_log import *
from xanes2d import *


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('/home/xf03id/user_macros/HXN_GUI/Scan/hxn_gui_admin2.ui', self)
        self.initParams()
        self.ImageCorrelationPage()
        self.webbrowserSetUpHxnWS1()

        self.energies = []
        self.roiDict = {}

        # updating resolution/tot time
        self.dwell.valueChanged.connect(self.initParams)
        self.x_step.valueChanged.connect(self.initParams)
        self.y_step.valueChanged.connect(self.initParams)
        self.x_start.valueChanged.connect(self.initParams)
        self.y_start.valueChanged.connect(self.initParams)
        self.x_end.valueChanged.connect(self.initParams)
        self.y_end.valueChanged.connect(self.initParams)

        # logic control for 1d or 2d scan selection
        self.rb_1d.clicked.connect(self.disableMot2)
        self.rb_2d.clicked.connect(self.enableMot2)
        self.rb_1d.clicked.connect(self.initParams)
        self.rb_2d.clicked.connect(self.initParams)

        # text files and editor controls
        self.pb_save_cmd.clicked.connect(self.save_file)
        self.pb_clear_cmd.clicked.connect(self.clear_cmd)
        self.pb_new_macro_gedit.clicked.connect(self.open_gedit)
        self.pb_new_macro_vi.clicked.connect(self.open_vi)
        self.pb_new_macro_emacs.clicked.connect(self.open_emacs)
        self.pb_browse_a_macro.clicked.connect(self.get_a_file)
        self.pb_ex_macro_open.clicked.connect(self.open_a_macro)

        # plotting controls
        self.pb_close_all_plot.clicked.connect(self.close_all_plots)
        self.pb_plot.clicked.connect(self.plot_me)
        self.pb_erf_fit.clicked.connect(self.plot_erf_fit)
        self.pb_plot_line_center.clicked.connect(self.plot_line_center)

        # xanes parameters
        self.pb_gen_elist.clicked.connect(self.generateEList)
        self.pb_set_epoints.clicked.connect(self.generate_epoints)
        # self.pb_start_xanes.clicked.connect(self.zpXANES)

        # scans and motor motion
        self.start.clicked.connect(self.initFlyScan)

        self.pb_move_smarx_pos.clicked.connect(self.move_smarx)
        self.pb_move_smary_pos.clicked.connect(self.move_smary)
        self.pb_move_smarz_pos.clicked.connect(self.move_smarz)
        self.pb_move_dth_pos.clicked.connect(self.move_dsth)
        self.pb_move_zpz_pos.clicked.connect(self.move_zpz1)

        self.pb_move_smarx_neg.clicked.connect(self.move_smarx(neg_=True))
        self.pb_move_smary_neg.clicked.connect(self.move_smary(neg_=True))
        self.pb_move_smarz_neg.clicked.connect(self.move_smarz(neg_=True))
        self.pb_move_dth_neg.clicked.connect(self.move_dsth(neg_=True))
        self.pb_move_zpz_neg.clicked.connect(self.move_zpz1(neg_=True))

        # Detector/Camera Motions
        self.pb_merlinOUT.clicked.connect(self.merlinOUT)
        self.pb_merlinIN.clicked.connect(self.merlinIN)
        self.pb_vortexOUT.clicked.connect(self.vortexOUT)
        self.pb_vortexIN.clicked.connect(self.vortexIN)
        self.pb_cam6IN.clicked.connect(self.cam6IN)
        self.pb_cam6OUT.clicked.connect(self.cam6OUT)
        self.pb_cam11IN.clicked.connect(self.cam11IN)

        # sample position
        self.pb_save_pos.clicked.connect(self.generatePositionDict)
        self.pb_roiList_import.clicked.connect(self.importROIDict)
        self.pb_roiList_export.clicked.connect(self.exportROIDict)
        self.pb_roiList_clear.clicked.connect(self.clearROIList)
        self.sampleROI_List.itemClicked.connect(self.showROIPos)
        self.pb_recover_scan_pos.clicked.connect(self.gotoPosSID)
        self.pb_show_scan_pos.clicked.connect(self.viewScanPosSID)
        self.pb_print_scan_meta.clicked.connect(self.viewScanMetaData)
        self.pb_recover_scan_pos.clicked.connect(self.gotoPosSID)
        self.pb_show_scan_pos.clicked.connect(self.viewScanPosSID)

        # Quick fill scan Params
        self.pb_3030.clicked.connect(self.fill_common_scan_params)
        self.pb_2020.clicked.connect(self.fill_common_scan_params)
        self.pb_66.clicked.connect(self.fill_common_scan_params)
        self.pb_22.clicked.connect(self.fill_common_scan_params)

        # elog
        self.pb_pdf_wd.clicked.connect(self.select_pdf_wd)
        self.pb_pdf_image.clicked.connect(self.select_pdf_image)
        self.pb_save_pdf.clicked.connect(self.force_save_pdf)
        self.pb_createpdf.clicked.connect(self.generate_pdf)
        self.pb_fig_to_pdf.clicked.connect(self.InsertFigToPDF)

        # admin control
        self.pb_apply_user_settings.clicked.connect(self.setUserLevel)

        # close the application
        self.actionClose_Application.triggered.connect(self.close_application)

        self.show()

    def webbrowserSetUpHxnWS1(self):
        try:
            chrome_path = '/usr/bin/google-chrome-stable'
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
            self.client = webbrowser.get('chrome')

        except:
            pass

    def setUserLevel(self):

        self.userButtonEnabler(self.cb_det_user, self.gb_det_control)
        self.userButtonEnabler(self.cb_xanes_user, self.rb_xanes_scan)
        self.userButtonEnabler(self.cb_xanes_user, self.gb_xanes_align)

    def userButtonEnabler(self, checkbox_name, control_btn_grp_name):

        if checkbox_name.isChecked():
            control_btn_grp_name.setEnabled(True)
        else:
            control_btn_grp_name.setEnabled(False)

    def getScanValues(self):
        self.det = self.pb_dets.currentText()

        self.mot1_s = self.x_start.value()
        self.mot1_e = self.x_end.value()
        self.mot1_steps = self.x_step.value()

        self.mot2_s = self.y_start.value()
        self.mot2_e = self.y_end.value()
        self.mot2_steps = self.y_step.value()

        self.dwell_t = self.dwell.value()

    def initParams(self):
        self.getScanValues()

        cal_res_x = (abs(self.mot1_s) + abs(self.mot1_e)) / self.mot1_steps
        cal_res_y = (abs(self.mot2_s) + abs(self.mot2_e)) / self.mot2_steps
        tot_t_2d = self.mot1_steps * self.mot2_steps * self.dwell_t / 60
        tot_t_1d = self.mot1_steps * self.dwell_t / 60

        if self.rb_1d.isChecked():
            self.label_scan_info_calc.setText(f'X: {(cal_res_x * 1000):.2f} nm, Y: {(cal_res_y * 1000):.2f} nm \n'
                                              f'{tot_t_1d:.2f} minutes + overhead')
            self.label_scanMacro.setText(f'fly1d({self.det}, {self.mot1_s}, '
                                         f'{self.mot1_e}, {self.mot1_steps}, {self.dwell_t:.3f})')

        else:
            self.label_scan_info_calc.setText(f'X: {(cal_res_x * 1000):.2f} nm, Y: {(cal_res_y * 1000):.2f} nm \n'
                                              f'{tot_t_2d:.2f} minutes + overhead')
            self.label_scanMacro.setText(f'fly2d({self.det}, {self.mot1_s}, {self.mot1_e}, {self.mot1_steps}, '
                                         f'{self.mot2_s},{self.mot2_e},{self.mot2_steps},{self.dwell_t:.3f})')

    def initFlyScan(self):
        self.getScanValues()

        self.motor1 = self.cb_motor1.currentText()
        self.motor2 = self.cb_motor2.currentText()

        self.motor_list = {'zpssx': zpssx, 'zpssy': zpssy, 'zpssz': zpssz}
        self.det_list = {'dets1': dets1, 'dets2': dets2, 'dets3': dets3,
                         'dets4': dets4, 'dets_fs': dets_fs}

        if self.rb_1d.isChecked():
            RE(fly1d(self.det_list[self.det], self.motor_list[self.motor1],
                     self.mot1_s, self.mot1_e, self.mot1_steps, self.dwell_t))

        else:
            RE(fly2d(self.det_list[self.det], self.motor_list[self.motor1], self.mot1_s, self.mot1_e, self.mot1_steps,
                     self.motor_list[self.motor2], self.mot2_s, self.mot2_e, self.mot2_steps, self.dwell_t))

    def disableMot2(self):
        self.y_start.setEnabled(False)
        self.y_end.setEnabled(False)
        self.y_step.setEnabled(False)

    def enableMot2(self):
        self.y_start.setEnabled(True)
        self.y_end.setEnabled(True)
        self.y_step.setEnabled(True)

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

    def moveAMotor(self, val_box, mot_name, unit_conv_factor: float = 1, neg=False):

        if neg:
            move_by = val_box.value() * -1
        else:
            move_by = val_box.value()

        RE(bps.movr(mot_name, move_by * unit_conv_factor))
        self.ple_info.appendPlainText(f'{mot_name.name} moved by {move_by} um ')

    def move_smarx(self, neg_=False):
        self.moveAMotor(self.db_move_smarx, smarx, 0.001, neg=neg_)

    def move_smary(self, neg_=False):
        self.moveAMotor(self.db_move_smary, smary, 0.001, neg=neg_)

    def move_smarz(self, neg_=False):
        self.moveAMotor(self.db_move_smarz, smarz, 0.001, neg=neg_)

    def move_dsth(self, neg_=False):
        self.moveAMotor(self.db_move_dth, zpsth, neg=neg_)

    def move_zpz1(self, neg_=False):
        if neg_:

            RE(movr_zpz1(self.db_move_zpz.value() * 0.001 * -1))

        else:
            RE(movr_zpz1(self.db_move_zpz.value() * 0.001))

    def ZP_OSA_OUT(self):
        RE(bps.movr(zposay, 2700))
        self.ple_info.appendPlainText('OSA Y moved OUT')

    def ZP_OSA_IN(self):
        RE(bps.movr(zposay, 2700))
        self.ple_info.appendPlainText('OSA Y moved IN')

    def merlinIN(self):
        self.client.open('http://10.66.17.43')
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            RE(go_det('merlin'))
        else:
            pass

    def merlinOUT(self):
        self.client.open('http://10.66.17.43')
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            RE(bps.mov(diff.x, -600))
        else:
            pass

    def vortexIN(self):
        RE(bps.mov(fdet1.x, -8))
        self.ple_info.appendPlainText('Vortex is IN')

    def vortexOUT(self):
        RE(bps.mov(fdet1.x, -107))
        self.ple_info.appendPlainText('Vortex is OUT')

    def cam11IN(self):
        self.client.open('http://10.66.17.43')
        QtTest.QTest.qWait(5000)
        choice = QMessageBox.question(self, 'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            RE(go_det('cam11'))
            self.ple_info.appendPlainText('CAM11 is IN')
        else:
            pass

    def cam6IN(self):
        caput('XF:03IDC-OP{Stg:CAM6-Ax:X}Mtr.VAL', 0)
        self.ple_info.appendPlainText('CAM6 Motion Done!')

    def cam6OUT(self):
        caput('XF:03IDC-OP{Stg:CAM6-Ax:X}Mtr.VAL', -50)
        self.ple_info.appendPlainText('CAM6 Motion Done!')

    def plot_me(self):
        sd = self.pb_plot_sd.text()
        elem = self.pb_plot_elem.text()
        plot_data(int(sd), elem, 'sclr1_ch4')

    def plot_erf_fit(self):
        sd = self.pb_plot_sd.text()
        elem = self.pb_plot_elem.text()
        erf_fit(int(sd), elem, linear_flag=self.cb_erf_linear_flag.isChecked())

    def plot_line_center(self):
        sd = self.pb_plot_sd.text()
        elem = self.pb_plot_elem.text()
        return_line_center(int(sd), elem, threshold=self.dsb_line_center_thre.value())

    def close_all_plots(self):
        plt.close('all')

    #xanes
    def generate_epoints(self):

        pre = np.linspace(self.dsb_pre_s.value(), self.dsb_pre_e.value(), self.sb_pre_p.value())
        XANES1 = np.linspace(self.dsb_ed1_s.value(), self.dsb_ed1_e.value(), self.sb_ed1_p.value())
        XANES2 = np.linspace(self.dsb_ed2_s.value(), self.dsb_ed2_e.value(), self.sb_ed2_p.value())
        post = np.linspace(self.dsb_post_s.value(), self.dsb_post_e.value(), self.sb_post_p.value())

        self.energies = np.concatenate([pre, XANES1, XANES2, post])
        self.ple_info.setPlainText(str(self.energies))

    def importEPoints(self):
        file_name = QFileDialog().getOpenFileName(self, "Save Parameter File", ' ',
                                                                 'txt file(*txt)')

        if file_name:
            self.energies = np.loadtxt(file_name[0])
        else:
            pass

    def exportEPoints(self):
        self.generate_epoints()
        file_name = QFileDialog().getSaveFileName(self, "Save Parameter File",
                                                            'xanes_e_points.txt',
                                                            'txt file(*txt)')
        if file_name:
            np.savetxt(file_name[0],np.array(self.energies))
        else:
            pass

    def importXanesParams(self):

        file_name = QFileDialog().getOpenFileName(self, "Save Parameter File", ' ',
                                                                 'json file(*json)')
        if file_name:
            with open(file_name[0], 'r') as fp:
                self.xanesParam = json.load(fp)
        else:
            pass

        self.fillXanesParamBoxes(self.xanesParam)

    def exportXanesParams(self):
        self.xanesParam = {}
        e_pos = {'low': self.dsb_monoe_l.value(), 'high':self.dsb_monoe_h.value()}
        ugap_pos= {'low': self.dsb_ugap_l.value(), 'high': self.dsb_ugap_h.value()}
        crl_pos = {'low': self.dsb_crl_l.value(), 'high': self.dsb_crl_h.value()}
        zpz1_pos = {'low': self.dsb_zpz_l.value(), 'high': self.dsb_zpz_h.value()}
        crl_combo = {'crl_combo_num': self.le_crl_combo_xanes.text()}

        self.xanesParam['mono_e'] = e_pos
        self.xanesParam['ugap'] = ugap_pos
        self.xanesParam['crl'] = crl_pos
        self.xanesParam['zpz1'] = zpz1_pos
        self.xanesParam['crl_combo'] = crl_combo

        file_name = QFileDialog().getSaveFileName(self, "Save Parameter File",
                                                            'hxn_xanes_parameters.json',
                                                            'json file(*json)')
        if file_name:

            with open(f'{file_name[0]}', 'w') as fp:
                json.dump(self.xanesParam,fp, indent=4)
        else:
            pass

    def fillXanesParamBoxes(self,xanesParam:dict ):

        e_low, e_high = xanesParam['mono_e']['low'], xanesParam['mono_e']['high']
        ugap_low, ugap_high = xanesParam['ugap']['low'], xanesParam['ugap']['high']
        crl_low, crl_high = xanesParam['crl']['low'], xanesParam['crl']['high']
        zpz1_low, zpz1_high = xanesParam['zpz1']['low'], xanesParam['zpz1']['high']
        crl_combo = xanesParam['crl_combo']['crl_combo_num']

        self.dsb_monoe_l.setValue(e_low), self.dsb_monoe_h.setValue(e_high)
        self.dsb_ugap_l.setValue(ugap_low), self.dsb_ugap_h.setValue(ugap_high)
        self.dsb_crl_l.setValue(crl_low), self.dsb_crl_h.setValue(crl_high)
        self.dsb_zpz_l.setValue(zpz1_low), self.dsb_zpz_h.setValue(zpz1_high)
        self.le_crl_combo_xanes.setText(crl_combo)

    def loadCommonXanesParams(self):
        with open(os.path.join('.','xanes_common_elem_params.json'), 'r') as fp:
            self.commonXanesParam = json.load(fp)

    def insertCommonXanesParams(self):
        mot_list = [self.dsb_monoe_l, self.dsb_monoe_h, self.dsb_ugap_l, self.dsb_ugap_h,
                    self.dsb_crl_l, self.dsb_crl_h, self.dsb_zpz_l, self.dsb_zpz_h, self.le_crl_combo_xanes]
        commonElems = self.commonXanesParam.keys()

        button_name = self.sender().objectName()
        if button_name in commonElems:
            elemParam = self.commonXanesParam[button_name]
            self.fillXanesParamBoxes(elemParam)

        else:
            pass

    def generateEList(self):

        if not len(self.energies) == 0:

            # print(energies)
            dE = (self.dsb_monoe_h.value() - self.dsb_monoe_l.value())

            ugap_slope = (self.dsb_ugap_h.value() - self.dsb_ugap_l.value()) / dE
            ugap_list = self.dsb_ugap_h.value() + (self.energies - self.dsb_monoe_h.value()) * ugap_slope

            crl_slope = (self.dsb_crl_h.value() - self.dsb_crl_l.value()) / dE
            crl_list = self.dsb_crl_h.value() + (self.energies - self.dsb_monoe_h.value()) * crl_slope

            zpz_slope = (self.dsb_zpz_h.value() - self.dsb_zpz_l.value()) / dE
            zpz_list = self.dsb_zpz_h.value() + (self.energies - self.dsb_monoe_h.value()) * zpz_slope

            self.e_list = np.column_stack((self.energies, ugap_list, zpz_list, crl_list))
            self.ple_info.setPlainText(str(self.e_list))

        else:
            self.statusbar.showMessage('No energy list found; set or load an e list first')

    def zpXANES(self):
        self.getScanValues()
        RE(zp_list_xanes2d(self.e_list, self.det_list[self.det], self.motor_list[self.motor1],
                           self.mot1_s, self.mot1_e, self.mot1_steps, self.motor_list[self.motor2],
                           self.mot2_s, self.mot2_e, self.mot2_steps, self.dwell_t))

    #tomo
    def zpTomoStepResCalc(self):
        pass

    def zpTomo(self):

        startAngle = self.sb_tomo_start_angle.value()
        endAngle = self.sb_tomo_end_angle.value()
        stepsAngle = self.sb_tomo_steps.value()

        xAlignStart = None
        xAlignEnd = None
        xAlignSteps = None
        xAlignDwell = None
        xAlignElem = None
        xAlignThreshold = None

        yAlignStart = None
        yAlignEnd = None
        yAlignSteps = None
        yAlignDwell = None
        yAlignElem = None
        yAlignThreshold = None



    #special scans
    def zpMosaic(self):
        pass

    def zpFocusScan(self):
        pass

    def zpRotAlignment(self):
        pass

    #custom macros

    def save_file(self):
        S__File = QFileDialog.getSaveFileName(None, 'SaveFile', '/', "Python Files (*.py)")

        Text = self.pte_run_cmd.toPlainText()
        if S__File[0]:
            with open(S__File[0], 'w') as file:
                file.write(Text)

    def clear_cmd(self):
        self.pte_run_cmd.clear()

    def open_gedit(self):
        subprocess.Popen(['gedit'])

    def open_vi(self):
        subprocess.Popen(['vi'])

    def open_emacs(self):
        subprocess.Popen(['emacs'])

    def get_a_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file")
        self.le_ex_macro.setText(str(file_name[0]))

    def open_a_macro(self):
        editor = self.cb_ex_macro_with.currentText()
        filename = self.le_ex_macro.text()

        subprocess.Popen([editor, filename])

    def abort_scan(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        RE.abort()

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

    def generatePositionDict(self):

        fx, fy, fz = zpssx.position, zpssy.position, zpssz.position
        cx, cy, cz = smarx.position, smary.position, smarz.position
        zpz1_pos = zp.zpz1.position
        zp_sx, zp_sz = zps.zpsx.position, zps.zpsz.position
        th = zpsth.position
        roi = {
            zpssx: fx, zpssy: fy, zpssz: fz,
            smarx: cx, smary: cy, smarz: cz,
            zp.zpz1: zpz1_pos, zpsth: th,
            zps.zpsx: zp_sx, zps.zpsz: zp_sz
        }
        roi_name = 'ROI' + str(self.sampleROI_List.count())
        self.roiDict[roi_name] = roi
        self.sampleROI_List.addItem(roi_name)
        item = self.sampleROI_List.item(self.sampleROI_List.count()-1)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

    def applyDictWithLabel(self):
        label_ = {}
        for idx in range(self.sampleROI_List.count()):
            label = self.sampleROI_List.item(idx).text()
            label_[label] = idx
        self.roiDict['user_labels'] = label_
        print(self.roiDict)

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
        print(self.roiDict[f'ROI{item_num}'])

    def gotoROIPosition(self):
        roi_num = self.sampleROI_List.currentRow()
        param_file = self.roiDict[roi_num]
        for key, value in param_file.items():
            if not key == zp.zpz1:
                RE(bps.mov(key, value))
            elif key == zp.zpz1:
                RE(mov_zpz1(value))
            self.ple_info.appendPlainText(f'Sample moved to {key.name}:{value:.4f} ')

    def showROIPosition(self, item):
        item_num = self.sampleROI_List.row(item)
        param_file = self.roiDict[item_num]
        self.ple_info.appendPlainText(('*' * 20))
        for key, value in param_file.items():
            self.ple_info.appendPlainText(f'{key.name}:{value:.4f}')

        #self.sampleROI_List.itemClicked.connect(lambda: self.ple_info.appendPlainText(
        # (self.roiDict[self.sampleROI_List.currentItem().text()])))

    def gotoPosSID(self):
        sd = self.le_sid_position.text()
        recover_zp_scan_pos(int(sid), 1, 1)
        self.ple_info.appendPlainText(f'Positions recovered from {sid}')

    def viewScanPosSID(self):
        sd = self.le_sid_position.text()
        self.ple_info.appendPlainText(str(RE(recover_zp_scan_pos(int(sd), 0, 0))))

    def viewScanMetaData(self):
        sd = self.le_sid_position.text()
        h = db[int(sd)]
        self.ple_info.appendPlainText(str(h.start))

    #Image correlation tool

    def ImageCorrelationPage(self):

        self.coords = collections.deque(maxlen=4)

        # connections
        self.pb_RefImageLoad.clicked.connect(self.loadRefImage)
        self.pb_apply_calculation.clicked.connect(self.scalingCalculation)
        self.dsb_x_off.valueChanged.connect(self.offsetCorrectedPos)
        self.dsb_y_off.valueChanged.connect(self.offsetCorrectedPos)
        self.pb_grabXY_1.clicked.connect(self.insertCurrentPos1)
        self.pb_grabXY_2.clicked.connect(self.insertCurrentPos2)
        self.pb_import_param.clicked.connect(self.importScalingParamFile)
        self.pb_export_param.clicked.connect(self.exportScalingParamFile)
        self.pb_gotoTargetPos.clicked.connect(self.gotoTargetPos)

    def loadRefImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                                 'image file(*png *jpeg *tiff *tif )')
        if self.file_name:
            self.ref_image = plt.imread(self.file_name[0])
            if self.ref_image.ndim == 3:
                self.ref_image = self.ref_image.sum(2)
            self.statusbar.showMessage(f'{self.file_name[0]} selected')
        else:
            self.statusbar.showMessage("No file has selected")
            pass

        try:
            self.ref_view.clear()
        except:
            pass

        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.ref_view.addPlot(title="")

        # Item for displaying image data
        self.img = pg.ImageItem()
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        self.ref_view.addItem(hist)

        self.p1.addItem(self.img)
        self.ref_image = rotate(self.ref_image, -90)
        self.img.setImage(self.ref_image)
        self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        # self.img.translate(100, 50)
        # self.img.scale(0.5, 0.5)
        self.img.hoverEvent = self.imageHoverEvent
        self.img.mousePressEvent = self.MouseClickEvent

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        val = self.ref_image[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = np.around(ppos.x(), 2), np.around(ppos.y(), 2)
        self.p1.setTitle(f'pos: {x, y}  pixel: {i, j}  value: {val}')

    def MouseClickEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.button() == QtCore.Qt.LeftButton:

            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
            j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
            self.coords.append((i, j))
            val = self.ref_image[i, j]
            ppos = self.img.mapToParent(pos)
            # x, y = np.around(ppos.x(), 2) , np.around(ppos.y(), 2)
            x, y = smarx.position, smary.position
            self.coords.append((x, y))
            if len(self.coords) == 2:
                self.le_ref1_pxls.setText(f'{self.coords[0][0]}, {self.coords[0][1]}')
                self.dsb_ref1_x.setValue(self.coords[1][0])
                self.dsb_ref1_y.setValue(self.coords[1][1])
            elif len(self.coords) == 4:
                self.le_ref1_pxls.setText(f'{self.coords[0][0]}, {self.coords[0][1]}')
                self.dsb_ref1_x.setValue(self.coords[1][0])
                self.dsb_ref1_y.setValue(self.coords[1][1])
                self.le_ref2_pxls.setText(f'{self.coords[2][0]}, {self.coords[2][1]}')
                self.dsb_ref2_x.setValue(self.coords[-1][0])
                self.dsb_ref2_y.setValue(self.coords[-1][1])

    def createLabAxisImage(self):
        # A plot area (ViewBox + axes) for displaying the image

        try:
            self.labaxis_view.clear()
        except:
            pass

        self.p2 = self.labaxis_view.addPlot(title="")

        # Item for displaying image data
        self.img2 = pg.ImageItem()
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img2)
        self.labaxis_view.addItem(hist)
        self.p2.addItem(self.img2)
        self.img2.setImage(self.ref_image)
        self.img2.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        # self.img2.setImage(self.ref_image.T,opacity = 0.5)

    def getScalingParams(self):

        self.lm1_px, self.lm1_py = self.le_ref1_pxls.text().split(',')  # r chooses this pixel
        self.lm2_px, self.lm2_py = self.le_ref2_pxls.text().split(',')  # chooses this pixel

        # motor values from the microscope at pixel pos 1
        self.lm1_x, self.lm1_y = self.dsb_ref1_x.value(), self.dsb_ref1_y.value()
        # motor values from the microscope at pixel pos 2
        self.lm2_x, self.lm2_y = self.dsb_ref2_x.value(), self.dsb_ref2_y.value()

    def exportScalingParamFile(self):
        self.getScalingParams()
        self.scalingParam = {}
        ref_pos1 = {'px1': int(self.lm1_px), 'py1':int(self.lm1_py), 'cx1':self.lm1_x, 'cy1':self.lm1_y}
        ref_pos2 = {'px2': int(self.lm2_px), 'py2': int(self.lm2_py), 'cx2': self.lm2_x, 'cy2': self.lm2_y}
        self.scalingParam['lm1_vals'] = ref_pos1
        self.scalingParam['lm2_vals'] = ref_pos2

        file_name = QtWidgets.QFileDialog().getSaveFileName(self, "Save Parameter File", 'scaling_parameters.json',
                                                                 'json file(*json)')
        if file_name:

            with open(f'{file_name[0]}', 'w') as fp:
                json.dump(self.scalingParam,fp, indent=4)
        else:
            pass

    def importScalingParamFile(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open Parameter File", '',
                                                                 'json file(*json)')
        if file_name:
            with open(file_name[0], 'r') as fp:
                self.scalingParam = json.load(fp)
        else:
            pass

        px1, py1 = self.scalingParam['lm1_vals']['px1'], self.scalingParam['lm1_vals']['py1']
        px2, py2 = self.scalingParam['lm2_vals']['px2'], self.scalingParam['lm2_vals']['py2']

        self.le_ref1_pxls.setText(f'{px1},{py1}')
        self.dsb_ref1_x.setValue(self.scalingParam['lm1_vals']['cx1'])
        self.dsb_ref1_y.setValue(self.scalingParam['lm1_vals']['cy1'])
        self.le_ref2_pxls.setText(f'{px2},{py2}')
        self.dsb_ref2_x.setValue(self.scalingParam['lm2_vals']['cx2'])
        self.dsb_ref2_y.setValue(self.scalingParam['lm2_vals']['cy2'])

    def scalingCalculation(self):
        self.generateScalingParam()
        self.yshape, self.xshape = np.shape(self.ref_image)
        self.pixel_val_x = (self.lm2_x - self.lm1_x) / (int(self.lm2_px) - int(self.lm1_px))  # pixel value of X
        self.pixel_val_y = (self.lm2_y - self.lm1_y) / (int(self.lm2_py) - int(self.lm1_py))  # pixel value of Y; ususally same as X

        self.xi = self.lm1_x - (self.pixel_val_x * int(self.lm1_px))  # xmotor pos at origin (0,0)
        xf = self.xi + (self.pixel_val_x * self.xshape)  # xmotor pos at the end (0,0)
        self.yi = self.lm1_y - (self.pixel_val_y * int(self.lm1_py))  # xmotor pos at origin (0,0)
        yf = self.yi + (self.pixel_val_y * self.yshape)  # xmotor pos at origin (0,0)
        self.createLabAxisImage()

        self.label_scale_info.setText(f'Scaling: {self.pixel_val_x:.4f}, {self.pixel_val_y:.4f}, \n '
                                      f' X Range {self.xi:.2f}:{xf:.2f}, \n'
                                      f'Y Range {self.yi:.2f}:{yf:.2f}')
        self.img2.scale(abs(self.pixel_val_x), abs(self.pixel_val_y))
        self.img2.translate(self.xi, self.yi)
        # self.img2.setRect(QtCore.QRect(xi,yf,yi,xf))
        self.img2.hoverEvent = self.imageHoverEvent2
        self.img2.mousePressEvent = self.MouseClickEventToPos

    def imageHoverEvent2(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p2.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        val = self.ref_image[i, j]
        x = self.xi + (self.pixel_val_x * i)
        y = self.yi + (self.pixel_val_y * j)
        self.p2.setTitle(f'pos: {x:.2f},{y:.2f}  pixel: {i, j}  value: {val:.2f}')

    def MouseClickEventToPos(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
            j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
            self.xWhere = self.xi + (self.pixel_val_x * i)
            self.yWhere = self.yi + (self.pixel_val_y * j)
            self.offsetCorrectedPos()

    def offsetCorrectedPos(self):
        self.dsb_calc_x.setValue(self.xWhere + (self.dsb_x_off.value() * 0.001))
        self.dsb_calc_y.setValue(self.yWhere + (self.dsb_y_off.value() * 0.001))

    def insertCurrentPos1(self):
        posX = smarx.position
        posY = smary.position

        self.dsb_ref1_x.setValue(posX)
        self.dsb_ref1_y.setValue(posY)

    def insertCurrentPos2(self):
        posX = smarx.position
        posY = smary.position

        self.dsb_ref2_x.setValue(posX)
        self.dsb_ref2_y.setValue(posY)

    def gotoTargetPos(self):
        targetX = self.dsb_calc_x.value()
        targetY = self.dsb_calc_y.value()
        RE(bps.mov(smarx, targetX))
        RE(bps.mov(smary, targetY))

    #exit gui

    def close_application(self):

        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            plt.close('all')
            print('quit application')
            sys.exit()
        else:
            pass

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error:`tuple` (exctype, value, traceback.format_exc() )
    - result: `object` data returned from processing, anything
    - progress: `tuple` indicating progress metadata
    '''
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
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
