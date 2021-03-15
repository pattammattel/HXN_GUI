# -*- coding: utf-8 -*-

#Author: Ajith Pattammattel
#Original Date:06-23-2020

import os
import signal
import subprocess
import sys
import collections
import pyqtgraph as pg
from scipy.ndimage import rotate

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from pdf_log import *
from xanes2d import *



class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('/home/xf03id/user_macros/HXN_GUI/Scan/hxn_gui_admin2.ui', self)
        self.initParams()
        self.ImageCorrelationPage()

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

        #plotting controls
        self.pb_close_all_plot.clicked.connect(self.close_all_plots)
        self.pb_close_plot.clicked.connect(self.close_plot)
        self.pb_plot.clicked.connect(self.plot_me)
        self.pb_erf_fit.clicked.connect(self.plot_erf_fit)

        #generate xanes parameters
        self.pb_gen_elist.clicked.connect(self.generate_elist)
        #self.pb_start_xanes.clicked.connect(self.zp_xanes)

        #scans and motor motion
        self.start.clicked.connect(self.initFlyScan)
        self.pb_move_smarx.clicked.connect(self.move_smarx)
        self.pb_move_smary.clicked.connect(self.move_smary)
        self.pb_move_smarz.clicked.connect(self.move_smarz)
        self.pb_move_dth.clicked.connect(self.move_dsth)
        self.pb_move_zpz.clicked.connect(self.move_zpz1)
        
        #Detector/Camera Motions
        self.pb_merlinOUT.clicked.connect(self.merlinOUT)
        self.pb_merlinIN.clicked.connect(self.merlinIN)
        self.pb_vortexOUT.clicked.connect(self.vortexOUT)
        self.pb_vortexIN.clicked.connect(self.vortexIN)
        self.pb_cam6IN.clicked.connect(self.cam6IN)
        self.pb_cam6OUT.clicked.connect(self.cam6OUT)
        self.pb_cam11IN.clicked.connect(self.cam11IN)

        #Quick fill scan Params
        self.pb_3030.clicked.connect(self.fill_common_scan_params)
        self.pb_2020.clicked.connect(self.fill_common_scan_params)
        self.pb_66.clicked.connect(self.fill_common_scan_params)
        self.pb_22.clicked.connect(self.fill_common_scan_params)

        #admin control
        self.pb_apply_user_settings.clicked.connect(self.setUserLevel)

        #elog 
        self.pb_folder_log.clicked.connect(self.select_pdf_wd)
        self.pb_pdf_image.clicked.connect(self.select_pdf_image)
        self.pb_date_ok.clicked.connect(self.generate_pdf)
        self.pb_save_pdf.clicked.connect(self.force_save_pdf)
        self.pb_createpdf.clicked.connect(self.insert_pdf)

        #close the application
        self.actionClose_Application.triggered.connect(self.close_application)

        self.show()

    def setUserLevel(self):

        self.userButtonEnabler(self.cb_det_user, self.gb_det_control)
        self.userButtonEnabler(self.cb_xanes_user, self.rb_xanes_scan)
        self.userButtonEnabler(self.cb_xanes_user, self.gb_xanes_align)

    def userButtonEnabler(self,checkbox_name,control_btn_grp_name):

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
        tot_t = self.mot1_steps * self.mot2_steps * self.dwell_t / 60
        self.label_scan_info_calc.setText(f'X: {(cal_res_x * 1000):.2f} nm, Y: {(cal_res_y * 1000):.2f} nm \n'
                                          f'{tot_t:.2f} minutes + overhead')

        if self.rb_1d.isChecked():

            self.label_scanMacro.setText(f'fly1d({self.det}, {self.mot1_s}, '
                                         f'{self.mot1_e}, {self.mot1_steps}, {self.dwell_t:.3f})')

        else:
            self.label_scanMacro.setText(f'fly2d({self.det}, {self.mot1_s}, {self.mot1_e}, {self.mot1_steps}, '
                                         f'{self.mot2_s},{self.mot2_e},{self.mot2_steps},{self.dwell_t})')

    def initFlyScan(self):
        self.getScanValues()

        self.motor1 = self.cb_motor1.currentText()
        self.motor2 = self.cb_motor2.currentText()
        
        self.motor_list = {'zpssx':zpssx,'zpssy':zpssy,'zpssz':zpssz}
        self.det_list = {'dets1': dets1, 'dets2': dets2, 'dets3': dets3,
                    'dets4': dets4, 'dets_fs': dets_fs}
 

        if self.rb_1d.isChecked():
            RE(fly1d(self.det_list[self.det], self.motor_list[self.motor1],
                     self.mot1_s,self.mot1_e ,self.mot1_steps, self.dwell_t))
        
        else:
            RE(fly2d(self.det_list[self.det], self.motor_list[self.motor1], self.mot1_s,self.mot1_e ,self.mot1_steps,
                     self.motor_list[self.motor2], self.mot2_s,self.mot2_e, self.mot2_steps, self.dwell_t))

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
        button_names = {'pb_2020':(20,20,100,100,0.03),
                        'pb_3030':(30,30,30,30,0.03),
                        'pb_66':(6,6,100,100,0.05),
                       'pb_22':(2,2,100,100,0.03)
                        }
        if button_name.objectName() in button_names.keys():

            valsToFill = button_names[button_name.objectName()]
            self.x_start.setValue(valsToFill[0]/-2)
            self.x_end.setValue(valsToFill[0]/2)
            self.y_start.setValue(valsToFill[1]/-2)
            self.y_end.setValue(valsToFill[1]/2)
            self.x_step.setValue(valsToFill[2])
            self.y_step.setValue(valsToFill[3])
            self.dwell.setValue(valsToFill[4])

    def moveAMotor(self,val_box,mot_name, unit_conv_factor:float = 1):
        move_by = val_box.value()
        RE(bps.movr(mot_name, move_by * unit_conv_factor))

    def move_smarx(self):
        self.moveAMotor(self.db_move_smarx, smarx, 0.001)

    def move_smary(self):
        self.moveAMotor(self.db_move_smary, smary, 0.001)

    def move_smarz(self):
        self.moveAMotor(self.db_move_smarz, smarz, 0.001)

    def move_dsth(self):
        self.moveAMotor(self.db_move_dth, zpsth)

    def move_zpz1(self):
        RE(movr_zpz1(self.db_move_zpz.value()*0.001))
        
    def merlinIN(self):
        RE(go_det('merlin'))
        
    def merlinOUT(self):
        RE(bps.mov(diff.x,-600))
                
    def vortexIN(self):
        RE(bps.mov(fdet1.x,-8))
        
    def vortexOUT(self):
        RE(bps.mov(fdet1.x,-107))
        
    def cam11IN(self):
        RE(go_det('cam11'))

    def cam6IN(self):
        RE(bps.mov(cam6_x, 0))

    def cam6OUT(self):
        RE(bps.mov(cam6_x, -50))

    def generate_elist(self):

        pre = np.linspace(self.dsb_pre_s.value(), self.dsb_pre_e.value(), self.sb_pre_p.value())
        XANES1 = np.linspace(self.dsb_ed1_s.value(), self.dsb_ed1_e.value(), self.sb_ed1_p.value())
        XANES2 = np.linspace(self.dsb_ed2_s.value(), self.dsb_ed2_e.value(), self.sb_ed2_p.value())
        post = np.linspace(self.dsb_post_s.value(), self.dsb_post_e.value(), self.sb_post_p.value())

        self.energies = np.concatenate([pre, XANES1, XANES2, post])

        # print(energies)
        dE = (self.dsb_monoe_h.value() - self.dsb_monoe_l.value())

        ugap_slope = (self.dsb_ugap_h.value() - self.dsb_ugap_l.value()) / dE
        ugap_list = self.dsb_ugap_h.value() + (self.energies - self.dsb_monoe_h.value()) * ugap_slope

        crl_slope = (self.dsb_crl_h.value() - self.dsb_crl_l.value()) / dE
        crl_list = self.dsb_crl_h.value() + (self.energies - self.dsb_monoe_h.value()) * crl_slope

        zpz_slope = (self.dsb_zpz_h.value() - self.dsb_zpz_l.value()) / dE
        zpz_list = self.dsb_zpz_h.value() + (self.energies - self.dsb_monoe_h.value()) * zpz_slope

        self.e_list = np.column_stack((self.energies, ugap_list, zpz_list, crl_list))

    def zp_xanes(self):
        self.getScanValues()
        RE(zp_list_xanes2d(self.e_list, self.det_list[self.det], self.motor_list[self.motor1],
                           self.mot1_s, self.mot1_e, self.mot1_steps, self.motor_list[self.motor2],
                           self.mot2_s, self.mot2_e, self.mot2_steps, self.dwell_t))

    def plot_me(self):

        sd = self.lineEdit_5.text()
        elem = self.lineEdit_6.text()
        plot_data(int(sd), elem, 'sclr1_ch4')

    def plot_erf_fit(self):
        sd = self.lineEdit_5.text()
        elem = self.lineEdit_6.text()
        erf_fit(int(sd),elem, linear_flag = self.cb_erf_linear_flag.isChecked())
            
    def save_file(self):
      S__File = QFileDialog.getSaveFileName(None,'SaveFile','/', "Python Files (*.py)")
    
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
        
    def close_all_plots(self):
        return plt.close('all')
        
    def close_plot(self):
        return plt.close()
        
    def abort_scan(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        RE.abort()

    def select_pdf_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_folder_log.setText(str(folder_path))
        
    def select_pdf_image(self):
        file_name = QFileDialog().getOpenFileName(self, "Select an Image")
        self.le_elog_image.setText(str(file_name[0]))

    def generate_pdf(self):
        dt = self.dateEdit_elog.date()
        tmp_date = dt.toString(self.dateEdit_elog.displayFormat())
        tmp_file = os.path.join(self.le_folder_log.text(),self.le_elog_name.text())
        tmp_sample = self.le_elog_sample.text()
        tmp_experimenter = self.le_elog_experimenters.text()
        tmp_pic = self.le_elog_image.text()
        
        return setup_pdf_for_gui(tmp_file, tmp_date, tmp_sample, tmp_experimenter, tmp_pic)
        
    def insert_pdf(self):
        return insertTitle_for_gui()
        
    def force_save_pdf(self):
        return save_page_for_gui()

    def ImageCorrelationPage(self):

        self.coords = collections.deque(maxlen=4)

        #connections
        self.pb_RefImageLoad.clicked.connect(self.loadRefImage)
        self.pb_apply_calculation.clicked.connect(self.scalingCalculation)
        self.dsb_x_off.valueChanged.connect(self.offsetCorrectedPos)
        self.dsb_y_off.valueChanged.connect(self.offsetCorrectedPos)
        self.pb_grabXY_1.clicked.connect(self.insertCurrentPos1)
        self.pb_grabXY_2.clicked.connect(self.insertCurrentPos2)
        #self.pb_grabXY_2.clicked.connect(self.insertCurrentPos(self.dsb_ref2_x,self.dsb_ref2_y))
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
            #x, y = np.around(ppos.x(), 2) , np.around(ppos.y(), 2)
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
        #self.img2.setImage(self.ref_image.T,opacity = 0.5)

    def scalingCalculation(self):
        yshape, xshape = np.shape(self.ref_image)

        lm1_px, lm1_py = self.le_ref1_pxls.text().split(',')  # r chooses this pixel
        lm2_px, lm2_py = self.le_ref2_pxls.text().split(',')  # chooses this pixel

        lm1_x, lm1_y = self.dsb_ref1_x.value(), self.dsb_ref1_y.value()  # motor values from the microscope at pixel pos 1
        lm2_x, lm2_y = self.dsb_ref2_x.value(), self.dsb_ref2_y.value()  # motor values from the microscope at pixel pos 2

        self.pixel_val_x = (lm2_x - lm1_x) / (int(lm2_px) - int(lm1_px))  # pixel value of X
        self.pixel_val_y = (lm2_y - lm1_y) / (int(lm2_py) - int(lm1_py))  # pixel value of Y; ususally same as X

        self.xi = lm1_x - (self.pixel_val_x * int(lm1_px))  # xmotor pos at origin (0,0)
        xf = self.xi + (self.pixel_val_x * xshape)  # xmotor pos at the end (0,0)
        self.yi = lm1_y - (self.pixel_val_y * int(lm1_py))  # xmotor pos at origin (0,0)
        yf = self.yi + (self.pixel_val_y * yshape)  # xmotor pos at origin (0,0)
        self.createLabAxisImage()

        self.label_scale_info.setText(f'Scaling: {self.pixel_val_x:.4f}, {self.pixel_val_y:.4f}, \n '
                                      f' X Range {self.xi:.2f}:{xf:.2f}, \n'
                                      f'Y Range {self.yi:.2f}:{yf:.2f}')
        self.img2.scale(abs(self.pixel_val_x), abs(self.pixel_val_y))
        self.img2.translate(self.xi, self.yi)
        #self.img2.setRect(QtCore.QRect(xi,yf,yi,xf))
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

    def close_application(self):

        choice = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())


