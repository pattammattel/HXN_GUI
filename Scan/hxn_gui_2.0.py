# -*- coding: utf-8 -*-

#Author: Ajith Pattammattel
#Data:06-23-2020

import sys, os, signal, subprocess
import numpy as np

from pdf_log import *
from xanes2d import*
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QFileDialog




class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('hxn_gui_admin2.ui', self)

        # updating resolution/tot time
        self.dwell.valueChanged.connect(self.initParams)
        self.x_step.valueChanged.connect(self.initParams)
        self.y_step.valueChanged.connect(self.initParams)
        self.x_start.valueChanged.connect(self.initParams)
        self.y_start.valueChanged.connect(self.initParams)
        self.x_end.valueChanged.connect(self.initParams)
        self.y_end.valueChanged.connect(self.initParams)

        # logic control for 1d or 2d scan selection
        self.self.rb_1d.toggled.connect(self.disableMot2)
        self.self.rb_2d.toggled.connect(self.enableMot2)

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

        #generate xanes parameters
        self.pb_gen_elist.clicked.connect(self.generate_elist)
        self.pb_start_xanes.clicked.connect(self.zp_xanes)

        #scans and motor motion
        self.start.clicked.connect(self.initFlyScan)
        self.pb_move_smarx.clicked.connect(self.move_smarx)
        self.pb_move_smary.clicked.connect(self.move_smary)
        self.pb_move_smarz.clicked.connect(self.move_smarz)
        self.pb_move_dth.clicked.connect(self.move_dsth)

        #Quick fill scan Params
        self.pb_3030.clicked.connect(self.fill_common_scan_params)
        self.pb_2020.clicked.connect(self.fill_common_scan_params)
        self.pb_66.clicked.connect(self.fill_common_scan_params)
        self.pb_22.clicked.connect(self.fill_common_scan_params)

        #admin control
        self.pb_apply_user_settings.clicked.connect(self.setUserLevel)

        '''
        #elog 
        self.pb_folder_log.clicked.connect(self.select_pdf_wd)
        self.pb_pdf_image.clicked.connect(self.select_pdf_image)
        self.pb_date_ok.clicked.connect(self.generate_pdf)
        self.pb_save_pdf.clicked.connect(self.force_save_pdf)
        self.pb_createpdf.clicked.connect(self.insert_pdf)
        '''
        #close the application
        self.pb_exit.clicked.connect(self.close_application)

        self.show()

    def setUserLevel(self):

        self.userEnabler(self.cb_det_user, self.gb_det_control)
        self.userEnabler(self.cb_xanes_user, self.rb_xanes_scan)
        self.userEnabler(self.cb_xanes_user, self.gb_xanes_align)


    def userEnabler(self,checkbox_name,control_btn_grp_name):

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
        tot_t = str(self.mot1_steps * self.mot2_steps * self.dwell_t / 60)
        self.label_scan_info_calc.setText(f'X: {(cal_res_x * 1000):.2f}, Y: {(cal_res_y * 1000).:2f} \n'
                                          f'{tot_t} + overhead')

        if self.rb_1d.isChecked():

            self.label_scanMacro.setText(f'fly2d({det}, {self.mot1_s}, '
                                         f'{self.mot1_e}, {self.mot1_steps}, {self.dwell_t})')

        else:
            self.label_scanMacro.setText(f'fly2d({det}, {self.mot1_s}, {self.mot1_e}, {self.mot1_steps}, '
                                         f'{self.mot2_s},{self.mot2_e},{self.mot2_steps},{self.dwell_t})')

    def initFlyScan(self):
        self.getScanValues()
        
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
        self.y_start.setEnable(False)
        self.y_end.setEnable(False)
        self.y_step.setEnable(False)

    def enableMot2(self):
        self.y_start.setEnable(False)
        self.y_end.setEnable(False)
        self.y_step.setEnable(False)

    def fill_common_scan_params(self):
        button_name = self.sender()
        button_names = {'pb_2020':(20,20,100,100,0.03),
                        'pb_3030':(30,30,30,30,0.03),
                        'pb_66':(6,6,100,100,0.05),
                       'pb22':(2,2,100,100,0.03)
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
        
    def merlinIN(self):
        RE(go_det('merlin'))
        
    def merlinOUT(self):
        RE(bps.mov(diff_x,-400))
                
    def vortexIN(self):
        RE(bps.mov(fdet1_x,-8))
        
    def vortexOUT(self):
        RE(bps.mov(fdet1_x,-107))
        
    def cam11IN(self):
        RE(go_det('cam11'))

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
        if self.ch_b_norm.isChecked():
            plot_data(int(sd), elem, 'sclr1_ch4')
        else:
            plot_data(sd, elem)
            
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


