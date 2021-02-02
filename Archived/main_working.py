# -*- coding: utf-8 -*-

#Author: Ajith Pattammattel
#Data:06-23-2020



from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
import sys


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('fly_gui_edit_v2.ui', self)
                
        self.pb_exit.clicked.connect(self.close_application)
        self.dwell.valueChanged.connect(self.calc_res) # updating resolution/tot time
        self.x_step.valueChanged.connect(self.calc_res)#
        self.y_step.valueChanged.connect(self.calc_res)#
        self.x_start.valueChanged.connect(self.calc_res)#
        self.y_start.valueChanged.connect(self.calc_res)#
        self.x_end.valueChanged.connect(self.calc_res)#
        self.y_end.valueChanged.connect(self.calc_res)#
        
        self.start.clicked.connect(self.generate_flyscan) 
        self.pb_plot.clicked.connect(self.plot_me)
        self.show()

    def generate_flyscan(self):
   
        motor1 = self.Select_Motor_4.currentText()
        motor2 = self.Select_Motor_3.currentText()
        det = self.Dets_2.currentText()
        
        mot1_s = self.x_start.value()
        mot1_e = self.x_end.value()
        mot1_steps = self.x_step.value() 
        mot2_s = self.y_start.value()
        mot2_e = self.y_end.value()
        mot2_steps = self.y_step.value()
        dwell_t = self.dwell.value()

        cal_res_x = (abs(mot1_s)+abs(mot1_e))/mot1_steps
        cal_res_y = (abs(mot2_s)+abs(mot2_e))/mot2_steps
 

        if self.radioButton_1d.isChecked():
            #self.Select_Motor_3.setEnabled(False)
            RE (fly1d([zebra, sclr1, merlin2] , motor1, mot1_s,mot1_e ,mot1_steps, dwell_t))
        else:             
            RE(fly2d([zebra, sclr1, merlin2], motor1, mot1_s,mot1_e ,mot1_steps, motor2, mot2_s,mot2_e, mot2_steps, False))
        

    def calc_res(self):
        mot1_s = self.x_start.value()
        mot1_e = self.x_end.value()
        mot1_steps = self.x_step.value()

        mot2_s = self.y_start.value()
        mot2_e = self.y_end.value()       
        mot2_steps = self.y_step.value()
        
        dwell_t = self.dwell.value()
        
        cal_res_x = (abs(mot1_s)+abs(mot1_e))/mot1_steps
        cal_res_y = (abs(mot2_s)+abs(mot2_e))/mot2_steps
        self.res.setText(str(cal_res_x*1000))

        tot_t = str(mot1_steps*mot2_steps*dwell_t/60)
        self.tot_time.setText(tot_t)
        
        
        
            
    def plot_me(self):
    
        
        sd = self.lineEdit_5.text()
        elem = self.lineEdit_6.text()
        if self.ch_b_norm.isChecked():
            plot_data(sd, elem, 'sclr1_ch4')
        else:
            plot_data(sd, elem)


        
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


