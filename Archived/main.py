from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('fly_gui_edit.ui', self)
                #All the connections done here 
        self.dwell.valueChanged.connect(self.calc_res) # updating resolution/tot time
        self.x_step.valueChanged.connect(self.calc_res)#
        self.y_step.valueChanged.connect(self.calc_res)#
        self.x_start.valueChanged.connect(self.calc_res)#
        self.y_start.valueChanged.connect(self.calc_res)#
        self.x_end.valueChanged.connect(self.calc_res)#
        self.y_end.valueChanged.connect(self.calc_res)#
        
        self.start.clicked.connect(self.generate_flyscan) # running fly scan 
        self.show()

    

    def generate_flyscan(self):

        from bluesky import RunEngine
        from ophyd.sim import det1, det2, motor1, motor2
        from bluesky.plans import count, scan, grid_scan

        RE = RunEngine({})

        from bluesky.callbacks.best_effort import BestEffortCallback
        bec = BestEffortCallback()

        # Send all metadata/data captured to the BestEffortCallback.
        RE.subscribe(bec)

        # Make plots update live while scans run.
        from bluesky.utils import install_kicker
        install_kicker()
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        

        mot1_s = self.x_start.value()
        mot1_e = self.x_end.value()
        mot1_steps = self.x_step.value() 
        mot2_s = self.y_start.value()
        mot2_e = self.y_end.value()
        mot2_steps = self.y_step.value()
        dwell_t = self.dwell.value()

        cal_res_x = (abs(mot1_s)+abs(mot1_e))/mot1_steps
        cal_res_y = (abs(mot2_s)+abs(mot2_e))/mot2_steps

        '''

        if cal_res_x != cal_res_y:

            confirm_res_mismatch = QMessageBox.question(self, 'Warning', "X Resolution is NOT same as Y, Continue?",
                                                        QMessageBox.Yes|QMessageBox.No, QMessageBox.No)

            if confirm_res_mismatch == QMessageBox.Yes:
                pass

        
        '''

        if self.radioButton_1d.isChecked():
            #self.Select_Motor_3.setEnabled(False)
            RE(scan([det2], motor1, mot1_s,mot1_e ,mot1_steps))
        else:             
            RE(grid_scan([det2], motor1, mot1_s,mot1_e ,mot1_steps, motor2, mot2_s,mot2_e, mot2_steps, False))
        plt.show()

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



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
