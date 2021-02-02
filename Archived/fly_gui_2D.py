# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fly_gui_2D.ui'
#
# Created by Ajith Pattammattel using PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(893, 625)
        MainWindow.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ZP_FlyGui = QtWidgets.QTabWidget(self.centralwidget)
        self.ZP_FlyGui.setGeometry(QtCore.QRect(11, 11, 741, 351))
        self.ZP_FlyGui.setObjectName("ZP_FlyGui")
        self.fly2d = QtWidgets.QWidget()
        self.fly2d.setObjectName("fly2d")
        
        self.X_Start = QtWidgets.QDoubleSpinBox(self.fly2d)
        self.X_Start.setGeometry(QtCore.QRect(240, 60, 66, 20))
        self.X_Start.setPrefix("")
        self.X_Start.setMinimum(-15.0)
        self.X_Start.setMaximum(15.0)
        self.X_Start.setSingleStep(0.5)
        self.X_Start.setProperty("value", -15.0)
        self.X_Start.setObjectName("X_Start")
        
        self.X_end = QtWidgets.QDoubleSpinBox(self.fly2d)
        self.X_end.setGeometry(QtCore.QRect(350, 60, 66, 20))
        self.X_end.setSuffix("um")
        self.X_end.setMinimum(-15.0)
        self.X_end.setMaximum(15.0)
        self.X_end.setSingleStep(0.5)
        self.X_end.setProperty("value", 15.0)
        self.X_end.setObjectName("X_end")
        
        self.mot1_num_steps = QtWidgets.QSpinBox(self.fly2d)
        self.mot1_num_steps.setGeometry(QtCore.QRect(490, 60, 45, 20))
        self.mot1_num_steps.setMaximum(1000)
        self.mot1_num_steps.setSingleStep(10)
        self.mot1_num_steps.setProperty("value", 30)
        self.mot1_num_steps.setObjectName("mot1_num_steps")
        
        self.Dwell_Time = QtWidgets.QDoubleSpinBox(self.fly2d)
        self.Dwell_Time.setGeometry(QtCore.QRect(630, 90, 57, 20))
        self.Dwell_Time.setPrefix("")
        self.Dwell_Time.setSuffix("sec")
        self.Dwell_Time.setMaximum(0.5)
        self.Dwell_Time.setSingleStep(0.01)
        self.Dwell_Time.setProperty("value", 0.05)
        self.Dwell_Time.setObjectName("Dwell_Time")
        
        self.mot2_num_steps = QtWidgets.QSpinBox(self.fly2d)
        self.mot2_num_steps.setGeometry(QtCore.QRect(490, 130, 45, 20))
        self.mot2_num_steps.setMaximum(1000)
        self.mot2_num_steps.setSingleStep(10)
        self.mot2_num_steps.setProperty("value", 30)
        self.mot2_num_steps.setObjectName("mot2_num_steps")
        
        self.Y_Mots = QtWidgets.QComboBox(self.fly2d)
        self.Y_Mots.setGeometry(QtCore.QRect(120, 130, 66, 20))
        self.Y_Mots.setObjectName("Y_Mots")
        self.Y_Mots.addItem("")
        self.Y_Mots.addItem("")
        self.Y_Mots.addItem("")
        
        self.Y_end = QtWidgets.QDoubleSpinBox(self.fly2d)
        self.Y_end.setGeometry(QtCore.QRect(350, 130, 66, 20))
        self.Y_end.setSuffix("um")
        self.Y_end.setMinimum(-15.0)
        self.Y_end.setMaximum(15.0)
        self.Y_end.setSingleStep(0.5)
        self.Y_end.setProperty("value", 15.0)
        self.Y_end.setObjectName("Y_end")
        
        self.Y_Start = QtWidgets.QDoubleSpinBox(self.fly2d)
        self.Y_Start.setGeometry(QtCore.QRect(240, 130, 66, 20))
        self.Y_Start.setPrefix("")
        self.Y_Start.setMinimum(-15.0)
        self.Y_Start.setMaximum(15.0)
        self.Y_Start.setSingleStep(0.5)
        self.Y_Start.setProperty("value", -15.0)
        self.Y_Start.setObjectName("Y_Start")
        
        self.H_Mots = QtWidgets.QComboBox(self.fly2d)
        self.H_Mots.setGeometry(QtCore.QRect(120, 60, 61, 20))
        self.H_Mots.setObjectName("H_Mots")
        self.H_Mots.addItem("zpssx")
        self.H_Mots.addItem("zpssy")
        self.H_Mots.addItem("zpssz")
        
        self.Dets = QtWidgets.QComboBox(self.fly2d)
        self.Dets.setGeometry(QtCore.QRect(30, 100, 53, 20))
        self.Dets.setObjectName("Dets")
        self.Dets.addItem("")
        self.Dets.addItem("")
        self.Dets.addItem("")
        
        self.Start = QtWidgets.QPushButton(self.fly2d)
        self.Start.setGeometry(QtCore.QRect(200, 280, 93, 28))
        self.Start.setObjectName("Start")
        
        
        self.Stop = QtWidgets.QPushButton(self.fly2d)
        self.Stop.setGeometry(QtCore.QRect(330, 280, 93, 28))
        self.Stop.setObjectName("Stop")
        
        self.Motors_label = QtWidgets.QLabel(self.fly2d)
        self.Motors_label.setGeometry(QtCore.QRect(138, 11, 31, 16))
        self.Motors_label.setObjectName("Motors_label")
        self.Scan_label = QtWidgets.QLabel(self.fly2d)
        self.Scan_label.setGeometry(QtCore.QRect(300, 10, 53, 16))
        self.Scan_label.setTextFormat(QtCore.Qt.AutoText)
        self.Scan_label.setScaledContents(False)
        self.Scan_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Scan_label.setObjectName("Scan_label")
        
        self.Steps_label = QtWidgets.QLabel(self.fly2d)
        self.Steps_label.setGeometry(QtCore.QRect(519, 11, 24, 16))
        self.Steps_label.setTextFormat(QtCore.Qt.AutoText)
        self.Steps_label.setScaledContents(False)
        self.Steps_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Steps_label.setObjectName("Steps_label")
        
        self.Dwell_label = QtWidgets.QLabel(self.fly2d)
        self.Dwell_label.setGeometry(QtCore.QRect(646, 11, 100, 16))
        self.Dwell_label.setTextFormat(QtCore.Qt.AutoText)
        self.Dwell_label.setScaledContents(False)
        self.Dwell_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Dwell_label.setObjectName("Dwell_label")
        self.Dwell_label.setFont(QtGui.QFont('Arial', 10))
        
        self.ScanStart_label = QtWidgets.QLabel(self.fly2d)
        self.ScanStart_label.setGeometry(QtCore.QRect(250, 40, 51, 16))
        self.ScanStart_label.setTextFormat(QtCore.Qt.AutoText)
        self.ScanStart_label.setScaledContents(False)
        self.ScanStart_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ScanStart_label.setObjectName("ScanStart_label")
        
        self.Scan_end_label = QtWidgets.QLabel(self.fly2d)
        self.Scan_end_label.setGeometry(QtCore.QRect(360, 40, 51, 16))
        self.Scan_end_label.setTextFormat(QtCore.Qt.AutoText)
        self.Scan_end_label.setScaledContents(False)
        self.Scan_end_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Scan_end_label.setObjectName("Scan_end_label")
        
        self.calc_Res = QtWidgets.QLabel(self.fly2d)
        self.calc_Res.setGeometry(QtCore.QRect(220, 180, 80, 16))
        self.calc_Res.setObjectName("calc_Res")
        self.calc_Time = QtWidgets.QLabel(self.fly2d)
        self.calc_Time.setGeometry(QtCore.QRect(210, 220, 71, 31))
        self.calc_Time.setObjectName("calc_Time")
        
        self.Disc_Calc_Res = QtWidgets.QLineEdit(self.fly2d)
        self.Disc_Calc_Res.setGeometry(QtCore.QRect(290, 180, 113, 22))
        self.Disc_Calc_Res.setReadOnly(False)
        self.Disc_Calc_Res.setObjectName("Disc_Calc_Res")
        self.Disc_Calc_Res.setText('1000')
        
        self.Dis_Calc_time = QtWidgets.QLineEdit(self.fly2d)
        self.Dis_Calc_time.setGeometry(QtCore.QRect(290, 220, 113, 22))
        self.Dis_Calc_time.setReadOnly(True)
        self.Dis_Calc_time.setObjectName("Dis_Calc_time")
        self.Dis_Calc_time.setText('10')

        #All the connections done here 
        self.Dwell_Time.valueChanged.connect(self.calc_res)
        self.mot2_num_steps.valueChanged.connect(self.calc_res)
        self.mot1_num_steps.valueChanged.connect(self.calc_res)
        self.Start.clicked.connect(self.generate_flyscan)
        
        
        self.ZP_FlyGui.addTab(self.fly2d, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 893, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.ZP_FlyGui.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.X_Start.setSuffix(_translate("MainWindow", "um"))
        self.Y_Mots.setItemText(0, _translate("MainWindow", "zpssy"))
        self.Y_Mots.setItemText(1, _translate("MainWindow", "zpssx"))
        self.Y_Mots.setItemText(2, _translate("MainWindow", "zpssz"))
        self.Y_Start.setSuffix(_translate("MainWindow", "um"))
        self.Dets.setItemText(0, _translate("MainWindow", "dets1"))
        self.Dets.setItemText(1, _translate("MainWindow", "dets2"))
        self.Dets.setItemText(2, _translate("MainWindow", "dets3"))
        self.Start.setText(_translate("MainWindow", "Start"))
        self.Stop.setText(_translate("MainWindow", "Request Stop"))
        self.Motors_label.setText(_translate("MainWindow", "Motors"))
        self.Scan_label.setText(_translate("MainWindow", "Scan Range"))
        self.Steps_label.setText(_translate("MainWindow", "Steps"))
        self.Dwell_label.setText(_translate("MainWindow", "Dwell Time"))
        self.ScanStart_label.setText(_translate("MainWindow", "Start_Pos"))
        self.Scan_end_label.setText(_translate("MainWindow", "End_Pos"))
        self.calc_Res.setText(_translate("MainWindow", "Resolution (nm)"))
        self.calc_Time.setText(_translate("MainWindow", "Total Aq.Time (~minutes)"))
        self.ZP_FlyGui.setTabText(self.ZP_FlyGui.indexOf(self.fly2d), _translate("MainWindow", "2D Scan"))

    def generate_flyscan(self):

        from bluesky import RunEngine
        from ophyd.sim import det, motor1, motor2
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
        

        mot1_s = self.X_Start.value()
        mot1_e = self.X_end.value()
        mot1_steps = self.mot1_num_steps.value() 
        mot2_s = self.Y_Start.value()
        mot2_e = self.Y_end.value()
        mot2_steps = self.mot2_num_steps.value()
        dwell = self.Dwell_Time.value()
        plt.figure()

        RE(grid_scan([det], motor1, mot1_s,mot1_e ,mot1_steps, motor2, mot2_s,mot2_e, mot2_steps, False))
        plt.show()

    def calc_res(self):
        mot1_s = self.X_Start.value()
        mot1_e = self.X_end.value()
        mot1_steps = self.mot1_num_steps.value()
        mot2_steps = self.mot2_num_steps.value()
        dwell = self.Dwell_Time.value()
        
        res = (abs(mot1_s)+abs(mot1_e))/mot1_steps
        self.Disc_Calc_Res.setText(str(res*1000))

        tot_time = str(mot1_steps*mot2_steps*dwell/60)
        self.Dis_Calc_time.setText(tot_time)

        

        
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
