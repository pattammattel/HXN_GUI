# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fly_gui2.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from bluesky import RunEngine

RE = RunEngine({})

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        
        self.Run_Scan = QtWidgets.QPushButton(self.centralwidget)
        self.Run_Scan.setObjectName("Run_Scan")
        self.gridLayout.addWidget(self.Run_Scan, 0, 4, 1, 1)
        self.Run_Scan.clicked.connect(self.generate_flyscan)
        
        self.X_start = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_start.setObjectName("X_start")
        self.X_start.setMinimum(-15)
        self.X_start.setMaximum(15)
        self.gridLayout.addWidget(self.X_start, 0, 0, 1, 1)
        
        self.X_end = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_end.setObjectName("X_end")
        self.X_end.setMinimum(-15)
        self.X_end.setMaximum(15)
        self.gridLayout.addWidget(self.X_end, 0, 1, 1, 1)
        
        self.X_num = QtWidgets.QSpinBox(self.centralwidget)
        self.X_num.setObjectName("X_num")
        self.X_num.setMinimum(0)
        self.X_num.setMaximum(1000)
        self.X_num.setSingleStep(15)
        self.gridLayout.addWidget(self.X_num, 0, 2, 1, 1)
        self.X_num.valueChanged.connect(self.fly_details)

        self.Dwell_Time = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Dwell_Time.setObjectName("Dwell_Time")
        self.Dwell_Time.setMaximum(0.5)
        self.Dwell_Time.setSingleStep(0.05)
        self.gridLayout.addWidget(self.Dwell_Time, 0, 3, 1, 1)
        self.Dwell_Time.valueChanged.connect(self.fly_details)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Run_Scan.setText(_translate("MainWindow", "Run"))
        
    def fly_details(self):
        print (f'Total time = {self.X_num.value()*self.Dwell_Time.value()}')

    def generate_flyscan(self):
        #print (f'yield from fly1d{self.X_start.value(),self.X_end.value(),self.X_num.value(), self.Dwell_Time.value()}')
		np.linspace(self.X_start.value(),self.X_end.value(),self.X_num.value())
		
		
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
