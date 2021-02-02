# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fly_gui2.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Select_Detector_Set = QtWidgets.QComboBox(self.centralwidget)
        self.Select_Detector_Set.setObjectName("Select_Detector_Set")
        self.horizontalLayout.addWidget(self.Select_Detector_Set)
        self.Select_Motor = QtWidgets.QComboBox(self.centralwidget)
        self.Select_Motor.setObjectName("Select_Motor")
        self.horizontalLayout.addWidget(self.Select_Motor)
        self.X_Start = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_Start.setPrefix("")
        self.X_Start.setMinimum(-15.0)
        self.X_Start.setMaximum(15.0)
        self.X_Start.setSingleStep(0.5)
        self.X_Start.setProperty("value", 15.0)
        self.X_Start.setObjectName("X_Start")
        self.horizontalLayout.addWidget(self.X_Start)
        self.X_end = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_end.setSuffix("um")
        self.X_end.setMinimum(-15.0)
        self.X_end.setMaximum(15.0)
        self.X_end.setSingleStep(0.5)
        self.X_end.setProperty("value", 15.0)
        self.X_end.setObjectName("X_end")
        self.horizontalLayout.addWidget(self.X_end)
        self.X_num = QtWidgets.QSpinBox(self.centralwidget)
        self.X_num.setMaximum(1000)
        self.X_num.setSingleStep(10)
        self.X_num.setProperty("value", 30)
        self.X_num.setObjectName("X_num")
        
        self.horizontalLayout.addWidget(self.X_num)
        
        self.Dwell_Time = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Dwell_Time.setPrefix("")
        self.Dwell_Time.setSuffix("sec")
        self.Dwell_Time.setMaximum(0.5)
        self.Dwell_Time.setSingleStep(0.01)
        self.Dwell_Time.setProperty("value", 0.05)
        self.Dwell_Time.setObjectName("Dwell_Time")
        self.Dwell_Time.valueChanged.connect(self.fly_details)
        self.X_num.valueChanged.connect(self.fly_details)
        
        self.horizontalLayout.addWidget(self.Dwell_Time)
        self.Run_Scan = QtWidgets.QPushButton(self.centralwidget)
        self.Run_Scan.setObjectName("Run_Scan")
        self.horizontalLayout.addWidget(self.Run_Scan)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
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
        self.X_Start.setSuffix(_translate("MainWindow", "um"))
        self.Run_Scan.setText(_translate("MainWindow", "Start"))

    def fly_details(self):
        print (f'Total time = {self.X_num.value()*self.Dwell_Time.value()}')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
