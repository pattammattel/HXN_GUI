# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fly_gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.X = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X.setGeometry(QtCore.QRect(160, 130, 62, 22))
        self.X.setObjectName("X")
        self.X_num = QtWidgets.QSpinBox(self.centralwidget)
        self.X_num.setGeometry(QtCore.QRect(320, 130, 42, 22))
        self.X_num.setObjectName("X_num")
        self.Dwell_Time = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Dwell_Time.setGeometry(QtCore.QRect(390, 130, 62, 22))
        self.Dwell_Time.setObjectName("Dwell_Time")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 90, 21, 16))
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(310, 90, 55, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(390, 90, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(240, 90, 41, 16))
        self.label_6.setObjectName("label_6")
        self.X_end = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_end.setGeometry(QtCore.QRect(240, 130, 62, 22))
        self.X_end.setObjectName("X_end")
        self.Run_Scan = QtWidgets.QPushButton(self.centralwidget)
        self.Run_Scan.setGeometry(QtCore.QRect(490, 130, 93, 28))
        self.Run_Scan.setObjectName("Run_Scan")
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
        self.label.setText(_translate("MainWindow", "X"))
        self.label_3.setText(_translate("MainWindow", "X_Steps"))
        self.label_4.setText(_translate("MainWindow", "Dwell_Time"))
        self.label_6.setText(_translate("MainWindow", "X_end"))
        self.Run_Scan.setText(_translate("MainWindow", "Run"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
