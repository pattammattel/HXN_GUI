# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fly_gui_edit.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox
from PyQt5.QtWidgets import QCalendarWidget, QFontDialog, QColorDialog, QTextEdit, QFileDialog
from PyQt5.QtWidgets import QCheckBox, QProgressBar, QComboBox, QLabel, QStyleFactory, QLineEdit, QInputDialog




class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(0,0,300, 200)
        self.setWindowTitle('HXN')
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        
        self.show()

class MyTableWidget(QWidget):
    
    def __init__(self, parent):
        super(QWidget,self).__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.tabs = QtWidgets.QTabWidget()
        self.scan = QWidget()
        self.diff = QWidget()
        self.xanes = QWidget()
        self.tomo = QWidget()
        self.alignement = QWidget()
        self.tabs.resize(300,200)
        
        self.tabs.addTab(self.scan, "Image Scan")
        self.tabs.addTab(self.diff, "Diffraction")
        self.tabs.addTab(self.xanes, "XANES")
        self.tabs.addTab(self.tomo, "Tomography")
        self.tabs.addTab(self.alignement, "Alignemnent (experts)")
        
        self.scan.layout = QtWidgets.QVBoxLayout(self.scan)
        self.pushButton1 = QPushButton("PyQt5 button")

        self.Select_Motor_4 = QtWidgets.QComboBox(self.scan)
        self.Select_Motor_4.addItem("zpssx")
        self.Select_Motor_4.setGeometry(QtCore.QRect(130, 90, 81, 22))

        
        self.scan.setLayout(self.scan.layout)
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        '''
        self.Select_Motor_4 = QtWidgets.QComboBox(self.scan)
        self.Select_Motor_4.setGeometry(QtCore.QRect(130, 90, 81, 22))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Select_Motor_4.sizePolicy().hasHeightForWidth())
        self.Select_Motor_4.setSizePolicy(sizePolicy)
        self.Select_Motor_4.setObjectName("Select_Motor_4")
        self.Select_Motor_4.addItem("zpssx")
        self.Select_Motor_4.addItem("zpssy")
        self.Select_Motor_4.addItem("zpssz")
        
        self.x_start = QtWidgets.QDoubleSpinBox(self.scan)
        self.x_start.setGeometry(QtCore.QRect(230, 90, 91, 20))
    
        
        self.x_start.setSizePolicy(sizePolicy)
        self.x_start.setSuffix("um")
        self.x_start.setMinimum(-15.0)
        self.x_start.setMaximum(15.0)
        self.x_start.setSingleStep(0.5)
        self.x_start.setProperty("value", -15.0)
        self.x_start.setObjectName("x_start")
    
        self.y_start = QtWidgets.QDoubleSpinBox(self.scan)
        self.y_start.setGeometry(QtCore.QRect(230, 150, 91, 20))

        self.y_start.setSizePolicy(sizePolicy)
        self.y_start.setSuffix("um")
        self.y_start.setMinimum(-15.0)
        self.y_start.setMaximum(15.0)
        self.y_start.setSingleStep(0.5)
        self.y_start.setProperty("value", -15.0)
        self.y_start.setObjectName("y_start")
        self.y_end = QtWidgets.QDoubleSpinBox(self.scan)
        self.y_end.setGeometry(QtCore.QRect(340, 150, 91, 20))
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.y_end.sizePolicy().hasHeightForWidth())
        
        self.y_end.setSizePolicy(sizePolicy)
        self.y_end.setSuffix("um")
        self.y_end.setMinimum(-15.0)
        self.y_end.setMaximum(15.0)
        self.y_end.setSingleStep(0.5)
        self.y_end.setProperty("value", 15.0)
        self.y_end.setObjectName("y_end")
        
        self.Select_Motor_3 = QtWidgets.QComboBox(self.scan)
        self.Select_Motor_3.setGeometry(QtCore.QRect(130, 150, 81, 22))
        
        self.Select_Motor_3.setSizePolicy(sizePolicy)
        self.Select_Motor_3.setObjectName("Select_Motor_3")
        self.Select_Motor_3.addItem("zpssx")
        self.Select_Motor_3.addItem("zpssy")
        self.Select_Motor_3.addItem("zpssz")
        self.Dets_2 = QtWidgets.QComboBox(self.scan)
        self.Dets_2.setGeometry(QtCore.QRect(31, 120, 71, 22))
 
        self.Dets_2.setSizePolicy(sizePolicy)
        self.Dets_2.setObjectName("Dets_2")
        self.Dets_2.addItem("dets1")
        self.Dets_2.addItem("dets2")
        self.Dets_2.addItem("dets3")
        self.label_8 = QtWidgets.QLabel(self.scan)
        self.label_8.setGeometry(QtCore.QRect(640, 130, 141, 16))

        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.tot_time = QtWidgets.QLineEdit(self.scan)
        self.tot_time.setGeometry(QtCore.QRect(630, 150, 154, 22))
        
        self.tot_time.setSizePolicy(sizePolicy)
        self.tot_time.setReadOnly(True)
        self.tot_time.setObjectName("tot_time")
        
        self.res = QtWidgets.QLineEdit(self.scan)
        self.res.setGeometry(QtCore.QRect(630, 90, 154, 22))
        self.res.setSizePolicy(sizePolicy)
        self.res.setReadOnly(True)
        self.res.setObjectName("res")
        
        self.start = QtWidgets.QPushButton(self.scan)
        self.start.setGeometry(QtCore.QRect(840, 90, 111, 61))
        self.start.setSizePolicy(sizePolicy)
        
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
		
        self.start.setFont(font)
        self.start.setStyleSheet("background-color: rgb(170, 255, 127);\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"")
        self.start.setObjectName("start")
        self.x_end = QtWidgets.QDoubleSpinBox(self.scan)
        self.x_end.setGeometry(QtCore.QRect(340, 90, 91, 20))
        
        self.x_end.setSizePolicy(sizePolicy)
        self.x_end.setSuffix("um")
        self.x_end.setMinimum(-15.0)
        self.x_end.setMaximum(15.0)
        self.x_end.setSingleStep(0.5)
        self.x_end.setProperty("value", 15.0)
        self.x_end.setObjectName("x_end")
		
        self.x_step = QtWidgets.QSpinBox(self.scan)
        self.x_step.setGeometry(QtCore.QRect(450, 90, 52, 20))
        self.x_step.setSizePolicy(sizePolicy)
        self.x_step.setMaximum(1000)
        self.x_step.setSingleStep(10)
        self.x_step.setProperty("value", 30)
        self.x_step.setObjectName("x_step")
        self.x_step.setMinimum(1)
        
        self.y_step = QtWidgets.QSpinBox(self.scan)
        self.y_step.setGeometry(QtCore.QRect(450, 150, 52, 20))
        self.y_step.setSizePolicy(sizePolicy)
        self.y_step.setMaximum(1000)
        self.y_step.setSingleStep(10)
        self.y_step.setProperty("value", 30)
        self.y_step.setObjectName("y_step")
        self.y_step.setMinimum(1)
        
        self.dwell = QtWidgets.QDoubleSpinBox(self.scan)
        self.dwell.setGeometry(QtCore.QRect(530, 120, 81, 20))
        self.dwell.setSizePolicy(sizePolicy)
        self.dwell.setPrefix("")
        self.dwell.setSuffix("sec")
        self.dwell.setMaximum(0.5)
        self.dwell.setSingleStep(0.01)
        self.dwell.setProperty("value", 0.05)
        self.dwell.setObjectName("dwell")
        
        self.label_4 = QtWidgets.QLabel(self.scan)
        self.label_4.setGeometry(QtCore.QRect(530, 100, 81, 16))
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setScaledContents(False)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label = QtWidgets.QLabel(self.scan)
        self.label.setGeometry(QtCore.QRect(140, 70, 40, 16))

        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_12 = QtWidgets.QLabel(self.scan)
        self.label_12.setGeometry(QtCore.QRect(240, 70, 51, 16))

        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName("label_12")
        self.label_16 = QtWidgets.QLabel(self.scan)
        self.label_16.setGeometry(QtCore.QRect(350, 70, 51, 16))
 
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setObjectName("label_16")
        self.label_3 = QtWidgets.QLabel(self.scan)
        self.label_3.setGeometry(QtCore.QRect(450, 70, 51, 16))

        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.disp_resln_label = QtWidgets.QLabel(self.scan)
        self.disp_resln_label.setGeometry(QtCore.QRect(640, 70, 131, 16))

        self.disp_resln_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.disp_resln_label.setFont(font)
        self.disp_resln_label.setObjectName("disp_resln_label")
        self.label_6 = QtWidgets.QLabel(self.scan)
        self.label_6.setGeometry(QtCore.QRect(30, 90, 71, 20))

        self.label_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.radioButton_1d = QtWidgets.QRadioButton(self.scan)
        self.radioButton_1d.setGeometry(QtCore.QRect(40, 20, 95, 20))

        self.radioButton_1d.setSizePolicy(sizePolicy)
        self.radioButton_1d.setStyleSheet("font: 75 12pt \"Arial\";")
        self.radioButton_1d.setObjectName("radioButton_1d")
        self.radioButton = QtWidgets.QRadioButton(self.scan)
        self.radioButton.setGeometry(QtCore.QRect(140, 20, 95, 20))

        self.radioButton.setSizePolicy(sizePolicy)
        self.radioButton.setStyleSheet("font: 75 12pt \"Arial\";\n"
"border-color: rgb(255, 235, 119);\n"
"")
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.groupBox_2 = QtWidgets.QGroupBox(self.scan)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 220, 991, 241))

        self.groupBox_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox_2.setStyleSheet("font: 10pt \"Arial\";")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox_2)

        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_2.addWidget(self.checkBox, 0, 4, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox_2)

        self.lineEdit_6.setSizePolicy(sizePolicy)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_2.addWidget(self.lineEdit_6, 0, 3, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)

        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Plot")
        self.comboBox.addItem("erf_fit")
        self.comboBox.addItem("return_line_center")
        self.gridLayout_2.addWidget(self.comboBox, 0, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox_2)

        self.lineEdit_5.setSizePolicy(sizePolicy)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_2.addWidget(self.lineEdit_5, 0, 2, 1, 1)
        
        self.menubar = QtWidgets.QMenuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1074, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar()
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        #self.retranslateUi(HXN_GUI_2)
        self.tabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)


        self.label_8.setText( "Total Aq.Time (min)")
        self.tot_time.setText("1")
        self.res.setText( "1000")
        self.start.setText( "Start")
        self.label_4.setText( "Dwell Time")
        self.label.setText( "Motors")
        self.label_12.setText( "Start")
        self.label_16.setText( "End")
        self.label_3.setText( "Steps")
        self.disp_resln_label.setText( "Resolution (nm)")
        self.label_6.setText( "Detectors")
        self.radioButton_1d.setText( "Fly1D")
        self.radioButton.setText( "Fly2D")
        self.checkBox.setText( "Normlization")
        self.lineEdit_6.setText( "Element")
        self.lineEdit_5.setText( "Scan ID")
	   
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
        
    #Functions are below
    
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

        

        if cal_res_x != cal_res_y:

            confirm_res_mismatch = QMessageBox.question(self, 'Warning', "X Resolution is NOT same as Y, Continue?",
                                                        QMessageBox.Yes|QMessageBox.No, QMessageBox.No)

            if confirm_res_mismatch == QMessageBox.Yes:
                pass

        
        

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
        '''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
