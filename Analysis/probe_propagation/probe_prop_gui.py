
import sys
import os
import json
import collections
import ast
import h5py
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import tifffile as tf
import time
from scipy.ndimage.measurements import center_of_mass
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDesktopWidget, QApplication, QSizePolicy
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QDate

from probe_propagation_calcs import  *
ui_path = os.path.dirname(os.path.abspath(__file__))



class ProbePropagationGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(ProbePropagationGUI, self).__init__()
        uic.loadUi(os.path.join(ui_path,'probe_prop_gui.ui'), self)

        self.sigma_selection_line = pg.InfiniteLine()

        self.image_view_probe.ui.menuBtn.hide()
        self.image_view_xhue.ui.menuBtn.hide()
        self.image_view_yhue.ui.menuBtn.hide()
        self.image_view_probe.ui.roiBtn.hide()
        self.image_view_xhue.ui.roiBtn.hide()
        self.image_view_yhue.ui.roiBtn.hide()

        self.pb_plot.clicked.connect(lambda:self.propagate_and_plot(self.probe_file))

        self.pb_load_probe.clicked.connect(self.load_probe)
        self.slider_for_index.valueChanged.connect(lambda:
                                                   self.update_with_slider(
                                                       np.abs(
                                                           self.prb_array.transpose(2, 0, 1)))
                                                   )



    def load_probe(self):

        filename = QFileDialog().getOpenFileName(self, "Select Probe", "", "npy file(*npy )")

        if filename[0]:
            self.probe_file = filename[0]

            #image here
            self.image_view_probe.setImage(np.abs(np.load(filename[0])))
            self.image_view_probe.setPredefinedGradient("viridis")

        else:
            return

    def propagate_and_plot(self, probe):

        self.pbar_propagation.setRange(0,0)

        self.prb_array, \
        self.sigma, \
        self.xfit, \
        self.yfit = propagate_probe(probe,
                                    det_distance_m = self.dsb_det_dist.value(),
                                    energy = self.dsb_energy.value(),
                                    det_pixel_size = self.dsb_det_pixel_size.value(),
                                    start_um = self.dsb_prop_start.value(),
                                    end_um = self.dsb_prop_end.value(),
                                    step_size_um = self.dsb_prop_size.value())

        nx,ny = np.shape(np.abs(np.load(probe)))

        self.pixel_size,_ = calculate_res_and_dof(self.dsb_energy.value(),
                                                  self.dsb_det_dist.value(),
                                                  self.dsb_det_pixel_size.value(),
                                                  nx)
        print(self.pixel_size)
        self.dsb_calc_pixel_size.setValue(self.pixel_size*1.e9)

        self.imshow_probe_stack(np.abs(self.prb_array.transpose(2,0,1)))
        self.plot_hue(np.abs(self.prb_array.transpose(2, 0, 1)))
        self.plot_sigma(self.sigma)
        self.pbar_propagation.setRange(0, 100)
        self.pbar_propagation.setValue(100)

    def imshow_probe_stack(self, im_stack):

        """im_stack expected to have shape (z, x,y); x,y are image axes"""

        #set the slider max with z dim of the image
        self.slider_for_index.setMaximum(np.shape(im_stack)[0]-1)
        self.slider_for_index.setMinimum(0)
        self.image_view_probe.setPredefinedGradient("viridis")

        # to show middle stack
        self.slider_for_index.setValue((np.shape(im_stack)[0]//2))
        self.update_with_slider(im_stack)

    def plot_linefit(self, data_and_fit, plotter, index = 0):

        """data_and_fit expected to have three columns;
         x-values,
         line profile (y1),
          and guassican fit (y2)"""

        plotter.plot(data_and_fit[index,0,:]/np.max(data_and_fit[index,0,:]),
                     pen=pg.mkPen(pg.mkColor( 0, 0, 255)),
                     symbol="o",
                     symbolSize=5,
                     symbolBrush="y",
                     clear = False)

        plotter.plot(data_and_fit[index,1,:],
                     pen=pg.mkPen(pg.mkColor( 255, 0, 0)))


    def update_with_slider(self,im_stack):

        self.plot_xfit.clear()
        self.plot_yfit.clear()
        self.image_view_probe.setImage(im_stack[self.slider_for_index.value()])
        self.plot_linefit(self.xfit, self.plot_xfit, self.slider_for_index.value())
        self.plot_linefit(self.yfit, self.plot_yfit, self.slider_for_index.value())
        self.slider_label.setText(f'{self.slider_for_index.value()} of {self.slider_for_index.maximum()}')

        # calculate pixel size late, setting to 7nm
        self.lbl_xfit.setText(f' FWHM = {self.sigma[1,self.slider_for_index.value()]*2.35482*self.pixel_size*1.e9 :.2f} nm,'
                              f'Distance  = {self.sigma[0,self.slider_for_index.value()] :.2f}' )

        self.lbl_yfit.setText(f' FWHM = {self.sigma[2, self.slider_for_index.value()] * 2.35482*self.pixel_size*1.e9 :.2f} nm,'
                              f'Distance  = {self.sigma[0, self.slider_for_index.value()] :.2f}')

        self.sigma_selection_line.setPos(self.sigma[0][self.slider_for_index.value()])
        
    def plot_hue(self, im_stack):

        x_hue = np.squeeze(im_stack.mean(1))
        self.image_view_xhue.setImage(x_hue)

        y_hue = np.squeeze(im_stack.mean(2))
        self.image_view_yhue.setImage(y_hue)

        self.image_view_xhue.setPredefinedGradient("viridis")
        self.image_view_yhue.setPredefinedGradient("viridis")

    def plot_sigma(self, sigma):
        self.sigma_plot.clear()
        self.sigma_plot.addLegend()
        self.sigma_plot.setLabel("bottom", "Distance")
        self.sigma_plot.setLabel("left", "Sigma", "A.U.")
        self.x_fwhm = sigma[1]*2.35482*self.pixel_size*1.e9
        self.y_fwhm = sigma[2]*2.35482*self.pixel_size*1.e9

        self.sigma_plot.plot(sigma[0], self.x_fwhm,
                             pen='c',
                             symbol="o",
                             symbolSize=4,
                             symbolBrush="y",
                             clear=False,
                             name = 'X')

        self.sigma_plot.plot(sigma[0], self.y_fwhm,
                             pen='m',
                             symbol="x",
                             symbolSize=4,
                             symbolBrush="c",
                             clear=False,
                             name = 'Y')

        min_index = np.argwhere(sigma[2] == np.min(sigma[2]))

        #print(min_index[0][0])

        self.sigma_selection_line = pg.InfiniteLine(pos=sigma[0][min_index[0][0]],
                                             angle=90,
                                             pen=pg.mkPen("m", width=2.5),
                                             movable=True,
                                             bounds=None,
                                             )

        self.sigma_plot.addItem(self.sigma_selection_line)

    def move_sigma_line(self):

        pos_x, pos_y = self.sigma_selection_line.pos()
        x_fwhm_ = self.x_fwhm[self.sigma[0] == np.floor(pos_x)]
        print(x_fwhm_)
        self.line_label.setText(str(x_fwhm_))




if __name__ == '__main__':
    try:
        QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except:
        pass
    app = QtWidgets.QApplication(sys.argv)
    w = ProbePropagationGUI()
    w.show()
    sys.exit(app.exec_())