import fnmatch
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
import configparser
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QSizePolicy, QErrorMessage

from probe_propagation_calcs import  *
ui_path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()


class ProbePropagationGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(ProbePropagationGUI, self).__init__()
        uic.loadUi(os.path.join(ui_path,'probe_prop_gui.ui'), self)

        self.sigma_selection_line = pg.InfiniteLine()
        self.deviation_selection_line = pg.InfiniteLine()
        self.sigma_xhue_line = pg.InfiniteLine()
        self.sigma_yhue_line = pg.InfiniteLine()

        self.image_view_probe.ui.menuBtn.hide()
        self.image_view_probe_pha.ui.menuBtn.hide()
        self.image_view_xhue.ui.menuBtn.hide()
        self.image_view_yhue.ui.menuBtn.hide()
        self.image_view_probe.ui.roiBtn.hide()
        self.image_view_probe_pha.ui.roiBtn.hide()
        self.image_view_xhue.ui.roiBtn.hide()
        self.image_view_yhue.ui.roiBtn.hide()

        self.pb_plot.clicked.connect(lambda:self.propagate_and_plot(self.probe_file))
        self.ck_pixel_size.clicked.connect(self.toggle_pixel_size)

        self.pb_load_probe.clicked.connect(self.load_probe)
        self.slider_for_index.valueChanged.connect(lambda:
                                                   self.update_with_slider(
                                                        self.prb_array.transpose(2, 0, 1))
                                                   )
        
        self.actionSave_All_Plotted_Data.triggered.connect(self.export_current_data)
        self.actionExport_Parameters.triggered.connect(self.export_params)
    
    def parse_ptycho_txtfile(self, txt_filename):
    
        config.read(txt_filename)

        energy = config.getfloat('GUI','xray_energy_kev',fallback=12.0)
        det_dist = config.getfloat('GUI','z_m',fallback=2.05)
        det_pixel_size = config.getfloat('GUI','ccd_pixel_um', fallback=75)

        return energy,det_dist,det_pixel_size


    def toggle_pixel_size(self):
        if self.ck_pixel_size.isChecked():
            self.dsb_calc_pixel_size.setEnabled(True)
            self.dsb_det_dist.setEnabled(False)
            self.dsb_det_pixel_size.setEnabled(False)
        else:
            self.dsb_calc_pixel_size.setEnabled(False)
            self.dsb_det_dist.setEnabled(True)
            self.dsb_det_pixel_size.setEnabled(True)



    def load_probe(self):

        filename = QFileDialog().getOpenFileName(self, "Select Probe", "", "npy file(*npy )")

        if filename[0]:
            self.probe_file = filename[0]
            self.le_probe_file.setText(self.probe_file)

            #image here
            self.image_view_probe.setImage(np.abs(np.load(filename[0])))
            self.image_view_probe_pha.setImage(np.angle(np.load(filename[0])))
            self.image_view_probe.setPredefinedGradient("viridis")
            self.image_view_probe_pha.setPredefinedGradient("viridis")

            #if self.ck_pixel_size.isChecked():
            #    self.pixel_size = self.dsb_calc_pixel_size.value()*1e-9
            #else:
            folder = os.path.dirname(filename[0])

            for file in os.listdir(folder):
                #print(file)
                if fnmatch.fnmatch(file, "*ptycho*"):
                    txtfile = file
                    print(txtfile)

            energy,\
            det_dist,\
            det_pixel_size = self.parse_ptycho_txtfile(os.path.join(folder,txtfile))

            shape = np.shape(np.abs(np.load(self.probe_file)))
            if len(shape) == 2:
                nx = shape[0]
                ny = shape[1]
            else:
                nx = shape[1]
                ny = shape[2]

            #nx, ny = np.shape(np.abs(np.load(self.probe_file)))
            self.dsb_energy.setValue(energy)
            self.dsb_det_dist.setValue(det_dist)
            self.dsb_det_pixel_size.setValue(det_pixel_size)
            self.pixel_size, _ = calculate_res_and_dof(self.dsb_energy.value(),
                                                    self.dsb_det_dist.value(),
                                                    self.dsb_det_pixel_size.value(),
                                                    nx)
            # print(self.pixel_size)
            self.dsb_calc_pixel_size.setValue(self.pixel_size * 1.e9)

                #except Exception as e:
                    #print(e)
                    #err_msg = QErrorMessage(self)
                    #err_msg.setWindowTitle("FileNotFoundError")
                    #err_msg.showMessage(f"Unable to find .txt file with recon parametrs"
                    #                    f"<br/> {e} "
                    #                   f"<br/> Manually enter energy,det_dist,det_pixel_size")
                    #pass

        else:
            return

    def propagate_and_plot(self, probe):
        #try:
            self.pbar_propagation.setRange(0,0)

            probe_array = np.load(probe)
            if probe_array.ndim > 2:
                probe_array = np.mean(probe_array,axis=0)

            nx,ny = np.shape(np.abs(probe_array))

            if self.ck_pixel_size.isChecked():
                self.pixel_size = self.dsb_calc_pixel_size.value()*1e-9
                nx_size_m = ny_size_m = self.pixel_size
            else:
            # pixel size , calculation is ;
                nx_size_m,_ = calculate_res_and_dof(self.dsb_energy.value(),
                                        self.dsb_det_dist.value(),
                                        self.dsb_det_pixel_size.value(),
                                        nx)
                ny_size_m,_ = calculate_res_and_dof(self.dsb_energy.value(),
                                        self.dsb_det_dist.value(),
                                        self.dsb_det_pixel_size.value(),
                                        ny)
                self.pixel_size = nx_size_m

            self.prb_array, \
            self.sigma, \
            self.deviation, \
            self.xfit, \
            self.yfit = propagate_probe(probe_array,
                                        self.dsb_energy.value(),
                                        nx_size_m, ny_size_m,
                                        start_um = self.dsb_prop_start.value(),
                                        end_um = self.dsb_prop_end.value(),
                                        step_size_um = self.dsb_prop_size.value())

            #print(self.pixel_size)
            self.dsb_calc_pixel_size.setValue(self.pixel_size*1.e9)
            self.imshow_probe_stack(self.prb_array.transpose(2,0,1))
            self.plot_hue(np.abs(self.prb_array.transpose(2, 0, 1)))
            self.plot_sigma(self.sigma)
            self.plot_deviation(self.deviation)
            self.pbar_propagation.setRange(0, 100)
            self.pbar_propagation.setValue(100)


    def imshow_probe_stack(self, im_stack):

        """im_stack expected to have shape (z, x,y); x,y are image axes"""

        #set the slider max with z dim of the image
        self.slider_for_index.setMaximum(np.shape(im_stack)[0]-1)
        self.slider_for_index.setMinimum(0)
        self.image_view_probe.setPredefinedGradient("viridis")
        self.image_view_probe_pha.setPredefinedGradient("viridis")

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
        
        def phase_unwrap(array):
            if np.max(array)-np.min(array) > 5:
                array[array<0] += 2*np.pi
                return array
            else:
                return array

        self.plot_xfit.clear()
        self.plot_yfit.clear()
        self.image_view_probe.setImage(np.abs(im_stack[self.slider_for_index.value()]))
        self.image_view_probe_pha.setImage(phase_unwrap(np.angle(im_stack[self.slider_for_index.value()])))
        self.plot_linefit(self.xfit, self.plot_xfit, self.slider_for_index.value())
        self.plot_linefit(self.yfit, self.plot_yfit, self.slider_for_index.value())
        self.slider_label.setText(f'{self.slider_for_index.value()} of {self.slider_for_index.maximum()}')

        self.prop_distance = self.sigma[0,self.slider_for_index.value()]
        self.xfwhm = self.sigma[1,self.slider_for_index.value()]*2.35482*self.pixel_size*1.e9
        self.yfwhm = self.sigma[2, self.slider_for_index.value()] * 2.35482*self.pixel_size*1.e9
        # calculate pixel size late, setting to 7nm
        self.lbl_xfit.setText(f' FWHM = { self.xfwhm :.2f} nm,'
                              f'Distance  = {self.prop_distance :.2f}' )

        self.lbl_yfit.setText(f' FWHM = {self.yfwhm :.2f} nm,'
                              f'Distance  = {self.prop_distance :.2f}')

        self.sigma_selection_line.setPos(self.sigma[0][self.slider_for_index.value()])
        self.deviation_selection_line.setPos(self.deviation[0][self.slider_for_index.value()])
        self.sigma_xhue_line.setPos(self.slider_for_index.value())
        self.sigma_yhue_line.setPos(self.slider_for_index.value())


        
    def plot_hue(self, im_stack):

        try:
            
            self.image_view_xhue.removeItem(self.sigma_xhue_line)
            self.image_view_yhue.removeItem(self.sigma_yhue_line)

        except:
            pass

        self.x_hue = np.squeeze(im_stack.mean(2))
        self.image_view_xhue.setImage(self.x_hue)

        self.y_hue = np.squeeze(im_stack.mean(1))
        self.image_view_yhue.setImage(self.y_hue)

        self.image_view_xhue.setPredefinedGradient("bipolar")
        self.image_view_yhue.setPredefinedGradient("bipolar")

        self.sigma_xhue_line = pg.InfiniteLine(pos=self.slider_for_index.value(),
                                             angle=90,
                                             pen=pg.mkPen("y", width=2),
                                             movable=True,
                                             bounds=None,
                                             )

        self.sigma_yhue_line = pg.InfiniteLine(pos=self.slider_for_index.value(),
                                             angle=90,
                                             pen=pg.mkPen("y", width=2),
                                             movable=True,
                                             bounds=None,
                                             )

        self.image_view_xhue.addItem(self.sigma_xhue_line)
        self.image_view_yhue.addItem(self.sigma_yhue_line)

    def plot_sigma(self, sigma):
        self.sigma_plot.clear()
        self.sigma_plot.addLegend()
        self.sigma_plot.setLabel("bottom", "Distance")
        self.sigma_plot.setLabel("left", "Sigma", "A.U.")
        self.x_fwhm = sigma[1]*2.35482*self.pixel_size*1.e9
        self.y_fwhm = sigma[2]*2.35482*self.pixel_size*1.e9

        self.sigma_plot.plot(sigma[0], self.y_fwhm,
                             pen='y',
                             symbol="o",
                             symbolSize=4,
                             symbolBrush="y",
                             clear=False,
                             name = 'Y')

        self.sigma_plot.plot(sigma[0], self.x_fwhm,
                             pen='c',
                             symbol="x",
                             symbolSize=4,
                             symbolBrush="c",
                             clear=False,
                             name = 'X')

        min_index = np.argwhere(sigma[2] == np.min(sigma[2]))

        #print(min_index[0][0])

        self.sigma_selection_line = pg.InfiniteLine(pos=sigma[0][min_index[0][0]],
                                             angle=90,
                                             pen=pg.mkPen("m", width=2),
                                             movable=True,
                                             bounds=None,
                                             )

        self.sigma_plot.addItem(self.sigma_selection_line)

    def plot_deviation(self, deviation):
            self.deviation_plot.clear()
            self.deviation_plot.addLegend()
            self.deviation_plot.setLabel("bottom", "Distance")
            self.deviation_plot.setLabel("left", "Deviation", "Normalized")

            dev_pha = (deviation[1]-np.min(deviation[1]))/((np.max(deviation[1])-np.min(deviation[1])))
            dev_amp = (deviation[2]-np.min(deviation[2]))/((np.max(deviation[2])-np.min(deviation[2])))

            self.deviation_plot.plot(deviation[0], dev_pha,
                                pen='c',
                                symbol="o",
                                symbolSize=4,
                                symbolBrush="y",
                                clear=False,
                                name = 'Pha')

            self.deviation_plot.plot(deviation[0], dev_amp,
                                pen='m',
                                symbol="x",
                                symbolSize=4,
                                symbolBrush="c",
                                clear=False,
                                name = 'Amp')
            
            min_index = np.argwhere(deviation[1] == np.min(deviation[1]))

            #print(min_index[0][0])

            self.deviation_selection_line = pg.InfiniteLine(pos=deviation[0][min_index[0][0]],
                                                angle=90,
                                                pen=pg.mkPen("m", width=2),
                                                movable=True,
                                                bounds=None,
                                                )

            self.deviation_plot.addItem(self.deviation_selection_line)

    def prep_export_params(self):
        paramas = {"probe_file":self.probe_file,
            "energy_kev":self.dsb_energy.value(),
            "recon_pixel_size_nm":self.dsb_calc_pixel_size.value(),
            "det_distance_m":self.dsb_det_dist.value(),
            "det_pixel_size_um":self.dsb_det_pixel_size.value(),
            "prop_start_um":self.dsb_prop_start.value(),
            "prop_end_um":self.dsb_prop_end.value(),
            "prop_step_size_um":self.dsb_prop_size.value(),
            "xfit_fwhm":self.xfwhm,
            "yfit_fwhm":self.yfwhm,
            "prop_distance":self.prop_distance
            }
        
        return paramas


    def export_current_data(self):

        save_path = os.path.abspath(self.probe_file).split(('.'))[0]

        export_file_name = QFileDialog.getSaveFileName(self,
                                                "save_all_live_data",
                                                save_path+"_prop_data",
                                                "All Files (*)")
        
        paramas = self.prep_export_params()

        exporter_csv_xfit = pg.exporters.CSVExporter(self.plot_xfit.plotItem)
        exporter_csv_xfit.parameters()["columnMode"] = '(x,y) per plot'
        exporter_csv_xfit.parameters()["separator"] = 'comma'

        exporter_csv_yfit = pg.exporters.CSVExporter(self.plot_yfit.plotItem)
        exporter_csv_yfit.parameters()["columnMode"] = '(x,y) per plot'
        exporter_csv_yfit.parameters()["separator"] = 'comma'

        exporter_csv_sigma = pg.exporters.CSVExporter(self.sigma_plot.plotItem)
        exporter_csv_sigma.parameters()["columnMode"] = '(x,y) per plot'
        exporter_csv_sigma.parameters()["separator"] = 'comma'

        exporter_png_xhue = pg.exporters.ImageExporter(self.image_view_xhue.getView())
        exporter_png_yhue = pg.exporters.ImageExporter(self.image_view_yhue.getView())
        exporter_png_probe = pg.exporters.ImageExporter(self.image_view_probe.getView())

        exporter_png_xfit = pg.exporters.ImageExporter(self.plot_xfit.scene())
        exporter_png_xfit.parameters()['width'] = 600
        exporter_png_xfit.parameters()['height'] = 600
        exporter_png_xfit.parameters()['antialias'] = True

        exporter_png_yfit = pg.exporters.ImageExporter(self.plot_yfit.scene())
        exporter_png_yfit.parameters()['width'] = 600
        exporter_png_yfit.parameters()['height'] = 600
        exporter_png_yfit.parameters()['antialias'] = True

        exporter_png_sigma = pg.exporters.ImageExporter(self.sigma_plot.scene())
        exporter_png_sigma.parameters()['width'] = 600
        exporter_png_sigma.parameters()['height'] = 600
        exporter_png_sigma.parameters()['antialias'] = True

        if export_file_name[0]:

            tf.imwrite(export_file_name[0]+"_propagated_stack.tiff", 
                      np.float32(np.abs(self.prb_array.transpose(2,0,1))),
                      imagej = True)
            exporter_csv_xfit.export(export_file_name[0]+"_xfit_data.csv")
            exporter_csv_yfit.export(export_file_name[0]+"_yfit_data.csv")
            exporter_csv_sigma.export(export_file_name[0]+"_sigma_data.csv")
            exporter_png_xfit.export(export_file_name[0]+"_xfit_image.png")
            exporter_png_yfit.export(export_file_name[0]+"_yfit_image.png")
            exporter_png_sigma.export(export_file_name[0]+"_sigma_image.png")
            exporter_png_xhue.export(export_file_name[0]+"_xhue_image.png")
            exporter_png_yhue.export(export_file_name[0]+"_yhue_image.png")
            exporter_png_probe.export(export_file_name[0]+"_probe_image.png")

            tf.imwrite(export_file_name[0]+"_xhue.tiff", 
                      np.float32(self.x_hue),
                      imagej = True)
            
            tf.imwrite(export_file_name[0]+"_yhue.tiff", 
                      np.float32(self.y_hue),
                      imagej = True)

            with open(export_file_name[0]+"_paramters.json", 'w') as fp:
                json.dump(paramas, fp, indent=6)

    def export_params(self):
        
        save_path = os.path.abspath(self.probe_file).split(('.'))[0]+"_prop_params.json"

        export_file_name = QFileDialog.getSaveFileName(self,
                                                "save_all_live_data",
                                                save_path,
                                                "All Files (*)")

        paramas = self.prep_export_params()

        with open(export_file_name[0], 'w') as fp:
            json.dump(paramas, fp, indent=6)


        
if __name__ == '__main__':
    try:
        QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except:
        pass
    app = QtWidgets.QApplication(sys.argv)
    w = ProbePropagationGUI()
    w.show()
    sys.exit(app.exec_())
