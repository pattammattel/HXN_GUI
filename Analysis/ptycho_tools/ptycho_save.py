#conda env : /GPFS/XF03ID1/shared/conda_envs/ptycho_production
#from probe_propagation.prop_probe_v2 import *
from ptycho_save_tools import  *

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
from scipy.ndimage.measurements import center_of_mass
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog,QErrorMessage
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal
ui_path = os.path.dirname(os.path.abspath(__file__))


class ptychoSaveWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(ptychoSaveWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'ptycho_save.ui'), self)
        self.image_view.setBackground((222, 222, 222))
        self.db = hxn_db
        self.first_load = True


        #self.updateDisplayParams(self.config)
        self.le_scan_num.setText(str(187309)) #testing purposes
        self.le_wd.setText((str(os.getcwd())))
        self.config = {"wd":os.getcwd()}

        #connections
        self.pb_wd.clicked.connect(self.chooseWD)
        self.pb_load_data.clicked.connect(self.initializeParameters)
        self.pb_save_ptycho_h5.clicked.connect(lambda:self.save_h5_thread(self.config))

        self.pb_start_ptycho_batch.clicked.connect(lambda:self.save_h5_batch_thread(self.config))
        self.le_hot_pixels.editingFinished.connect(self.userUpdateHotPixelList)
        self.le_outl_pixels.editingFinished.connect(self.userUpdateOutlPixelList)
        self.pb_show_corr_image.clicked.connect(lambda:self.display_corrected_image(self.single_img,
                                                                           self.config,
                                                                           crop = self.cb_dspl_crop.isChecked(),
                                                                           fftshift = self.cb_dspl_fft.isChecked(),
                                                                           plotAfter = True))

        self.rb_log_display.pressed.connect(lambda:self.toggle_image_log(self.ptychoImage))
        self.pb_add_roi.clicked.connect(self.addROI)
        self.le_scan_num.editingFinished.connect(self.userUpdateHotPixelList)
        self.pb_updateROI.clicked.connect(self.updateROI)

        #import/export json
        self.actionLoad_Param.triggered.connect(self.importParameters)
        self.actionSave_Param.triggered.connect(self.exportParameters)


        '''
        #memory issues
        [sb.valueChangeFinished.connect(self.updateROI) for sb in
         [self.sb_roi_xpos,self.sb_roi_ypos,self.sb_roi_xsize,self.sb_roi_ysize]]
        '''

        self.show()

    def exception_messager(func):

        def inner_function(self,*args, **kwargs):

            try:
                self.func(*args, **kwargs)

            except Exception as excep_msg:
                err_msg = QErrorMessage(self)
                err_msg.setWindowTitle("An Error Occured")
                err_msg.showMessage(excep_msg)

        return inner_function

    def updateDisplayParams(self, param_dict):

        self.le_wd.setText(param_dict["wd"])
        self.le_scan_num.setText(str(param_dict["scan_num"]))
        index = self.cb_detector.findText(param_dict["detector"], QtCore.Qt.MatchFixedString )
        if index >= 0:
            self.cb_detector.setCurrentIndex(index)

        self.sb_roi_xpos.setValue(param_dict["crop_roi"][0])
        self.sb_roi_ypos.setValue(param_dict["crop_roi"][1])
        self.sb_roi_xsize.setValue(param_dict["crop_roi"][2])
        self.sb_roi_ysize.setValue(param_dict["crop_roi"][3])

        self.le_hot_pixels.setText(str(param_dict["hot_pixels"])[1:-1])
        self.le_outl_pixels.setText(str(param_dict["outl_pixels"])[1:-1])

        self.ch_b_switch_xy.setChecked(param_dict["switchXY"])

    def chooseWD(self):

        """updates the line edit for working directory"""

        self.foldername = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.le_wd.setText((str(self.foldername)))


    def initializeParameters(self):

        """get necessary metadata and a single image frame for display"""

        if self.first_load == True:
            #set the config file with user input
            self.config["wd"] =  self.le_wd.text()
            self.config["scan_num"] = int(self.le_scan_num.text())
            self.config["detector"] = self.cb_detector.currentText()
            self.config["crop_roi"] = (64,64,128,128)
            self.config["hot_pixels"] = []
            self.config["outl_pixels"] = []
            self.config["switchXY"] = False
            self.config["db"]  = hxn_db

            # lists to store pixels values when clicked ; later add to config
            # directly adding to config and displaying could be time consuming
            self.list_of_hot_pixels = []
            self.list_of_outl_pixels = []

            self.first_load = False

        else:
            pass

        #get metadata from databorker; testwith 173961
        self.header = self.db[self.config["scan_num"]]

        #create some metadata as class object for loading whole data later
        self.plan_args = self.header.start['plan_args']
        self.scan_type = self.header.start['plan_name']
        self.bl = self.db.get_table(self.header, stream_name='baseline')
        self.config["energy"] = self.bl.energy.iloc[0]
        self.config["det_dist"] = 0.5
        self.config["scan_num"] = int(self.le_scan_num.text())
        print(self.config)
        self.updateDisplayParams(self.config)

        '''
        self.config["energy"] = self.bl.energy.iloc[0]
        lambda_nm = 1.2398 /self.config["energy"]
        det_pixel_um = 55.

        img_size = self.sb_roi_xsize.value()
        self.pixel_size,depth_of_field = calculate_res_and_dof(self.config["energy"],
                                                               self.config["det_dist"],
                                                               det_pixel_um,
                                                               img_size)

        self.label_re_dof.setText(f'Pixel Size = {self.pixel_size*1.e9 :.2f} nm, '
                                  f'depth_of_field = {depth_of_field*1.e6 :.2f} um')
        '''

        #findout the first scanning axis; some instances y scanned first
        self.motors = self.header.start['motors']

        #the switchXY bool will be used to revese det data later
        if self.motors[0].endswith('y'):
            self.config["switchXY"] = True
        else:
            self.config["switchXY"] = False

        #make sure the detector is in the metadata
        merlins = []
        for name in self.header.start['detectors']:
            if name.startswith('merlin'):
                merlins.append(name)

        #todo edit items based on what's in the det list
        det_name = self.config["detector"]

        if det_name in merlins:

            items = [det_name, 'sclr1_ch3', 'sclr1_ch4'] + self.motors
            #create dataframe with items
            self.df = self.db.get_table(self.header, fields=items, fill=False)
            mds_table = self.df[det_name]

            #lod a single merlin image
            self.single_img = get_single_image(hxn_db, self.sb_det_frame_num.value(), mds_table)
            self.displayAnImage(self.single_img)
            self.addROI()
            self.updateROI()

        else:
            #raise error window if detector is not part of the scan
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Error")
            error_message.showMessage(str(f"No {det_name} found"))
            return

    def importParameters(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,
                                                         'Select Parameter File',
                                                         self.config["wd"],
                                                          "All Files (*);;json Files (*.json)"
                                                         )
        if filename[0]:
            with open(filename[0],'r') as fp:
                self.config = json.load(filename[0])

            self.updateDisplayParams(self.config)

        else:
            pass

    def exportParameters(self):

        save_filename = QtWidgets.QFileDialog.getSaveFileName(self,
                                                            "Save Parameters",
                                                            os.path.join(self.config["wd"],"ptycho_parameters.json"),
                                                            "All Files (*);;json Files (*.json)")
        if save_filename[0]:
            with open(save_filename[0], "w") as fp:
                json.dump(self.config,fp,indent=4)


        else:
            pass

    def save_h5_thread(self, config):
        """ Long process saving h5 data"""

        try:
            self.saveThread.quit()

        except:
            pass

        try:
            self.statusbar.showMessage(" Loading data; This could take a while...Please wait... ")
            self.saveThread = QThread()
            self.saveWorker = savePtychoH5(config)
            self.saveWorker.moveToThread(self.saveThread)
            self.saveThread.started.connect(self.saveWorker.save_h5_thread)
            self.saveWorker.save_started.connect(lambda:self.h5_save_pbar.setRange(0,0))
            self.saveWorker.finished.connect(lambda: self.h5_save_pbar.setRange(0, 100))
            self.saveWorker.finished.connect(lambda: self.statusbar.showMessage("h5 Saved"))
            self.saveWorker.finished.connect(lambda: self.h5_save_pbar.setValue(100))
            self.saveWorker.finished.connect(self.saveThread.quit)
            self.saveThread.start()

        except Exception as e:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage(str(e))
            self.saveThread.quit()
            self.h5_save_pbar.reset()

            return

    #batch and live

    def save_h5_batch_thread(self, config):
        """ Long process saving h5 data"""
        scan_num_list = parse_scan_range(self.le_ptycho_batch_scan_nums.text())
        self.statusbar.showMessage(f"Scans to process : {scan_num_list}")
        self.statusbar.showMessage(" Please wait... Loading data")
        self.batch_save_thread = QThread()
        self.batch_worker = SavePtychoBatch(config)
        self.batch_worker.moveToThread(self.batch_save_thread)
        self.batch_save_thread.started.connect(lambda:self.batch_worker.save_h5_batch_generator(scan_num_list))
        self.batch_save_thread.start()
        self.batch_worker.finished.connect(self.batch_save_thread.quit)
        self.statusbar.showMessage("Done")

    def live_ptycho_processing(self, config):

        "live saving data"

        self.scan_num_gen_thread = QThread()
        self.scan_num_gen_worker = livePtychoConfigGenerator(self.sb_buffer_time.value(), config)
        self.scan_num_gen_worker.moveToThread(self.scan_num_gen_thread)
        self.scan_num_gen_worker.new_parameters.connect(self.save_h5_thread)
        self.scan_num_gen_thread.start()

    #probe propogation
    def choose_probe(self):
        file_name = QFileDialog().getOpenFileName(self, "Open Probe file", '', 'npy file (*.npy)')
        self.le_probe_file.setText(str(file_name[0]))
        self.probe_file = file_name[0]

    def plot_propagation_data(self):

        self.probe_plotter = self.prp_canvas.addPlot(title = "Probe", row=0, col=0)
        self.probe_stack_image = pg.ImageItem(axisOrder='row-major')
        self.probe_stack_image.setImage(tf.imread("test_data/prb_array.tiff").transpose(0,1,2))
        self.probe_plotter.addItem(self.probe_stack_image)

    def displayAnImage(self, img, im_title = " "):

        try:
            self.image_view.clear()
        except:
            pass

        self.ptychoImage = img
        ysize,xsize = np.shape(self.ptychoImage)
        self.statusbar.showMessage(f"Image Shape = {np.shape(self.ptychoImage)}")
        # A plot area (ViewBox + axes) for displaying the image

        self.p1 = self.image_view.addPlot(title= str(im_title))
        #self.p1.setAspectLocked(True)
        self.p1.getViewBox().invertY(True)
        self.p1.getViewBox().setLimits(xMin = 0,
                                       xMax = xsize,
                                       yMin = 0,
                                       yMax = ysize
                                       )

        # Item for displaying image data
        self.img = pg.ImageItem(axisOrder = 'row-major')
        self.img.setImage(np.ma.log(self.ptychoImage), opacity=1)
        self.p1.addItem(self.img)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.autoHistogramRange = True
        self.image_view.addItem(self.hist)
        #self.hist.plot.setLogMode(False, True)

        # Create color bar and have it control image levels
        #cmap = pg.colormap.getFromMatplotlib('viridis')
        #cmap = pg.colormap.get('spectrum')
        #cbi = pg.ColorBarItem(colorMap=cmap)
        #cbi.setImageItem(self.img, insert_in=self.p1)
        #cbi.setLevels([np.nanmin(self.ptychoImage), np.nanmax(self.ptychoImage)])  # colormap range
        
        #colormap = cmap_dict['red']
        #cmap = pg.ColorMap(pos=np.linspace(0, 1, len(colormap)), color=colormap)
        # image = np.squeeze(tf.imread(image_path))
        # set image to the image item with cmap

        
        #self.img.setImage(self.ptychoImage, opacity=1, lut=cmap.getLookupTable())
        #self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        # self.img.translate(100, 50)
        # self.img.scale(0.5, 0.5)

        self.img.hoverEvent = self.imageHoverEvent
        self.img.mousePressEvent = self.MouseClickEvent

    def toggle_image_log(self, image):

        if self.rb_log_display.isChecked():
            self.img.setImage(np.ma.log(image), opacity=1)

        else:
            self.img.setImage(image, opacity=1)

    def addROI(self):

        try:
            self.p1.removeItem(self.rectROI)
        except:
            pass

        yshape,xshape = np.shape(self.single_img)
        ycen, xcen = center_of_mass(self.single_img)

        #pixel_size = lambda_nm * 1.e-9 * det_distance_m / (img_size * det_pixel_um * 1e-6)
        lambda_nm = 1.2398 / self.config["energy"]
        det_pixel_um = 55.
        res,dof = calculate_res_and_dof(self.config["energy"],self.config["det_dist"],det_pixel_um,self.sb_roi_xsize.value())
        print(f'calc_res, dof = {res,dof}')
        print(f'params; {self.config["energy"]} {self.config["det_dist"]}, {det_pixel_um},{self.sb_roi_xsize.value()}')

        roi_size = calculate_img_size(self.config["energy"],self.config["det_dist"],det_pixel_um,round(res*1.e9))
        print(f"roi_size = {roi_size}")
        try:
            self.rectROI = pg.RectROI([int(xcen)-64,int(ycen)-64],
                                      [roi_size,roi_size],
                                      maxBounds=QtCore.QRectF(0, 0, xshape, yshape),
                                      pen='r')

        except:

            self.rectROI = pg.RectROI([0,0],
                                      [roi_size,roi_size],
                                      maxBounds=QtCore.QRectF(0, 0, xshape, yshape),
                                      pen='r')

        self.rectROI.addTranslateHandle([0.5,0.5],[0.5,0.5])
        self.p1.addItem(self.rectROI)
        self.updateROIParams()
        self.rectROI.sigRegionChangeFinished.connect(self.updateROIParams)

    def updateROIParams(self):
        roi_pos = self.rectROI.pos()
        roi_size = self.rectROI.size()
        self.sb_roi_xpos.setValue(int(roi_pos[0]+roi_size[0]//2))
        self.sb_roi_ypos.setValue(int(roi_pos[1]+roi_size[1]//2))
        self.sb_roi_xsize.setValue(int(roi_size[0]))
        self.sb_roi_ysize.setValue(int(roi_size[1]))

        self.config["crop_roi"] = (int(roi_pos.x()), int(roi_pos.y()), int(roi_size.x()), int(roi_size.y()))
        print(self.config["crop_roi"])

    def updateROI(self):
        try:
            roi_pos = QtCore.QPointF(self.sb_roi_xpos.value()-self.sb_roi_xsize.value()//2,
                                     self.sb_roi_ypos.value()-self.sb_roi_ysize.value()//2)
            roi_size = QtCore.QPointF(self.sb_roi_xsize.value(),self.sb_roi_ysize.value())
            self.rectROI.setPos(roi_pos)
            self.rectROI.setSize(roi_size)

        except RuntimeError:
            pass

        self.config["crop_roi"] = (int(roi_pos.x()),
                                   int(roi_pos.y()),
                                   int(roi_size.x()),
                                   int(roi_size.y()))

        self.config["energy"] = self.bl.energy.iloc[0]
        lambda_nm = 1.2398 / self.config["energy"]
        det_distance_m = self.config["det_dist"]
        det_pixel_um = 55.
        img_size = self.sb_roi_xsize.value()
        pixel_size, depth_of_field = calculate_res_and_dof(self.config["energy"],
                                                           det_distance_m,
                                                           det_pixel_um,
                                                           img_size)

        self.label_re_dof.setText(f'Pixel Size = {pixel_size*1.e9 :.2f} nm, '
                                  f'depth_of_field = {depth_of_field*1.e6:.2f} um')

    def updateHotOutlPixels(self):
        #update the pixel values to config
        self.config["hot_pixels"] = self.list_of_hot_pixels
        self.config["outl_pixels"] = self.list_of_outl_pixels

    def userUpdateHotPixelList(self):

        #get the new list of vals from the line edit.
        # Assuming user does not mess with the structure
        #may be change to a list widget later
        newList_str = '[' + self.le_hot_pixels.text() +']'

        # Converting string to list
        self.list_of_hot_pixels = ast.literal_eval(newList_str)
        self.config["hot_pixels"] = self.list_of_hot_pixels

    def userUpdateOutlPixelList(self):

        #get the new list of vals from the line edit.
        # Assuming user does not mess with the structure
        #may be change to a list widget later
        newList_str = '[' + self.le_outl_pixels.text() +']'

        # Converting string to list
        self.list_of_outl_pixels = ast.literal_eval(newList_str)
        self.config["outl_pixels"] = self.list_of_outl_pixels

    def display_corrected_image(self,img, config_, crop = True, fftshift = False, plotAfter = True):

        self.updateHotOutlPixels()
        replacePixelValues(img, config_["outl_pixels"], setToZero=True)
        replacePixelValues(img, config_["hot_pixels"], setToZero=False)

        #cx, cy = center of the ROI
        n, nn = int(self.config["crop_roi"][-2]), int(self.config["crop_roi"][-1])
        cx,cy = int(self.config["crop_roi"][0]), int(self.config["crop_roi"][1])

        # remove bad pixels

        if crop:
            data= img[cy:nn+cy, cx:n+cx]
        else:
            data = img

        if fftshift:
            data = np.fft.fftshift(data)
            threshold = 1.
            data = data - threshold
            data[data < 0.] = 0.
            data = np.sqrt(data)

        if plotAfter:

            self.displayAnImage(data)

        else:
            return data

    def MouseClickEvent(self, event = QtCore.QEvent):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
            if event.button() == QtCore.Qt.LeftButton:
                pos = self.img.mapToParent(event.pos())
                i, j = pos.x(), pos.y()
                limits = self.img.mapToParent(QtCore.QPointF(self.ptychoImage.shape[0], self.ptychoImage.shape[1]))
                i = int(np.clip(i, 0, limits.y() - 1))
                j = int(np.clip(j, 0, limits.x() - 1))

                if self.rb_choose_hot.isChecked():

                    if not (i,j) in self.list_of_hot_pixels:
                        self.list_of_hot_pixels.append((i,j))  # if not integer (self.xpixel,self.ypixel)
                        self.le_hot_pixels.setText(str(self.list_of_hot_pixels)[1:-1])

                elif self.rb_choose_outl.isChecked():
                    if not (i,j) in self.list_of_outl_pixels:
                        self.list_of_outl_pixels.append((i,j))  # if not integer (self.xpixel,self.ypixel)
                        self.le_outl_pixels.setText(str(self.list_of_outl_pixels)[1:-1])

            else: event.ignore()
        else: event.ignore()

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ptychoImage.shape[1] - 1))
        j = int(np.clip(j, 0, self.ptychoImage.shape[0] - 1))
        val = self.ptychoImage[int(j), int(i)]
        self.p1.setTitle(f'pixel: {i, j}, Intensity : {val:.2f}')

class savePtychoH5(QObject):

    finished = pyqtSignal()
    save_started = pyqtSignal()

    def __init__(self, parameters, new_scan_num = None):
        super(savePtychoH5, self).__init__()
        self.parameters = parameters
        self.new_scan_num = new_scan_num

        if not new_scan_num is None:
            self.parameters["scan_num"] = self.new_scan_num

    def save_h5_thread(self):
        """ process one scan """
        self.save_started.emit()
        save_ptycho_h5(self.parameters,1,1)
        self.finished.emit()

    def save_h5_batch(self, scan_num_list):
        """ process a list of scans """

        for sid in scan_num_list:
            self.parameters["scan_num"] = int(sid)
            save_ptycho_h5(self.parameters, 1, 1)

        self.finished.emit()

class SavePtychoBatch(QObject):

    finished = pyqtSignal()

    def __init__(self, parameters):
        super(SavePtychoBatch, self).__init__()
        self.parameters = parameters

    def save_h5_batch_generator(self, scan_num_list):
        """ process a list of scans """

        for sid in scan_num_list:
            self.parameters["scan_num"] = int(sid)
            save_ptycho_h5(self.parameters, 1, 1)

        self.finished.emit()

class livePtychoConfigGenerator(QObject):

    new_parameters = pyqtSignal(dict)
    enable_live_button = pyqtSignal(bool)

    def __init__(self, buffertime,parameters):
        super().__init__()
        self.buffertime = buffertime
        self.parameters = parameters

    def live_processing(self):
        self.enable_live_button.emit(False)
        while True:
            QtTest.QTest.qWait(333)
            while caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and caget('XF:03IDC-ES{Sclr:2}_cts1.D')>5000:

                sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
                header = hxn_db[(sid)]

                while header.stop == True and self.new_parameters["scan_num"] != sid:
                    self.parameters["scan_num"] = sid
                    self.new_parameters.emit(self.parameters)
                    #self.sleep(self.buffertime)
                    QtTest.QTest.qWait(self.buffertime*1000)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ptychoSaveWindow()
    w.show()
    sys.exit(app.exec_())