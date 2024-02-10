
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


class diffViewWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(diffViewWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'diff_view.ui'), self)


        #connections
        self.pb_choose_wd.clicked.connect(self.chooseWD)
        self.pb_view_diff_data.clicked.connect(lambda:self.display_img(self.xrf_plot_canvas, 
                                                                       img, 
                                                                       im_title = "XRF Image"))


    def chooseWD(self):

        """updates the line edit for working directory"""

        self.foldername = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.le_wd.setText((str(self.foldername)))


    def display_img(self, canvas_name, tiff_img_path, im_title = " "):

        try:
            canvas_name.clear()
        except:
            pass
        img = tf.imread(tiff_img_path)
        ysize,xsize = np.shape(img)
        self.statusbar.showMessage(f"Image Shape = {np.shape(img)}")
        # A plot area (ViewBox + axes) for displaying the image

        self.p1 = canvas_name.addPlot(title= str(im_title))
        #self.p1.setAspectLocked(True)
        self.p1.getViewBox().invertY(True)
        self.p1.getViewBox().setLimits(xMin = 0,
                                       xMax = xsize,
                                       yMin = 0,
                                       yMax = ysize
                                       )

        # Item for displaying image data
        self.img = pg.ImageItem(axisOrder = 'row-major')
        self.img.setImage(img, opacity=1)
        self.p1.addItem(self.img)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.autoHistogramRange = True
        canvas_name.addItem(self.hist)

        # self.img.hoverEvent = self.imageHoverEvent
        # self.img.mousePressEvent = self.MouseClickEvent


    def display_xrf_img(elem = "Au_L"):

        xrf = tf.imread(os.path.join(self.self.foldername, f"detsum_{elem}_norm.tiff"))

    

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
                limits = self.img.mapToParent(QtCore.QPointF(img.shape[0], img.shape[1]))
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
        i = int(np.clip(i, 0, img.shape[1] - 1))
        j = int(np.clip(j, 0, img.shape[0] - 1))
        val = img[int(j), int(i)]
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
    '''

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ptychoSaveWindow()
    w.show()
    sys.exit(app.exec_())