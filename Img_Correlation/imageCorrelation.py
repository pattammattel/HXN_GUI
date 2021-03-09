import sys
import os
import collections
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import rotate
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic


class ImageCorrelationWindow(QtWidgets.QMainWindow):
    def __init__(self, ref_image=None):
        super(ImageCorrelationWindow, self).__init__()
        uic.loadUi('imageCorrelation.ui', self)
        self.ref_image = ref_image
        self.coords = collections.deque(maxlen=4)

        # connections
        self.actionLoad_refImage.triggered.connect(self.loadRefImage)
        self.pb_apply_calculation.clicked.connect(self.scalingCalculation)

    def loadRefImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                                 'image file(*png *jpeg *tiff *tif )')
        if self.file_name:
            self.ref_image = plt.imread(self.file_name[0])
            if self.ref_image.ndim == 3:
                self.ref_image = self.ref_image.sum(2)
            self.statusbar.showMessage(f'{self.file_name[0]} selected')
        else:
            self.statusbar.showMessage("No file has selected")
            pass

        try:
            self.ref_view.clear()
        except:
            pass

        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.ref_view.addPlot(title="")

        # Item for displaying image data
        self.img = pg.ImageItem()
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        self.ref_view.addItem(hist)

        self.p1.addItem(self.img)
        self.ref_image = rotate(self.ref_image, -90)
        self.img.setImage(self.ref_image)
        # self.img.translate(100, 50)
        # self.img.scale(0.5, 0.5)
        self.img.hoverEvent = self.imageHoverEvent
        self.img.mousePressEvent = self.MouseClickEvent

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        val = self.ref_image[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = np.around(ppos.x(), 2), np.around(ppos.y(), 2)
        self.p1.setTitle(f'pos: {x, y}  pixel: {i, j}  value: {val}')

    def MouseClickEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.button() == QtCore.Qt.LeftButton:

            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
            j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
            self.coords.append((i, j))
            val = self.ref_image[i, j]
            ppos = self.img.mapToParent(pos)
            x, y = np.around(ppos.x(), 2) , np.around(ppos.y(), 2)
            # x, y = smarx.pos, smary.pos
            self.coords.append((x, y))
            if len(self.coords) == 2:
                self.le_ref1_pxls.setText(f'{self.coords[0][0]}, {self.coords[0][1]}')
                self.dsb_ref1_x.setValue(self.coords[1][0])
                self.dsb_ref1_y.setValue(self.coords[1][1])
            elif len(self.coords) == 4:
                self.le_ref1_pxls.setText(f'{self.coords[0][0]}, {self.coords[0][1]}')
                self.dsb_ref1_x.setValue(self.coords[1][0])
                self.dsb_ref1_y.setValue(self.coords[1][1])
                self.le_ref2_pxls.setText(f'{self.coords[2][0]}, {self.coords[2][1]}')
                self.dsb_ref2_x.setValue(self.coords[-1][0])
                self.dsb_ref2_y.setValue(self.coords[-1][1])

    def createLabAxisImage(self):
        # A plot area (ViewBox + axes) for displaying the image

        try:
            self.labaxis_view.clear()
        except:
            pass

        self.p2 = self.labaxis_view.addPlot(title="")

        # Item for displaying image data
        self.img2 = pg.ImageItem()
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img2)
        self.labaxis_view.addItem(hist)
        self.p2.addItem(self.img2)
        self.img2.setImage(self.ref_image)


    def scalingCalculation(self):
        yshape, xshape = np.shape(self.ref_image)

        lm1_px, lm1_py = self.le_ref1_pxls.text().split(',')  # r chooses this pixel
        lm2_px, lm2_py = self.le_ref2_pxls.text().split(',')  # chooses this pixel

        lm1_x, lm1_y = self.dsb_ref1_x.value(), self.dsb_ref1_y.value()  # motor values from the microscope at pixel pos 1
        lm2_x, lm2_y = self.dsb_ref2_x.value(), self.dsb_ref2_y.value()  # motor values from the microscope at pixel pos 2

        self.pixel_val_x = (lm2_x - lm1_x) / (int(lm2_px) - int(lm1_px))  # pixel value of X
        self.pixel_val_y = (lm2_y - lm1_y) / (int(lm2_py) - int(lm1_py))  # pixel value of Y; ususally same as X

        self.xi = lm1_x - (self.pixel_val_x * int(lm1_px))  # xmotor pos at origin (0,0)
        xf = self.xi + (self.pixel_val_x * xshape)  # xmotor pos at the end (0,0)
        self.yi = lm1_y - (self.pixel_val_y * int(lm1_py))  # xmotor pos at origin (0,0)
        yf = self.yi + (self.pixel_val_y * yshape)  # xmotor pos at origin (0,0)
        self.createLabAxisImage()

        self.label_scale_info.setText(f'Scaling: {self.pixel_val_x:.2f}, {self.pixel_val_y:.2f}, \n '
                                      f' X Range {self.xi:.2f}:{xf:.2f}, \n'
                                      f'Y Range {self.yi:.2f}:{yf:.2f}')
        self.img2.scale(abs(self.pixel_val_x), abs(self.pixel_val_y))
        self.img2.translate(self.xi, self.yi)
        #self.img2.setRect(QtCore.QRect(xi,yf,yi,xf))
        self.img2.hoverEvent = self.imageHoverEvent2
        self.img2.mousePressEvent = self.MouseClickEventToPos

    def imageHoverEvent2(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p2.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        val = self.ref_image[i, j]
        x = self.xi + (self.pixel_val_x * i)
        y = self.yi + (self.pixel_val_y * j)
        self.p2.setTitle(f'pos: {x:.2f},{y:.2f}  pixel: {i, j}  value: {val:.2f}')

    def MouseClickEventToPos(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
            j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
            xWhere = self.xi + (self.pixel_val_x * i)
            yWhere = self.yi + (self.pixel_val_y * j)

            self.dsb_calc_x.setValue(xWhere + (self.dsb_x_off.value() * 0.001))
            self.dsb_calc_y.setValue(yWhere + (self.dsb_y_off.value() * 0.001))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = ImageCorrelationWindow()
    window.show()
    sys.exit(app.exec_())
