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

        #connections
        self.actionLoad_refImage.triggered.connect(self.loadRefImage)

    def loadRefImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '', 'image file(*png *jpeg *tiff *tif )')
        if self.file_name:
            self.ref_image = plt.imread(self.file_name[0])
            self.statusbar.showMessage(f'{self.file_name[0]} selected')
        else:
            self.statusbar.showMessage("No file has selected")
            pass

        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.ref_view.addPlot(title="")

        # Item for displaying image data
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        self.ref_image = rotate(self.ref_image, -90)
        self.img.setImage(self.ref_image)
        #self.img.translate(100, 50)
        #self.img.scale(0.5, 0.5)
        self.img.hoverEvent = self.imageHoverEvent
        self.img.mousePressEvent = self.MouseClickEvent
        self.createLabAxisImage()

    def createLabAxisImage(self):
        # A plot area (ViewBox + axes) for displaying the image
        self.p2 = self.labaxis_view.addPlot(title="")

        # Item for displaying image data
        self.img2 = pg.ImageItem()
        self.p2.addItem(self.img2)
        self.img2.setImage(self.ref_image)
        self.img2.rotate(-90)
        self.img2.scale(0.5, 0.5)
        self.img2.translate(100, 50)


    def imageHoverEvent(self,event):
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
        x, y = np.around(ppos.x(),2), np.around(ppos.y(),2)
        self.p1.setTitle(f'pos: {x,y}  pixel: {i,j}  value: {val}')

    def MouseClickEvent(self,event):
        """Show the position, pixel, and value under the mouse cursor.
        """

        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        self.coords.append((i, j))
        val = self.ref_image[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = np.around(ppos.x(),2)/10, np.around(ppos.y(),2)/10
        #x, y = smarx.pos, smary.pos
        self.coords.append((x, y))
        if len(self.coords) == 2:
            self.le_ref1_pxls.setText(str(self.coords[0]))
            self.dsb_ref1_x.setValue(self.coords[1][0])
            self.dsb_ref1_y.setValue(self.coords[1][1])
        elif len(self.coords) == 4:
            self.le_ref1_pxls.setText(str(self.coords[0]))
            self.dsb_ref1_x.setValue(self.coords[1][0])
            self.dsb_ref1_y.setValue(self.coords[1][1])
            self.le_ref2_pxls.setText(str(self.coords[2]))
            self.dsb_ref2_x.setValue(self.coords[-1][0])
            self.dsb_ref2_y.setValue(self.coords[-1][1])
        print(self.coords[-1])
        print(len(self.coords))


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    #app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = ImageCorrelationWindow()
    window.show()
    sys.exit(app.exec_())