import sys,os,json,collections
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic
pg.setConfigOption('imageAxisOrder', 'row-major')


class MultiChannelWindow(QtWidgets.QMainWindow):
    def __init__(self, ref_image=None):
        super(MultiChannelWindow, self).__init__()
        uic.loadUi('mutlichannel.ui', self)

    def loadImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                                 'image file(*png *jpeg *tiff *tif )')
        if self.file_name[0]:
            self.ref_image = cv2.imread(self.file_name[0])
            self.ref_image = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2RGB)
            self.yshape, self.xshape = np.shape(self.ref_image)[:2]

            try:
                self.ref_view.clear()
            except:
                pass

            # A plot area (ViewBox + axes) for displaying the image
            self.p1 = self.ref_view.addPlot(title="")
            self.p1.getViewBox().invertY(True)
            # Item for displaying image data
            self.img = pg.ImageItem()
            self.p1.addItem(self.img)
            self.img.setImage(self.ref_image)
            self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
            # self.img.translate(100, 50)
            # self.img.scale(0.5, 0.5)

            self.img.hoverEvent = self.imageHoverEvent
            self.img.mousePressEvent = self.MouseClickEvent

        else:
            self.statusbar.showMessage("No file has selected")
            pass

    def loadImage2(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image2 = cv2.imread(file_name[0])
            self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img3 = pg.ImageItem()
            self.p1.addItem(self.img3)
            self.img3.setImage(self.ref_image2)
            self.img3.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
            self.img.setZValue(10)

        else:
            pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = MultiChannelWindow()
    window.show()
    sys.exit(app.exec_())