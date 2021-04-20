import sys,os,json,collections
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic
import tifffile as tf
pg.setConfigOption('imageAxisOrder', 'row-major')
from itertools import combinations

cmap_names = ['CET-L13','CET-L14','CET-L15']
cmap_combo = combinations(cmap_names, 2)
cmap_label1 = ['r','g','b']
cmap_label2 = ['rg','rb','gb']
cmap_dict = {}
for i,name in zip(cmap_names,cmap_label1):
    cmap_dict[name] = pg.colormap.get(i).getLookupTable(alpha=True)

for i,name in zip(cmap_combo,cmap_label2):
    cmap_dict[name] = (pg.colormap.get(i[0]).getLookupTable(alpha=True)+
                       pg.colormap.get(i[1]).getLookupTable(alpha=True))//2

class MultiChannelWindow(QtWidgets.QMainWindow):
    def __init__(self, ref_image=None):
        super(MultiChannelWindow, self).__init__()
        uic.loadUi('mutlichannel.ui', self)

        self.actionLoad_1.triggered.connect(self.loadImage)
        self.actionLoad_2.triggered.connect(self.loadImage2)
        self.actionLoad_3.triggered.connect(self.loadImage3)

    def loadImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                                 'image file(*png *jpeg *tiff *tif )')
        if self.file_name[0]:
            self.ref_image = tf.imread(self.file_name[0])
            #self.ref_image = cv2.imread(self.file_name[0])
            #self.ref_image = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2RGB)

            # A plot area (ViewBox + axes) for displaying the image
            self.p1 = self.img_view.addPlot(title="")
            self.p1.getViewBox().invertY(True)
            # Item for displaying image data
            self.img = pg.ImageItem()
            self.p1.addItem(self.img)
            self.img.setImage(self.ref_image, lut = cmap_dict['rg'])
            #self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)


        else:
            self.statusbar.showMessage("No file has selected")
            pass

    def loadImage2(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image2 = tf.imread(file_name[0])
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img2 = pg.ImageItem()
            self.p1.addItem(self.img2)
            cmap = pg.colormap.get('CET-L14')
            self.img2.setImage(self.ref_image2,lut = cmap_dict['rb'])
            self.img2.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
            #self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        else:
            pass

    def loadImage3(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image3 = tf.imread(file_name[0])
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img3 = pg.ImageItem()
            self.p1.addItem(self.img3)
            cmap = pg.colormap.get('CET-L15')
            cmap2 = pg.colormap.get('CET-L14')
            self.img3.setImage(self.ref_image3, lut = cmap_dict['gb'])
            self.img3.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        else:
            pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = MultiChannelWindow()
    window.show()
    sys.exit(app.exec_())