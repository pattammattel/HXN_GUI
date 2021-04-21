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
cmap_label1 = ['red','green','blue']
cmap_label2 = ['yellow','magenta','cyan']
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

        self.canvas = self.img_view.addPlot(title="")
        self.canvas.getViewBox().invertY(True)

        self.actionLoad_1.triggered.connect(self.loadMultipleImages)
        self.actionLoad_2.triggered.connect(self.loadImage2)
        self.actionLoad_3.triggered.connect(self.loadImage3)
        self.actionLoad_4.triggered.connect(self.loadImage4)
        self.actionLoad_5.triggered.connect(self.loadImage5)
        self.actionLoad_6.triggered.connect(self.loadImage6)

    def generateImageDictionary(self):
        filter = "TIFF (*.tiff);;TIF (*.tif)"
        file_name = QtWidgets.QFileDialog()
        file_name.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        names = file_name.getOpenFileNames(self, "Open files", " ", filter)
        if names[0]:
            self.image_dict = {}
            for n, image in enumerate(names[0]):
                self.image_dict['image'+str(n+1)] = np.squeeze(tf.imread(image))
        else:
            pass

    def loadAnImage(self, image, colormap):
        img = pg.ImageItem()
        self.canvas.addItem(img)
        img.setImage(image, lut=colormap)
        img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

    def createMultiColorView(self, image_dictionary):
        self.canvas.clear()
        for im, colors in zip(image_dictionary.values(),cmap_dict.values()):
            self.loadAnImage(im, colors)

    def loadMultipleImages(self):
        self.generateImageDictionary()
        self.createMultiColorView(self.image_dict)

    def loadImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                                 'image file(*png *jpeg *tiff *tif )')
        if self.file_name[0]:
            self.ref_image = np.squeeze(tf.imread(self.file_name[0]))
            #self.ref_image = cv2.imread(self.file_name[0])
            #self.ref_image = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2RGB)
            self.p1 = self.img_view.addPlot(title="")
            self.p1.getViewBox().invertY(True)
            # Item for displaying image data
            self.img = pg.ImageItem()
            hlut = pg.HistogramLUTItem(image=self.img,fillHistogram = False)
            self.p1.addItem(self.img)
            #self.p1.addItem(hlut)
            #hlut.gradient.setColorMap(cmap_dict['red'])
            self.img.setImage(self.ref_image, lut = cmap_dict['red'])
            #self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)


        else:
            self.statusbar.showMessage("No file has selected")
            pass

    def loadImage2(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image2 = np.squeeze(tf.imread(file_name[0]))
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img2 = pg.ImageItem()
            self.p1.addItem(self.img2)
            self.img2.setImage(self.ref_image2,lut = cmap_dict['green'])
            self.img2.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        else:
            pass

    def loadImage3(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image3 = np.squeeze(tf.imread(file_name[0]))
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img3 = pg.ImageItem()
            self.p1.addItem(self.img3)
            self.img3.setImage(self.ref_image3, lut = cmap_dict['blue'])
            self.img3.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        else:
            pass

    def loadImage4(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image4 = np.squeeze(tf.imread(file_name[0]))
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img4 = pg.ImageItem()
            self.p1.addItem(self.img4)
            self.img4.setImage(self.ref_image4, lut = cmap_dict['yellow'])
            self.img4.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        else:
            pass

    def loadImage5(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image5 = np.squeeze(tf.imread(file_name[0]))
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img5 = pg.ImageItem()
            self.p1.addItem(self.img5)
            self.img5.setImage(self.ref_image5, lut = cmap_dict['magenta'])
            self.img5.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        else:
            pass

    def loadImage6(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                            'image file(*png *jpeg *tiff *tif )')
        if file_name[0]:
            self.ref_image6 = np.squeeze(tf.imread(file_name[0]))
            #self.ref_image2 = cv2.cvtColor(self.ref_image2, cv2.COLOR_BGR2RGB)

            self.img6 = pg.ImageItem()
            self.p1.addItem(self.img6)
            self.img6.setImage(self.ref_image6, lut = cmap_dict['blue'])
            self.img6.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        else:
            pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = MultiChannelWindow()
    window.show()
    sys.exit(app.exec_())