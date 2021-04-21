import sys,os,json,collections
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic
import tifffile as tf
from itertools import combinations
import time

pg.setConfigOption('imageAxisOrder', 'row-major')

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

        #connections
        self.actionLoad.triggered.connect(self.loadMultipleImages)
        self.cb_choose_color.currentTextChanged.connect(self.updateImageDictionary)

    def generateImageDictionary(self):
        filter = "TIFF (*.tiff);;TIF (*.tif)"
        file_name = QtWidgets.QFileDialog()
        file_name.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        names = file_name.getOpenFileNames(self, "Open files", " ", filter)
        if names[0]:
            self.image_dict = {}
            self.imageDir = os.path.dirname(names[0][0])
            for colorName, image in zip(cmap_dict.keys(),names[0]):
                im_name = os.path.basename(image)
                self.image_dict[f'{os.path.basename(image)}'] = {'ImageName':im_name,
                                                                 'ImageDir':self.imageDir,
                                                                 'Color':colorName
                                                                 }
        else:
            pass

    def loadAnImage(self, image_path, colormap):
        img = pg.ImageItem()
        self.canvas.addItem(img)
        image = np.squeeze(tf.imread(image_path))
        img.setImage(image, lut=colormap)
        img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

    def createMultiColorView(self, image_dictionary):
        self.canvas.clear()
        self.listWidget.clear()
        for path_and_color in image_dictionary.values():
            self.loadAnImage(os.path.join(path_and_color['ImageDir'],path_and_color['ImageName']),
                             cmap_dict[path_and_color['Color']])

    def loadMultipleImages(self):
        ''' Load Images with default color assignment'''
        with pg.BusyCursor():
            self.generateImageDictionary()
            if self.image_dict:
                self.createMultiColorView(self.image_dict)
                self.displayImageNames(self.image_dict)

            else:
                pass

    def displayImageNames(self,image_dictionary):
        for im_name,vals in image_dictionary.items():
            self.listWidget.addItem(f"{im_name}, {vals['Color']}")

    def updateImageDictionary(self):
        newColor = self.cb_choose_color.currentText()
        editItem = self.listWidget.currentItem().text()
        editItemName = editItem.split(',')[0]
        self.image_dict[editItemName] = {'ImageName':editItemName,
                                         'ImageDir':self.imageDir,
                                         'Color':newColor
                                         }
        self.createMultiColorView(self.image_dict)
        self.displayImageNames(self.image_dict)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = MultiChannelWindow()
    window.show()
    sys.exit(app.exec_())