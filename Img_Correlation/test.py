import pyqtgraph as pg
import sys
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic
import matplotlib.pyplot as plt


im1 = plt.imread(r'C:\Users\pattammattel\Desktop\Satish_2021Q1\Optical\JD_Diatom_1X.png')
im2 = plt.imread(r'C:\Users\pattammattel\Desktop\Satish_2021Q1\Optical\JD_Diatom_100X_2.png')

win = pg.GraphicsWindow()
vb = win.addPlot()
img1 = pg.ImageItem(im1)
img2 = pg.ImageItem(im2)
vb.addItem(img1)
vb.addItem(img2)
hist = pg.HistogramLUTItem()
hist.setImageItem(img2)
hist2 = pg.HistogramLUTItem()
hist2.setImageItem(img1)
win.addItem(hist)
win.addItem(hist2)
img2.setZValue(10) # make sure this image is on top
img2.setCompositionMode(QtGui.QPainter.CompositionMode_Overlay)
img2.setOpacity(0.5)


#img2.scale(10, 10)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = win
    window.show()
    sys.exit(app.exec_())