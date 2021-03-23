import pyqtgraph

app = pyqtgraph.QtGui.QApplication([])
win = pyqtgraph.GraphicsLayoutWidget()
qi = pyqtgraph.QtGui.QImage()
label = pyqtgraph.QtGui.QLabel(win)

qi.load('reg+with_fluo_offset_on.png')
label.setPixmap(pyqtgraph.QtGui.QPixmap.fromImage(qi))

win.show()


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pyqtgraph.QtGui.QApplication.instance().exec_()
