import sys,os
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5 import QtWidgets, uic, QtCore
import pyqtgraph as pg
import random
ui_path = os.path.dirname(os.path.abspath(__file__))

class livePlotViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super(livePlotViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_path, 'uis/LiveTest.ui'), self)

        self.L = deque([0], maxlen=10)
        self.t = deque([0], maxlen=10)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateplot)
        self.timer.start(1)

    def updateplot(self):
        val = round(random.uniform(0,10), 2)
        self.L.append(val)
        self.t.append(self.t[-1]+1)
        self.plot_view.plot(list(self.t), list(self.L))

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = livePlotViewer()
    window.show()
    sys.exit(app.exec_())
