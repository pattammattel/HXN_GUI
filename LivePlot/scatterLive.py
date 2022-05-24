import matplotlib as mpl
import matplotlib.cm as cm
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)


win = QtGui.QWidget()
win.setWindowTitle('pyqtgraph example: ScatterPlotSpeedTest')
ui = Ui_Form()
ui.setupUi(win)
win.show()

p = ui.plot
p.setRange(xRange=[-500, 500], yRange=[-500, 500])

data = np.random.normal(size=(50,10000), scale=100)
sizeArray = (np.random.random(500) * 20.).astype(int)
ptr = 0
lastTime = time()
fps = None
def update():
    global curve, data, ptr, p, lastTime, fps
    p.clear()
    if ui.randCheck.isChecked():
        size = sizeArray
    else:
        size = ui.sizeSpin.value()


    z=np.random.randint(0,10000,10000)
    norm = mpl.colors.Normalize(vmin=min(z), vmax=max(z))
    m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors =  m.to_rgba(z, bytes=True)

    colors_=[]
    for i in colors:
      colors_.append(pg.mkBrush(tuple(i)))
      #colors.append(pg.intColor(np.random.randint(0,255), 100))

    curve = pg.ScatterPlotItem(x=data[ptr%50], y=data[(ptr+1)%50],
                   pen='w', brush=colors_, size=size,
                   pxMode=ui.pixelModeCheck.isChecked())


    '''
    curve = pg.ScatterPlotItem(pen='w', size=size, pxMode=ui.pixelModeCheck.isChecked())
    spots3=[]

    for i,j,k in zip(data[ptr%50],data[(ptr+1)%50],colors):
      spots3.append({'pos': (i, j), 'brush':pg.mkBrush(tuple(k))})

    curve.addPoints(spots3)
    '''



    p.addItem(curve)
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    p.repaint()
    #app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()