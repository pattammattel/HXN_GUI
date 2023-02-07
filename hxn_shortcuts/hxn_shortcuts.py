

import sys, os, logging

from PyQt5.QtCore import QObject, QThread
from PyQt5 import QtWidgets, uic, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog

logger = logging.getLogger()
ui_path = os.path.dirname(os.path.abspath(__file__))


class HXN_Shortcuts(QtWidgets.QMainWindow):
    def __init__(self):
        super(HXN_Shortcuts, self).__init__()
        uic.loadUi(os.path.join(ui_path, "ui.ui"), self)

        self.button_to_command_dict = {
                                        self.pb_bsui: 'bsui',
                                        self.pb_bsui_gui: 'run-hxn-gui',
                                        self.pb_pyxrf: 'run-pyxrf',
                                        self.pb_pyxrf_batch: 'run-pyxrf-batch',
                                        self.pb_dpc: 'run-dpc',
                                        self.pb_ptycho: 'run-ptycho',
                                        self.pb_ptycho_save: 'run-ptycho-save',
                                        self.pb_probe_prop: 'run-probe-prop',
                                      }
        #not working
        #[key.clicked.connect(lambda key = key: os.system(f'gnome-terminal --tab --command {value} --active'),) for key,value in self.button_to_command_dict.items()]
        
        self.pb_bsui.clicked.connect(lambda: os.system(f'gnome-terminal -t "HXN BSUI" --tab -e bsui --active'))
        self.pb_bsui_gui.clicked.connect(lambda: os.system(f'gnome-terminal -t "HXN BSUI-GUI"  --tab -e run-hxn-gui --active'))
        self.pb_pyxrf.clicked.connect(lambda: os.system(f'gnome-terminal -t "PyXRF" --tab  -e run-pyxrf --active'))
        self.pb_pyxrf_batch.clicked.connect(lambda: os.system(f'gnome-terminal -t "PyXRF-Batch" --tab -e run-pyxrf-batch --active'))
        self.pb_dpc.clicked.connect(lambda: os.system(f'gnome-terminal --tab -t "DPC" -e run-dpc --active'))
        self.pb_ptycho_save.clicked.connect(lambda: os.system(f'gnome-terminal -t "ptycho-save" --tab -e run-ptycho-save --active'))
    


        self.pb_xmidas.clicked.connect(lambda: os.system(f'gnome-terminal -t "XMIDAS" --tab -e run-midas --active'))
        self.pb_ptycho.clicked.connect(lambda: os.system(f'gnome-terminal -t "ptycho-gui" --tab -e run-ptycho --active'))
        self.pb_css.clicked.connect(lambda: os.system(f'run-css'))
        self.pb_imagej.clicked.connect(lambda: os.system(f'gnome-terminal -t "imagej" --tab -e imagej --active'))
        self.pb_screenshot.clicked.connect(lambda: os.system(f'gnome-screenshot -i'))



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = HXN_Shortcuts()
    window.show()
    sys.exit(app.exec_())
