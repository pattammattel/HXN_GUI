#!/usr/bin/env python
"""
Launcher script for HXN GUI that works in both standalone and IPython environments.
"""
import sys
import os

# Ensure matplotlib uses Qt backend before any Qt imports
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
if 'MPLBACKEND' not in os.environ:
    os.environ['MPLBACKEND'] = 'qtagg'

def launch_gui():
    """Launch the HXN GUI, handling both standalone and IPython environments."""
    from PySide6 import QtWidgets
    
    # Check if we're in IPython
    try:
        __IPYTHON__
        in_ipython = True
    except NameError:
        in_ipython = False
    
    # Get or create QApplication instance
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True
    else:
        created_app = False
        print("Using existing QApplication instance")
    
    # Import and create the GUI window
    from hxn_gui_v3 import Ui
    window = Ui()
    window.show()
    
    # Only start event loop if we created a new app and not in IPython
    if created_app and not in_ipython:
        try:
            sys.exit(app.exec())
        except SystemExit:
            pass
    
    return window

if __name__ == "__main__":
    window = launch_gui()
