import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,QMessageBox
from PyQt5 import QtWidgets, uic,QtTest
from functools import wraps
from utilities import *

ui_path = os.path.dirname(os.path.abspath(__file__))
param_file_to_run = "/nsls2/data/hxn/legacy/user_macros/HXN_GUI/Scan/temp_files/mll_tomo_params.json"



class MLLTomoGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MLLTomoGUI, self).__init__()
        uic.loadUi(os.path.join(ui_path,"ui_files/mll_tomo_ui_.ui"), self)
        self.active_folder = "/nsls2/data/hxn/legacy/user_macros"

        #connections
        self.pb_import_params.clicked.connect(lambda:self.import_parameter_file())
        self.pb_export_params.clicked.connect(lambda:self.export_parameter_file())
        self.pb_update_scan.clicked.connect(lambda:self.export_parameter_file(auto = True))
        self.pb_abort_scan.clicked.connect(lambda:self.update_bool_choices(update = "stop_iter"))
        self.pb_pause_scan.clicked.connect(lambda:self.update_bool_choices(update = "pause_scan"))
        self.pb_stop_pdf.clicked.connect(lambda:self.update_bool_choices(update = "stop_pdf"))
        self.pb_resume_scan.clicked.connect(lambda:self.update_bool_choices(update = "resume_scan"))
        self.pb_start_pdf.clicked.connect(lambda:self.update_bool_choices(update = "start_pdf"))
        self.pb_start_scan.clicked.connect(lambda: self.run_tomo())
        #self.pb_start_scan.clicked.connect(lambda: self.send_tomo_to_queue())
        
        #open the last file to fill in paramaters
        self.display_imported_parameters(param_file_to_run)

    @show_error_message_box
    def send_tomo_to_queue(self):
        RM.item_add((BPlan("mll_tomo_json",
                    param_file_to_run)))

    
    @show_error_message_box
    def run_tomo(self):
        RE((mll_tomo_json(param_file_to_run)))


    @show_error_message_box
    def import_parameter_file(self):
        
        # Open a file dialog to select a JSON file
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open JSON File', '', 'JSON Files (*.json)')

        if file_name:
            self.json_param_file = file_name
            self.display_imported_parameters(self.json_param_file)
            self.export_parameter_file(auto = True)

    

    @show_error_message_box
    def display_imported_parameters(self,json_file):

        # Read the contents of the JSON file and display them in the text edit widget
        with open(json_file, 'r') as f:
                self.param_dict = json.load(f)
        
        #angle info
        self.sb_start_angle.setValue(self.param_dict["angle_info"]["start"])
        self.sb_end_angle.setValue(self.param_dict["angle_info"]["end"])
        self.sb_angle_num.setValue(self.param_dict["angle_info"]["num"])

        #fly2d info
        self.sb_xstart.setValue(self.param_dict["fly2d_scan"]["x_start"])
        self.sb_xend.setValue(self.param_dict["fly2d_scan"]["x_end"])
        self.sb_x_num.setValue(self.param_dict["fly2d_scan"]["x_num"])
        self.sb_ystart.setValue(self.param_dict["fly2d_scan"]["y_start"])
        self.sb_yend.setValue(self.param_dict["fly2d_scan"]["y_end"])
        self.sb_y_num.setValue(self.param_dict["fly2d_scan"]["y_num"])
        self.sb_exposure.setValue(self.param_dict["fly2d_scan"]["exposure"])

        #x align
        self.cb_do_xalign.setChecked(self.param_dict["xalign"]["do_align"])
        self.sb_xalign_start.setValue(self.param_dict["xalign"]["start"])
        self.sb_xalign_end.setValue(self.param_dict["xalign"]["end"])
        self.sb_xalign_num.setValue(self.param_dict["xalign"]["num"])
        self.sb_xalign_exposure.setValue(self.param_dict["xalign"]["exposure"])
        self.le_xalign_elem.setText(self.param_dict["xalign"]["elem"])
        self.cb_xalign_center_with.setCurrentText(self.param_dict["xalign"]["center_with"])
        self.sb_xalign_threshold.setValue(self.param_dict["xalign"]["threshold"])
        self.dsb_xalign_offset.setValue(self.param_dict["xalign"]["offset"])

        #y align
        self.cb_do_yalign.setChecked(self.param_dict["yalign"]["do_align"])
        self.sb_yalign_start.setValue(self.param_dict["yalign"]["start"])
        self.sb_yalign_end.setValue(self.param_dict["yalign"]["end"])
        self.sb_yalign_num.setValue(self.param_dict["yalign"]["num"])
        self.sb_yalign_exposure.setValue(self.param_dict["yalign"]["exposure"])
        self.le_yalign_elem.setText(self.param_dict["yalign"]["elem"])
        self.cb_yalign_center_with.setCurrentText(self.param_dict["yalign"]["center_with"])
        self.sb_xalign_threshold.setValue(self.param_dict["yalign"]["threshold"])
        self.dsb_yalign_offset.setValue(self.param_dict["yalign"]["offset"])
        
        #align 2d
        self.cb_do_align2d.setChecked(self.param_dict["align_2d_com"]["do_align"])
        self.sb_align2d_x_start.setValue(self.param_dict["align_2d_com"]["x_start"])
        self.sb_align2d_x_end.setValue(self.param_dict["align_2d_com"]["x_end"])
        self.sb_align2d_x_num.setValue(self.param_dict["align_2d_com"]["x_num"])
        self.sb_align2d_y_start.setValue(self.param_dict["align_2d_com"]["y_start"])
        self.sb_align2d_y_end.setValue(self.param_dict["align_2d_com"]["y_end"])
        self.sb_align2d_y_num.setValue(self.param_dict["align_2d_com"]["y_num"])
        self.sb_align2d_exposure.setValue(self.param_dict["align_2d_com"]["exposure"])
        self.le_align2d_elem.setText(self.param_dict["align_2d_com"]["elem"])
        self.sb_align2d_threshold.setValue(self.param_dict["align_2d_com"]["threshold"])
        self.cb_align2d_move_x.setChecked(self.param_dict["align_2d_com"]["move_x"])
        self.cb_align2d_move_y.setChecked(self.param_dict["align_2d_com"]["move_y"])

        print(self.param_dict["add_angles"])
        print(self.param_dict["remove_angles"])

        #add/remove angles
        if self.param_dict["add_angles"] == None:
            self.le_add_angles.setText("None")
        else:
            self.le_add_angles.setText(str(self.param_dict["add_angles"])[1:-1])
            

        if self.param_dict["remove_angles"] == None:
            self.le_remove_angles.setText("None")
        else:   
            self.le_remove_angles.setText(str(self.param_dict["remove_angles"])[1:-1])

    @show_error_message_box
    def update_parameter_dict(self):

        #angle info
        self.param_dict["angle_info"]["start"] = self.sb_start_angle.value()
        self.param_dict["angle_info"]["end"] = self.sb_end_angle.value()
        self.param_dict["angle_info"]["num"] = self.sb_angle_num.value()

        #fly2d info
        self.param_dict["fly2d_scan"]["x_start"] = self.sb_xstart.value()
        self.param_dict["fly2d_scan"]["x_end"] = self.sb_xend.value()
        self.param_dict["fly2d_scan"]["x_num"] = self.sb_x_num.value()
        self.param_dict["fly2d_scan"]["y_start"] = self.sb_ystart.value()
        self.param_dict["fly2d_scan"]["y_end"] = self.sb_yend.value()
        self.param_dict["fly2d_scan"]["y_num"] = self.sb_y_num.value()
        self.param_dict["fly2d_scan"]["exposure"] = self.sb_exposure.value()

        #x align
        self.param_dict["xalign"]["do_align"] = self.cb_do_xalign.isChecked()
        self.param_dict["xalign"]["start"] = self.sb_xalign_start.value()
        self.param_dict["xalign"]["end"] = self.sb_xalign_end.value()
        self.param_dict["xalign"]["num"] = self.sb_xalign_num.value()
        self.param_dict["xalign"]["exposure"] = self.sb_xalign_exposure.value()
        self.param_dict["xalign"]["elem"] = self.le_xalign_elem.text()
        self.param_dict["xalign"]["center_with"] = self.cb_xalign_center_with.currentText()
        self.param_dict["xalign"]["threshold"] = self.sb_xalign_threshold.value()
        self.param_dict["xalign"]["offset"] = self.dsb_xalign_offset.value()

        #y align
        self.param_dict["yalign"]["do_align"] = self.cb_do_yalign.isChecked()
        self.param_dict["yalign"]["start"] = self.sb_yalign_start.value()
        self.param_dict["yalign"]["end"] = self.sb_yalign_end.value()
        self.param_dict["yalign"]["num"] = self.sb_yalign_num.value()
        self.param_dict["yalign"]["exposure"] = self.sb_yalign_exposure.value()
        self.param_dict["yalign"]["elem"] = self.le_yalign_elem.text()
        self.param_dict["yalign"]["center_with"] = self.cb_yalign_center_with.currentText()
        self.param_dict["yalign"]["threshold"] = self.sb_xalign_threshold.value()
        self.param_dict["yalign"]["offset"] = self.dsb_yalign_offset.value()
        
        #align 2d
        self.param_dict["align_2d_com"]["do_align"] = self.cb_do_align2d.isChecked()
        self.param_dict["align_2d_com"]["x_start"] = self.sb_align2d_x_start.value()
        self.param_dict["align_2d_com"]["x_end"] = self.sb_align2d_x_end.value()
        self.param_dict["align_2d_com"]["x_num"] = self.sb_align2d_x_num.value()
        self.param_dict["align_2d_com"]["y_start"] = self.sb_align2d_y_start.value()
        self.param_dict["align_2d_com"]["y_end"] = self.sb_align2d_y_end.value()
        self.param_dict["align_2d_com"]["y_num"] = self.sb_align2d_y_num.value()
        self.param_dict["align_2d_com"]["exposure"] = self.sb_align2d_exposure.value()
        self.param_dict["align_2d_com"]["elem"] = self.le_align2d_elem.text()
        self.param_dict["align_2d_com"]["threshold"] = self.sb_align2d_threshold.value()
        self.param_dict["align_2d_com"]["move_x"] = self.cb_align2d_move_x.isChecked()
        self.param_dict["align_2d_com"]["move_y"] = self.cb_align2d_move_y.isChecked()


        #add/remove angles 
        #TODO change to parse angle list

        print(self.le_remove_angles.text())

        if self.le_add_angles.text():
            self.param_dict["add_angles"] = parse_angle_range(self.le_add_angles.text())
        else:
            self.param_dict["add_angles"]=None
        if self.le_remove_angles.text():
            self.param_dict["remove_angles"] = parse_angle_range(self.le_remove_angles.text())
        else:
            self.param_dict["remove_angles"]=None

        
    @show_error_message_box
    def export_parameter_file(self, auto = False):

        self.update_parameter_dict()

        if auto:
            save_file_path = param_file_to_run
            # write the data to the temp location as JSON
            if save_file_path:
                with open(save_file_path, 'w') as outfile:
                    json.dump(self.param_dict, outfile, indent=6)
            
        
        else:
            #manual saving
            # Open a file dialog to get the save location
            save_file_path, _ = QFileDialog.getSaveFileName(self,"Save JSON file", self.active_folder, "JSON Files (*.json)")
            
            # If a file was selected, write the data to the file as JSON
            if save_file_path:
                self.active_folder = save_file_path
                with open(save_file_path, 'w') as outfile:
                    json.dump(self.param_dict, outfile, indent=6)

    @show_error_message_box
    def update_bool_choices(self, update = "stop_iter"):

        if update == "resume_scan":
            self.param_dict["pause_scan"] = False

        elif update == "start_pdf":
            self.param_dict["stop_pdf"] = False

        else:
            self.param_dict[update] = True
        
        self.statusbar.showMessage(f"{update} change applied")
        self.export_parameter_file(auto = True)

    @show_error_message_box
    def update_angles(self):
        
        #self.param_dict["add_angles"] = list(self.le_add_angles.text())
        #self.param_dict["remove_angles"] = list(self.le_add_angles.text())

        self.param_dict["add_angles"] = parse_angle_range(self.le_add_angles.text())
        self.param_dict["remove_angles"] = parse_angle_range(self.le_add_angles.text())


    def closeEvent(self,event):
        
        reply = QMessageBox.question(self, 'Quit GUI', "Are you sure you want to close the window?")

        if reply == QMessageBox.Yes:
            if RE.state=="idle":
                plt.close('all')
                QtTest.QTest.qWait(1000)
                event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication([sys.argv]
)
    widget = MLLTomoGUI()
    widget.show()
    sys.exit(app.exec_())
