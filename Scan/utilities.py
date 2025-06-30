import numpy as np
from contextlib import contextmanager
from functools import wraps
from PyQt5.QtWidgets import QMessageBox

@contextmanager
def try_ignored(*exceptions):

    """ 
    usage;
    with try_ignored(Exception):
        funct(*arg, **kwarg)

    """
    try:
        yield
    except exceptions:
        print(f"{exceptions} occured; passing to next step")
        pass

def show_error_message_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(None, "Error", error_message)
            return None
    return wrapper

def show_info_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(None, "Error", error_message)
            pass
    return wrapper


def show_confirm_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        choice = QMessageBox.question(None,'Detector Motion Warning',
                                      "Make sure this motion is safe. \n Move?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        
        if choice == QMessageBox.Yes:
            return func(*args, **kwargs)
        else:
            return
    return wrapper

def try_except_pass(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            pass
    return wrapper

@show_error_message_box
def parse_angle_range(str_scan_range,seperator = ":"):
    scan_numbers = []
    slist = str_scan_range.split(",")
    #print(slist)
    for item in slist:
        if seperator in item:
            slist_s, slist_e = item.split(seperator)[0],item.split(seperator)[1]
            
            if len(item.split(seperator)) == 2:
                spacing = int(abs(eval(slist_e)-eval(slist_s))+1)
                print(slist_s, slist_e,spacing)

            else:
                spacing = int(abs(eval(item.split(seperator)[2])+1))
                print(slist_s, slist_e,spacing)

            scan_numbers.extend(list(np.linspace(eval(slist_s),
                                    eval(slist_e), 
                                    spacing)))


        else:
            scan_numbers.append(eval(item))
    
    return scan_numbers

