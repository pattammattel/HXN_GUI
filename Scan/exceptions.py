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
            pass
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
