import time
import numpy as np
import threading
from threading import Thread
import pyqtgraph as pg
import PyQt5

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class MySensor():
    def get_position(self, mean=0.0, standard_dev=0.1):
        # Random sensor data
        return np.random.normal(mean, standard_dev, 1)[0]


class SignalCommunicate(PyQt5.QtCore.QObject):
    # https://stackoverflow.com/a/45620056
    got_new_sensor_data = PyQt5.QtCore.pyqtSignal(float, float)
    position_updated = PyQt5.QtCore.pyqtSignal(float)

    request_graph_update = PyQt5.QtCore.pyqtSignal()


class LiveSensorViewer():

    def __init__(self, sensor_update_interval=1):
        # super().__init__()

        # How frequently to get sensor data and update graph
        self.sensor_update_interval = sensor_update_interval

        # Init sensor object which gives live data
        self.my_sensor = MySensor()

        # Init with default values
        self.current_position = self.my_sensor.get_position(mean=0.0, standard_dev=0.1)
        self.current_position_timestamp = time.time()

        # Init array which stores sensor data
        self.log_time = [self.current_position_timestamp]
        self.log_position_raw = [self.current_position]
        self.moving_avg = 5

        # Define the array size on max amount of data to store in the list
        self.log_size = 1 * 60 * 1000 / self.sensor_update_interval

        # Setup the graphs which will display sensor data
        self.plot_widget = pg.GraphicsLayoutWidget(show=True)
        self.my_graph = self.plot_widget.addPlot(axisItems={'bottom': pg.DateAxisItem()})
        self.my_graph.showGrid(x=True, y=True, alpha=0.25)
        self.my_graph.addLegend()

        # Curves to be drawn on the graph
        self.curve_position_raw = self.my_graph.plot(self.log_time, self.log_position_raw, name='Position raw (mm)',
                                                     pen=pg.mkPen(color='#525252'))
        self.curve_position_moving_avg = self.my_graph.plot(self.log_time, self.log_position_raw,
                                                            name='Position avg. 5 periods (mm)',
                                                            pen=pg.mkPen(color='#FFF'))

        # A dialog box which displays the sensor value only. No graph.
        self.my_dialog = PyQt5.QtWidgets.QWidget()
        self.verticalLayout = PyQt5.QtWidgets.QVBoxLayout(self.my_dialog)

        self.my_label = PyQt5.QtWidgets.QLabel()
        self.verticalLayout.addWidget(self.my_label)
        self.my_label.setText('Current sensor position:')

        self.my_sensor_value = PyQt5.QtWidgets.QDoubleSpinBox()
        self.verticalLayout.addWidget(self.my_sensor_value)
        self.my_sensor_value.setDecimals(6)

        self.my_dialog.show()

        # Signals that can be emitted
        self.signalComm = SignalCommunicate()
        # Connect the signal 'position_updated' to the QDoubleSpinBox
        self.signalComm.position_updated.connect(self.my_sensor_value.setValue)
        # Update graph whenever the 'request_graph_update' signal is emitted
        self.signalComm.request_graph_update.connect(self.update_graph)

        # Setup thread which will continuously query the sensor for data
        self.position_update_thread = Thread(target=self.read_position,
                                             args=(self.my_sensor, self.sensor_update_interval))
        self.position_update_thread.daemon = True
        self.position_update_thread.start()  # Start the thread to query sensor data

    def read_position(self, sensor_obj, update_interval):
        # print('Thread ={}          Function = read_position()'.format(threading.currentThread().getName()))

        # This function continuously runs in a seprate thread to continuously query the sensor for data

        sc = SignalCommunicate()
        sc.got_new_sensor_data.connect(self.handle_sensor_data)

        while True:
            # Get data and timestamp from sensor
            new_pos = sensor_obj.get_position(mean=0.0, standard_dev=0.1)
            new_pos_time = time.time()

            # Emit signal with sensor data and  timestamp
            sc.got_new_sensor_data.emit(new_pos, new_pos_time)

            # Wait before querying the sensor again
            time.sleep(update_interval / 1000)

    def handle_sensor_data(self, new_pos, new_pos_time):
        print('Thread ={}          Function = handle_sensor_data()'.format(threading.currentThread().getName()))
        # Get the sensor position/timestamp emitted from the separate thread
        self.current_position_timestamp = new_pos_time
        self.current_position = new_pos

        # Emit a singal with new position info
        self.signalComm.position_updated.emit(self.current_position)

        # Add data to log array
        self.log_time.append(self.current_position_timestamp)
        if len(self.log_time) > self.log_size:
            # Append new data to the log and remove old data to maintain desired log size
            self.log_time.pop(0)

        self.log_position_raw.append(self.current_position)
        if len(self.log_position_raw) > self.log_size:
            # Append new data to the log and remove old data to maintain desired log size
            self.log_position_raw.pop(0)

        if len(self.log_time) <= self.moving_avg:
            # Skip calculating moving avg if only 10 data points collected from sensor to prevent errors
            return
        else:
            self.calculate_moving_avg()

        # Request a graph update
        # self.update_graph()                     # Uncomment this if you want update_graph() to run in the same thread as handle_sensor_data() function

        # Emitting this signal ensures update_graph() will run in the main thread since the signal was connected in the __init__ function (main thread)
        self.signalComm.request_graph_update.emit()

    def calculate_moving_avg(self):
        print('Thread ={}          Function = calculate_moving_avg()'.format(threading.currentThread().getName()))
        # Get moving average of the position
        self.log_position_moving_avg = moving_average(self.log_position_raw, self.moving_avg)

    def update_graph(self):
        print('Thread ={}          Function = update_graph()'.format(threading.currentThread().getName()))
        self.curve_position_raw.setData(self.log_time, self.log_position_raw)
        self.curve_position_moving_avg.setData(self.log_time[:len(self.log_position_moving_avg)], self.log_position_moving_avg)


if __name__ == '__main__':
    import sys
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    z = LiveSensorViewer()

    app.exec_()
    sys.exit(app.exec_())