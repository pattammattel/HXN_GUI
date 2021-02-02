   
   
   
def generate_flyscan(self):

    from bluesky import RunEngine
    from ophyd.sim import det, motor1, motor2
    from bluesky.plans import count, scan, grid_scan

    RE = RunEngine({})

    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()

        # Send all metadata/data captured to the BestEffortCallback.
    RE.subscribe(bec)

        # Make plots update live while scans run.
    from bluesky.utils import install_kicker
    install_kicker()
    import matplotlib
    import PyQt5
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
        

    mot1_s = self.x_start.value()
    mot1_e = self.x_end.value()
    mot1_steps = self.mot1_num_steps.value() 
    mot2_s = self.y_start.value()
    mot2_e = self.y_end.value()
    mot2_steps = self.mot2_num_steps.value()
    dwell = self.dwell_2d.value()
    plt.figure()

    RE(grid_scan([det], motor1, mot1_s,mot1_e ,mot1_steps, motor2, mot2_s,mot2_e, mot2_steps, False))
    plt.show()

def calc_res(self):
    mot1_s = self.x_start.value()
    mot1_e = self.x_end.value()
    mot1_steps = self.mot1_num_steps.value()

    mot2_s = self.y_start.value()
    mot2_e = self.y_end.value()
    mot2_steps = self.mot2_num_steps.value()
    
    dwell = self.dwell_2d.value()
        
    res_x = (abs(mot1_s)+abs(mot1_e))/mot1_steps
    res_y = (abs(mot2_s)+abs(mot2_e))/mot2_steps
    self.Disc_Calc_Res.setText(str(res_x*1000,res_y*1000))

    tot_time = str(mot1_steps*mot2_steps*dwell/60)
    self.Dis_Calc_time.setText(tot_time)
