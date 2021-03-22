roi_list = {}


roi = {
    'zpssx': 'fx', 'zpssy': 'fy', 'zpssz': 'fz'
}

roi2 = {
    'zpssx': 'fx2', 'zpssy': 'fy2', 'zpssz': 'fz2'
}

roi_list['roi_1'] = roi
roi_list['roi_2'] = roi2

print(roi_list['roi_1'])