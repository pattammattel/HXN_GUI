roi_list = {}


roi = {
    'zpssx': 'fx', 'zpssy': 'fy', 'zpssz': 'fz'
}

roi2 = {
    'zpssx': 'fx2', 'zpssy': 'fy2', 'zpssz': 'fz2'
}

roi_list['roi_1'] = roi
roi_list['roi_2'] = roi2

import json

#with open('data.json', 'w') as fp:
    #json.dump(roi_list, fp, indent=4)

with open('data.json', 'r') as roi_:
    data = json.load(roi_)

print(data)


