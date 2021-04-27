import numpy as np
image_dict ={}
file = np.load('mulicolor_view.npz', allow_pickle=True)
for item in file:
    print(item)
    for inner_item in file[item]:
        print(inner_item)
