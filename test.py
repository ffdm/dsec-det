import cv2 
from pathlib import Path 
import numpy as np

from dsec_det.dataset import DSECDet
from dsec_det.io import yaml_file_to_dict

from dsec_det.visualize import render_events_on_image 


split_config = yaml_file_to_dict(Path("./config/train_val_test_split.yaml"))

import os

"""
dataset = DSECDet(root=Path(os.environ["DSEC"]),
                  split="test",              # can be test/train/val
                  sync="back",               # load 50 ms of event after ('back'), or before  ('front') the image
                  split_config=split_config, # which sequences go into train/val/test. See yaml file for details.
                  debug=True)                # generate debug output, available in output['debug']
"""
"""
dataset = DSECDet(root=Path(os.environ["DSEC"]),
                  split="train",              # can be test/train/val
                  sync="back",               # load 50 ms of event after ('back'), or before  ('front') the image
                  debug=True)                # generate debug output, available in output['debug']
"""

dataset = DSECDet(root=Path(os.environ["DSEC"]),
                  split="train",
                  sync="back",
                  split_config=split_config,
                  debug=True)              # can be test/train/val
        

indices = np.zeros(0)
for i in range(0, len(dataset)):
    if (len(dataset[i]['tracks']) == 0):
        indices = np.append(indices, i)

np.save("empty_indices.npy", indices)



"""
print(len(dataset))
#dataset[15091]

print(len(dataset.directories))

for key in dataset.directories:
    print(key)

black_image = np.zeros((480,640,3))

events = output['events']

#print(type(events['t'][0]))
print(1/((events['t'][-1] - events['t'][0])*1e-6))

for key in output:
    print(key)

"""
"""
for i in range(0, 150):
    output = dataset[i]
    print(output['tracks'].size)
"""

"""
maximum = 0
prev = events['t'][0]
for t in events['t']:
    diff = t - prev
    prev = t
    maximum = max(maximum, diff)

print(f'Max time step between events: {maximum}')
print(events['t'][300:310])
"""
    

"""
sum = 0
count = 0
for index in range(200, 400):
    sum += len(dataset[index]['tracks']['t'])
    count+=1 

print(f"Avg detections per fram: {sum/count}")


cv2.imshow("meow", render_events_on_image(black_image, x=events['x'], y=events['y'],
        p=events['p']))
cv2.waitKey(0)

print("MEOW")
print(type(output['image']))
print(output['image'].shape)
#im = cv2.imread(str(""))

print(len(output['events']['p']))
for key in output['events']:
    print(key)
"""


"""
print(len(dataset))
for key in output:
    print("---------")
    out = output[key]
    print(key)
    print(type(out))
    print(len(out))

for i in range(5000, 5125):
    cv2.imshow("Image", dataset[i]['debug'])
    cv2.waitKey(0)
    track = dataset[i]['tracks']
    #print(len(track['t']))
"""

#cv2.imshow("Debug", output['debug'])
#cv2.waitKey(0)

