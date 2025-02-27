import cv2 
from pathlib import Path 
import numpy as np

from dsec_det.dataset import DSECDet
from dsec_det.io import yaml_file_to_dict

from dsec_det.visualize import render_events_on_image 


split_config = yaml_file_to_dict(Path("./config/train_val_test_split.yaml"))

import os


dataset = DSECDet(root=Path(os.environ["DSEC"]),
                  split="train",
                  sync="back",
                  split_config=split_config,
                  debug=True)              # can be test/train/val
        

indices = np.load("empty_indices.npy")

print(len(indices))
