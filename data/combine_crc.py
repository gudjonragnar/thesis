import scipy.io
import numpy as np
import os
from params import sccnn_params as params

from collections import defaultdict

"""
This script creates a train and test dataset split for the CRC dataset.

It creates two files 'train_list.npy' and 'test_list.npy' that each contains a numpy array.
Each row in the array has format [dir_name, x_coord, y_coord, class] (dir_name is img1, img2, etc.).

It also writes the class weights that can be used to normalize the classes during training.
"""

root_dir = params.crc_root_dir
if not root_dir:
    raise Exception("CRC_ROOT_DIR env variable is missing")
dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
mat_names = ["others", "fibroblast", "epithelial", "inflammatory"]

classes = {cell_type: mat_names.index(cell_type) for cell_type in mat_names}
class_count = defaultdict(int)

dirs_train = []
dirs_test = []

train_list = []
test_list = []

counter = 0
test_counter = 0
multiplier = params.multiples
for j, dir in enumerate(dirs):
    # Iterate through all the images (img1, img2, ...)
    mats = [
        scipy.io.loadmat(os.path.join(root_dir, dir, f"{dir}_{mat}.mat"))["detection"]
        for mat in mat_names
    ]
    img_path = f"{root_dir}/{dir}/{dir}.bmp"
    for i, mat in enumerate(mats):
        # Iterate through the different cell types
        for center in mat:
            # Add each nucleus to a dataset
            sample = [img_path, *center, classes[mat_names[i]]]
            if j < 80:  # 80/20 split
                # TODO: Do this splitting properly (shuffle and then split)
                class_count[i] += 1
                class_count["total"] += 1
                for _ in range(multiplier):
                    train_list.append(np.array(sample))
                counter += 1
            else:
                test_list.append(np.array(sample))
                test_counter += 1


np.random.shuffle(train_list)
np.random.shuffle(test_list)

train_stack = np.stack(train_list)
test_stack = np.stack(test_list)

np.save(os.path.join(root_dir, "train_list.npy"), train_stack)
np.save(os.path.join(root_dir, "test_list.npy"), test_stack)
np.save(os.path.join(root_dir, "class_weights.npy"), class_count)
