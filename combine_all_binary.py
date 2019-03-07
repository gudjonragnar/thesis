import scipy.io
import numpy as np 
import os

from collections import defaultdict

def read_dict(filename):
    return np.load(filename).item()

root_dir = '/Users/gudjonragnar/Documents/KTH/Thesis/CRCHistoPhenotypes_2016_04_28/Classification'
dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
mat_names = ['others', 'fibroblast', 'epithelial', 'inflammatory']

classes = {cell_type: mat_names.index(cell_type) for cell_type in mat_names}
class_count = defaultdict(int)

dirs_train = []
dirs_test = []

train_dict = defaultdict(list)
test_dict = defaultdict(list)

others_counter = 0
counter = 0
test_counter = 0
for j, d in enumerate(dirs):
    filename = d+'_{}.mat'
    mats = [scipy.io.loadmat(os.path.join(root_dir,d,filename.format(m)))['detection'] for m in mat_names]
    for i,m in enumerate(mats):
        for item in m:
            l = [d, item, 0 if i<2 else 1]
            if l[2] == 0:
                others_counter += 1
            if j < 80:
                class_count[i] += 1
                class_count['total'] += 1
                train_dict[counter] = l
                counter += 1
            else:
                test_dict[test_counter] = l
                test_counter += 1


np.save(os.path.join(root_dir,'train_dict.npy'), train_dict)
np.save(os.path.join(root_dir,'test_dict.npy'), test_dict)
np.save(os.path.join(root_dir,'class_weights.npy'), class_count)
# print(others_counter)