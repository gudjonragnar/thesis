import scipy.io
import numpy as np 
import os
import params

from collections import defaultdict

def read_dict(filename):
    return np.load(filename).item()

root_dir = params.root_dir
dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
mat_names = ['others', 'fibroblast', 'epithelial', 'inflammatory']

classes = {cell_type: mat_names.index(cell_type) for cell_type in mat_names}
class_count = defaultdict(int)

dirs_train = []
dirs_test = []

# train_dict = defaultdict(list)
# test_dict = defaultdict(list)
train_list = []
test_list = []

counter = 0
test_counter = 0
multiplier = params.multiples
for j, d in enumerate(dirs):
    filename = d+'_{}.mat'
    mats = [scipy.io.loadmat(os.path.join(root_dir,d,filename.format(m)))['detection'] for m in mat_names]
    for i,m in enumerate(mats):
        for item in m:
#             if mat_names[i] == 'inflammatory':
#                 cla = 1
#             else:
#                 cla = 0
#             l = [d, item, cla]
            l = [d, item, classes[mat_names[i]]]
            if j < 80:
                class_count[i] += 1
                # class_count[cla] += 1
                class_count['total'] += 1
                # train_dict[counter] = l
                for _ in range(multiplier):
                    train_list.append(l)
                counter += 1
            else:
                # test_dict[test_counter] = l
                test_list.append(l)
                test_counter += 1

np.random.shuffle(train_list)
np.random.shuffle(test_list)

# np.save(os.path.join(root_dir,'train_dict.npy'), train_dict)
# np.save(os.path.join(root_dir,'test_dict.npy'), test_dict)
np.save(os.path.join(root_dir,'train_list.npy'), train_list)
np.save(os.path.join(root_dir,'test_list.npy'), test_list)
np.save(os.path.join(root_dir,'class_weights.npy'), class_count)
