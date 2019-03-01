import os
import torch
import torch.nn as nn
import imageio
import scipy.io
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from combine_class import read_dict

class ClassificationDataset(Dataset):
    """ Dataset containing histological tissue scans """

    def __init__(self, root_dir, train=True, shift=None, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.items = np.load(os.path.join(root_dir,'{}_dict.npy'.format('train' if self.train else 'test'))).item()
        # self.dirs = np.array([[d]*7 for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]).flatten()
        self.transform = transform
        self.shift = shift
        self.width = 27
        self.height = 27
        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_mean, _std)
            ])
        self.offset = np.array([13,13])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Item has structure [dir_name, coords, class] where dir_name is like 'img3'
        item = self.items[idx]
        dir_name = item[0]
        img_name = os.path.join(self.root_dir, dir_name,'{}.bmp'.format(dir_name))
        img_pil = Image.open(img_name)
        
        center = item[1].astype(float)
        if self.shift:
            center += np.random.uniform(0-self.shift,self.shift, size=(2,)).astype(float)
        center = np.round_(np.array(center)).astype(int)

        left_top = np.maximum.reduce([center - self.offset, np.zeros(2)])
        cropped_img = transforms.functional.crop(img_pil,*left_top[::-1], self.width, self.height)

        if self.transform:
            net_input = self.transform(cropped_img)

        target = torch.tensor(item[2], dtype=torch.long)

        return (net_input, target)

if __name__ == '__main__':
    ds = ClassificationDataset(root_dir='/Users/gudjonragnar/Documents/KTH/Thesis/CRCHistoPhenotypes_2016_04_28/Classification')
    item = ds.__getitem__(100)
    utils.save_image(item[0], filename='/Users/gudjonragnar/Desktop/test.png')
    print(item[1])




