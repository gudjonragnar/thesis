import os
import torch
import torch.nn as nn
import imageio
import scipy.io
import numpy as np
import params

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from mark_centers import mark_cen

class ClassificationDataset(Dataset):
    """ Dataset containing histological tissue scans """

    def __init__(self, root_dir, width=27, height=27, train=True, shift=None, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.items = np.load(os.path.join(root_dir,'{}_list.npy'.format('train' if self.train else 'test')))  #.item()
        # self.dirs = np.array([[d]*7 for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]).flatten()
        self.transform = transform
        self.shift = shift
        self.width = width
        self.height = height
        self.rotations = [0, 90, 180, 270]
        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(hue=0.05, saturation=0.1, brightness=0.1),
                transforms.ToTensor()
                # transforms.Normalize(_mean, _std)
            ])
        offset_half = (self.width-1)/2 if self.width % 2 == 1 else self.width/2
        self.offset = np.array([offset_half, offset_half])
        self.pad = 25

    def __len__(self):
        return len(self.items)

    def load_img(self, item):
        dir_name = item[0]
        img_name = os.path.join(self.root_dir, dir_name,'{}.bmp'.format(dir_name))
        img_pil = Image.open(img_name)
        # mark_cen(img_pil,[item])
        img_pil = transforms.Pad(self.pad)(img_pil)
        return img_pil

    def __getitem__(self, idx):
        # Item has structure [dir_name, coords, class] where dir_name is like 'img3'
        item = self.items[idx]
        img_pil = self.load_img(item)
        
        center = item[1].astype(float)
        if self.shift:
            center += np.random.uniform(0-self.shift,self.shift, size=(2,)).astype(float)
        center = np.round_(np.array(center)).astype(int) + self.pad 

        # left_top = np.maximum.reduce([center - self.offset, np.zeros(2)])
        left_top = center-self.offset
        cropped_img = transforms.functional.crop(img_pil,*left_top[::-1], self.width, self.height)

        cropped_img = transforms.functional.rotate(cropped_img, np.random.choice(self.rotations))

        if self.transform and self.train:
            net_input = self.transform(cropped_img)
        else:
            net_input = transforms.ToTensor()(cropped_img)

        target = torch.tensor(item[2], dtype=torch.long)

        return (net_input, target)


class NEPValidationDataset(ClassificationDataset):
    
    def __init__(self, root_dir, d=2):
        super().__init__(root_dir, train=False)
        self.d = d
        self.shifts = [np.array([i,j]) for i in range(-self.d,self.d+1) for j in range(-self.d,self.d+1)]
    
    def __len__(self):
        return len(self.shifts)*len(self.items)
    
    def __getitem__(self, idx):
        _idx = idx//len(self.shifts)
        _idx_mod = idx % len(self.shifts)
        item = self.items[_idx]
        img_pil = self.load_img(item)

        center = item[1].astype(float) + self.shifts[_idx_mod] + self.pad
        
        left_top = np.round_(center-self.offset)
        cropped_img = transforms.functional.crop(img_pil,*left_top[::-1], self.width, self.height)
        if cropped_img.size[0] != self.width or cropped_img.size[1] != self.height:
            print("Dimensions of crop is wrong! The dimensions are {},\n\tthe center is {} and left_top is {}."
                .format(cropped_img.size, center, left_top))
        
        net_input = transforms.ToTensor()(cropped_img)
        target = torch.tensor(item[2], dtype=torch.long)

        return net_input, target






if __name__ == '__main__':
    ds = ClassificationDataset(root_dir=params.root_dir)
    item = ds.__getitem__(np.random.randint(0,len(ds)))
    utils.save_image(item[0], filename='/Users/gudjonragnar/Desktop/test.png')
    print(item[1])




