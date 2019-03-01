import numpy as np
from PIL import Image
import os
from torchvision.transforms.functional import crop

root_dir='/Users/gudjonragnar/Documents/KTH/Thesis/CRCHistoPhenotypes_2016_04_28/Classification'

til = Image.new('RGB',(2,2), (250,0,0))

def load(img_name):
    img = Image.open(os.path.join(root_dir,img_name,img_name+'.bmp'))
    centers = np.load(os.path.join(root_dir,img_name,img_name+'.npy')).item()
    return img, centers

def mark_centers(img,centers):
    for c, cla in centers.items():
        c_tuple = tuple(np.round_(c,decimals=0).astype(int))
        img.paste(til, c_tuple)

def crop_(img, center, offset=np.array([13,13]), size=[27,27]):
    rounded_center = np.round_(center,decimals=0).astype(int)
    left_top = np.maximum.reduce([rounded_center - offset, np.zeros(2)])
    cropped_1 = crop(img,*left_top,*size)
    cropped_2 = crop(img,*left_top[::-1],*size)
    return cropped_1, cropped_2

img, centers = load('img8')
mark_centers(img, centers)
# img.show()

for key, val in centers.items():
    c = key
    break

c1, c2 = crop_(img, c)






