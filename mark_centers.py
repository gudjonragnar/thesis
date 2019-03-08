import numpy as np
from PIL import Image
import os
from torchvision.transforms.functional import crop
import scipy.io
import params

root_dir=params.root_dir

til = Image.new('RGB',(2,2), (250,0,0))

def load(img_name):
    mat_names = ['others', 'fibroblast', 'epithelial', 'inflammatory']
    mats = [scipy.io.loadmat(os.path.join(root_dir,img_name,(img_name+"_{}").format(m)))['detection'] for m in mat_names]
    img = Image.open(os.path.join(root_dir,img_name,img_name+'.bmp'))
    centers = []
    for m in mats:
        for item in m:
            centers.append(['d',item,1])
    return img, centers

def mark_cen(img,centers):
    for c in centers:
        c_tuple = tuple(np.round_(c[1],decimals=0).astype(int))
        img.paste(til, c_tuple)

def crop_(img, center, offset=np.array([13,13]), size=[27,27]):
    rounded_center = np.round_(center,decimals=0).astype(int)
    left_top = np.maximum.reduce([rounded_center - offset, np.zeros(2)])
    cropped_1 = crop(img,*left_top,*size)
    cropped_2 = crop(img,*left_top[::-1],*size)
    return cropped_1, cropped_2

if __name__ == '__main__':
    img, centers = load('img8')
    mark_cen(img, centers)
    img.show()








