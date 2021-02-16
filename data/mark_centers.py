from typing import List, Tuple
import numpy as np
from PIL import Image
import os
from torchvision.transforms.functional import crop
import scipy.io
import params
import sys

"""
A script to mark centers in an image
"""

root_dir = params.root_dir

til = Image.new("RGB", (2, 2), (250, 0, 0))


def load(img_name: str) -> Tuple[Image.Image, list]:
    mat_names = ["others", "fibroblast", "epithelial", "inflammatory"]
    mats = [
        scipy.io.loadmat(
            os.path.join(root_dir, img_name, (img_name + "_{}").format(m))
        )["detection"]
        for m in mat_names
    ]
    img = Image.open(os.path.join(root_dir, img_name, img_name + ".bmp"))
    centers = []
    for mat in mats:
        for center in mat:
            centers.append(center)
    return img, centers


def mark_cen(img: Image.Image, centers: List[np.float64]) -> None:
    for center in centers:
        c_tuple = tuple(np.round_(center, decimals=0).astype(int))
        img.paste(til, c_tuple)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Supply the image name as an argument")
        exit()
    img, centers = load(sys.argv[1])
    mark_cen(img, centers)
    img.show()
