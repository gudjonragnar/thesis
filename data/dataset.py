import os
import torch
import numpy as np
from torch.tensor import Tensor
import params

from torch.utils.data import Dataset
from torchvision import utils, transforms
from torchvision.transforms import functional
from PIL import Image


class ClassificationDataset(Dataset):
    """ Dataset containing histological tissue scans """

    def __init__(
        self,
        root_dir: str,
        width: int = 27,
        height: int = 27,
        train: bool = True,
        shift=None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.train = train
        # Item has structure [img_path, x_coord, y_coord class] where dir_name is like 'img3'
        self.items = np.load(
            os.path.join(
                root_dir, "{}_list.npy".format("train" if self.train else "test")
            )
        )
        self.transform = transform
        self.shift = shift
        self.width = width
        self.height = height
        self.rotations = [0.0, 90.0, 180.0, 270.0]
        if not self.transform:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(hue=0.05, saturation=0.1, brightness=0.1),
                    transforms.ToTensor(),
                ]
            )
        offset_half = self.width // 2
        self.offset = np.array([offset_half, offset_half])
        self.pad = 25

    def __len__(self):
        return len(self.items)

    def load_img(self, dir_name) -> Tensor:
        img_name = os.path.join(self.root_dir, dir_name, "{}.bmp".format(dir_name))
        img_pil = Image.open(img_name)
        img_pil.load()
        img = transforms.Pad(self.pad).forward(img_pil)
        img_pil.close()
        return img

    def __getitem__(self, idx):
        dir_name, x, y, cell_class = self.items[idx]
        img_pil = self.load_img(dir_name)

        center = np.array([float(x), float(y)])
        if self.shift:
            center += np.random.uniform(0 - self.shift, self.shift, size=(2,)).astype(
                float
            )
        center = np.round_(center).astype(int) + self.pad

        left_top = center - self.offset
        left, top = left_top
        cropped_img = functional.crop(img_pil, top, left, self.height, self.width)

        cropped_img = functional.rotate(
            cropped_img, angle=np.random.choice(self.rotations)
        )

        if self.transform and self.train:
            net_input = self.transform(cropped_img)
        else:
            net_input = transforms.ToTensor()(cropped_img)

        target = torch.tensor(cell_class.astype(np.int16), dtype=torch.long)

        return (net_input, target)


class NEPValidationDataset(ClassificationDataset):
    def __init__(self, root_dir, d=2):
        super().__init__(root_dir, train=False)
        self.d = d
        self.shifts = [
            np.array([i, j])
            for i in range(-self.d, self.d + 1)
            for j in range(-self.d, self.d + 1)
        ]

    def __len__(self):
        return len(self.shifts) * len(self.items)

    def __getitem__(self, idx):
        _idx = idx // len(self.shifts)
        _idx_mod = idx % len(self.shifts)
        item = self.items[_idx]
        img_pil = self.load_img(item)

        center = item[1].astype(float) + self.shifts[_idx_mod] + self.pad

        left_top = np.round_(center - self.offset)
        left, top = left_top
        cropped_img = functional.crop(img_pil, top, left, self.height, self.width)
        if cropped_img.size[0] != self.width or cropped_img.size[1] != self.height:
            print(
                "Dimensions of crop is wrong! The dimensions are {},\n\tthe center is {} and left_top is {}.".format(
                    cropped_img.size, center, left_top
                )
            )

        net_input = transforms.ToTensor()(cropped_img)
        target = torch.tensor(item[2], dtype=torch.long)

        return net_input, target


if __name__ == "__main__":
    ds = ClassificationDataset(root_dir=params.root_dir)
    item = ds.__getitem__(np.random.randint(0, len(ds)))
    utils.save_image(item[0], fp="/Users/gudjonragnar/Desktop/test.png")
    print(item[1])
