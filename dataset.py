import os
from PIL import Image
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
# from torchvision.transforms import functional as F


class CoData(data.Dataset):
    def __init__(self, img_root, img_size):

        class_list = os.listdir(img_root)
        self.size = [img_size, img_size]

        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))

        imgs = torch.Tensor(num, 3, self.size[0], self.size[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img

        return imgs, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


def get_loader(img_root, img_size, num_workers=4, pin=True):
    dataset = CoData(img_root, img_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader
