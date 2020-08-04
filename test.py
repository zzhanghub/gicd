from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import os
import argparse
import numpy as np


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def main(args):
    test_loader = get_loader(args.input_root,
                             args.size,
                             num_workers=8,
                             pin=True)

    # Init model
    device = torch.device("cuda")
    exec('from models import ' + args.model)
    model = eval(args.model + '()')
    model = model.to(device)
    ginet_dict = torch.load(args.param_path)
    model.to(device)
    model.ginet.load_state_dict(ginet_dict)

    model.eval()

    tensor2pil = transforms.ToPILImage()

    for batch in tqdm(test_loader):
        inputs = batch[0].to(device)
        subpaths = batch[1]
        ori_sizes = batch[2]

        scaled_preds = model(inputs)

        os.makedirs(os.path.join(args.save_root, subpaths[0][0].split('/')[0]),
                    exist_ok=True)
        num = len(scaled_preds)
        for inum in range(num):
            subpath = subpaths[inum][0]
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            res = nn.functional.interpolate(scaled_preds[inum][-1],
                                            size=ori_size,
                                            mode='bilinear',
                                            align_corners=True)
            save_tensor_img(res, os.path.join(args.save_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='GICD', type=str)
    parser.add_argument('--input_root', type=str, help="Your dataset")
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--param_path',
                        default='./gicd_ginet.pth',
                        type=str,
                        help='model folder')
    parser.add_argument('--save_root',
                        default='./SalMaps/pred',
                        type=str,
                        help='Output folder')
    args = parser.parse_args()

    main(args)
