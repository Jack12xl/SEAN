"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os, cv2, torch
import util.util as util

from torch.nn import functional as F
from torchvision import utils

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel

from data.base_dataset import get_params, get_transform
from PIL import Image

import numpy as np
def labels_to_num(labels):
    number = 0
    for item in labels:
            number += 2**(item+1)
    return number

def load_style(batch):
    index = np.arange(len(styleList))
    np.random.shuffle(index)

    styles,labels, path = [],[],[]
    for i in range(batch):
        image_path = os.path.join(styleImg_root,styleList[index[i]])
        seg_path = os.path.join(styleSeg_root,styleList[index[i]][:-4]+'.png')

        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = transform_image(image).unsqueeze(0)

        path.append(image_path)
        segmap = Image.open(seg_path)
        segmap = transform_A(segmap).unsqueeze(0) * 255.0
        data_i = {'label': segmap, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
        input_semantics, _ = model.preprocess_input(data_i)

        labels.append(labels_to_num(torch.unique(segmap.long())))
        z = model.netG.get_z_style(image_tensor.cuda(), input_semantics.cuda())

        styles.append(z.detach())
    styles = torch.cat(styles, dim=0)
    labels = torch.tensor(labels)
    return styles,labels,path


remap_list = torch.tensor([0,1,7,6,5,4,2,2,10,11,12,9,8,15,3,17,16,18,13,14]).float()
def id_remap(seg):
    return remap_list[seg.long()]

opt = TestOptions().parse()
opt.status = 'test'

root = '/data/new_disk2/liury/log/SRNs/change_view_nose_center_seg_20'
save_root = './results/SEAN/'
styleImg_root = '/data/new_disk/chenap/code/SEAN/datasets/CelebA-HQ/test_raw/images'
styleSeg_root = '/data/new_disk/chenap/code/SEAN/datasets/CelebA-HQ/test_raw/labels'
styleList = sorted(os.listdir(styleImg_root))
folders = os.listdir(root)
resolution_vis = 512
nrows,ncols = 2,3

params = get_params(opt, (512,512))
transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
transform_image = get_transform(opt, params)

model = Pix2PixModel(opt)
model.eval()

styles, labels, path = load_style(len(styleList))
for i, folder in enumerate(folders[1:]):
    folder_img = os.path.join(root, folder, 'seg')
    img_list = sorted(os.listdir(folder_img))


    print('Processing folder %s.'%folder)
    for j, item in enumerate(img_list):
        if not item.endswith('g'):
            continue

        img_path = os.path.join(folder_img, item)
        save_path = os.path.join(folder_img, item)

        segmap = Image.open(img_path)
        segmap = id_remap(transform_A(segmap).unsqueeze(0) * 255.0)
        if 0==j:
            id = labels_to_num(torch.unique(segmap.long()))
            style = styles[labels==id]
            index = np.arange(style.shape[0])
            np.random.shuffle(index)
            if style.shape[0] >= nrows * ncols:
                out = cv2.VideoWriter(os.path.join(save_root, '%s.avi' % folder), cv2.VideoWriter_fourcc(*'XVID'), 10, \
                                  (resolution_vis * ncols + 2 * (ncols + 1), resolution_vis * nrows + 2 * (nrows + 1)))

        if style.shape[0] < nrows*ncols:
            continue

        data_i = {'label': segmap, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
        input_semantics, _ = model.preprocess_input(data_i)
        input_semantics = input_semantics.cuda()
        with torch.no_grad():
            result = []
            for k in range(nrows*ncols):
                style_test, path_test = style[[index[k]]], path[index[k]]

                fake_image = model.netG(input_semantics, style_codes=style_test, obj_dic=path_test)#obj_dic=obj_dic,
                result.append(fake_image.detach().cpu())

        result = torch.cat(result, dim=0)
        result = F.interpolate(result, (resolution_vis, resolution_vis))
        result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
        result = (result.numpy().astype('uint8')[[2, 1, 0]]).transpose((1, 2, 0))
        out.write(result)


