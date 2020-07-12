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

def vis_condition_img(img):
    part_colors = torch.tensor([[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],#'skin',1 'eye_brow'2,  'eye'3
                    [240, 157, 240], [255, 212, 255], #'r_nose'4, 'l_nose'5
                    [31, 162, 230], [127, 255, 255], [127, 255, 255],#'mouth'6, 'u_lip'7,'l_lip'8
                    [0, 255, 85], [0, 255, 170], #'ear'9 'ear_r'10
                    [255, 255, 170],
                    [127, 170, 255], [85, 0, 255], [255, 170, 127], #'neck'11, 'neck_l'12, 'cloth'13
                    [212, 127, 255], [0, 170, 255],#, 'hair'14, 'hat'15
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]).float()

    N,C,H,W = img.size()
    condition_img_color = torch.zeros((N,3,H,W))
    num_of_class = int(torch.max(img))
    for pi in range(1, num_of_class + 1):
        index = (img == pi).nonzero()
        condition_img_color[index[:,0],:,index[:,2], index[:,3]] = part_colors[pi]
    condition_img_color = condition_img_color/255*2.0-1.0
    return condition_img_color

remap_list = torch.tensor([0,1,7,6,5,4,2,2,10,11,12,9,8,15,3,17,16,18,13,14]).float()
def id_remap(seg):
    return remap_list[seg.long()]

opt = TestOptions().parse()
opt.status = 'test'

save_root = './results/%s'%opt.name
styleImg_root = opt.image_dir
styleSeg_root = opt.label_dir
styleList = sorted(os.listdir(styleImg_root))
img_list = sorted(os.listdir(styleSeg_root))
resolution_vis, rows, cols = 512, 2, 3


params = get_params(opt, (512,512))
transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
transform_image = get_transform(opt, params)

model = Pix2PixModel(opt)
model.eval()

styles, labels, path = load_style(len(styleList))

for j, item in enumerate(img_list):
    if not item.endswith('g'):
        continue

    img_path = os.path.join(styleSeg_root, item)
    save_path = os.path.join(save_root, item)

    segmap = Image.open(img_path)
    segmap = transform_A(segmap).unsqueeze(0) * 255.0#id_remap()

    id = labels_to_num(torch.unique(segmap.long()))
    style = styles[labels==id]
    index = np.arange(style.shape[0])
    np.random.shuffle(index)

    data_i = {'label': segmap, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
    input_semantics, _ = model.preprocess_input(data_i)
    input_semantics = input_semantics.cuda()

    print('process image... %s' % img_path)
    
    result = []
    result.append(vis_condition_img(segmap))
    with torch.no_grad():
        for k in range(rows*cols-1):
            style_test, path_test = style[[index[k]]], path[index[k]]

            fake_image = model.netG(input_semantics, style_codes=style_test, obj_dic=path_test)#obj_dic=obj_dic,
            result.append(fake_image.detach().cpu())

    result = torch.cat(result, dim=0)
    result = F.interpolate(result, (resolution_vis, resolution_vis))
    result = (utils.make_grid(result, nrow=cols) + 1) / 2 * 255
    result = (result.numpy().astype('uint8')[[2, 1, 0]]).transpose((1, 2, 0))
    cv2.imwrite(save_path,result)


