"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os, cv2, torch
import util.util as util
from torch import nn
from io import BytesIO
from scipy import linalg
from calc_inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
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

def load_style_LMDB(batch):
    index = np.arange(length_img)
    np.random.shuffle(index)

    pbar = range(batch)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    styles,labels, path = [],[],[]
    for i in pbar:
        pbar.set_description(
            (
                    'Extracting styles % d of % d' % (i, batch)
            )
        )

        key = f'{1024}-{str(index[i]).zfill(5)}'.encode('utf-8')
        with styleList.begin(write=False) as txn:
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            image = Image.open(buffer)
            image = image.convert('RGB')
            image_tensor = transform_image(image).unsqueeze(0)

        key = f'{512}-{str(index[i]).zfill(5)}'.encode('utf-8')
        with img_list.begin(write=False) as txn:
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            segmap = Image.open(buffer)
            segmap = transform_A(segmap).unsqueeze(0) * 255.0

        path.append(opt.label_dir)
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

######################################################
class FID(nn.Module):
    def __init__(self,dims=2048):
        super().__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).cuda().eval()
        self.real_mean = np.load('./ffhq_mu512.npy')
        self.real_cov = np.load('./ffhq_sigma512.npy')


    def cal_mean_latent(self, truncation, generator, truncation_mean=4096):
        if truncation < 1:
            with torch.no_grad():
                mean_latent = generator.mean_latent(truncation_mean)
        else:
            mean_latent = None
        return mean_latent

    @torch.no_grad()
    def extract_feature_from_samples(self, model, saveName='checkpoint', batch_size=6, n_sample=50000):

        n_batch = n_sample // batch_size
        resid = n_sample - (n_batch * batch_size)
        batch_sizes = [batch_size] * n_batch + [resid]
        features = []

        pbar = range(len(batch_sizes))
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

        count = 0
        for idx in pbar:

            pbar.set_description(
                (
                    'Processing % d of % d'%(count, n_sample)
                )
            )

            index = np.random.choice(length_label,1)[0]
            if isinstance(img_list,list):
                file = img_list[index]
                img_path = os.path.join(styleSeg_root, file)
                segmap = Image.open(img_path)
            else:
                key = f'{512}-{str(index).zfill(5)}'.encode('utf-8')
                with img_list.begin(write=False) as txn:
                    condition_bytes = txn.get(key)
                    buffer = BytesIO(condition_bytes)
                    segmap = Image.open(buffer)
            segmap = transform_A(segmap).unsqueeze(0) * 255.0  # id_remap()

            # id = labels[index]
            id = labels_to_num(torch.unique(segmap.long()))
            style = styles[labels == id]
            index = np.arange(style.shape[0])
            np.random.shuffle(index)

            data_i = {'label': segmap, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
            input_semantics, _ = model.preprocess_input(data_i)
            input_semantics = input_semantics.cuda()

            sample = min(batch_size, len(index))
            count += sample
            results = []
            for k in range(sample):
                style_test, path_test = style[[index[k]]], path[index[k]]
                fake_image = model.netG(input_semantics, style_codes=style_test,
                                        obj_dic=path_test)  # obj_dic=obj_dic,

                results.append(fake_image.detach().cpu())

                fake_image = (fake_image.clamp(-1.0, 1.0) + 1) / 2
                pred = self.inception(fake_image)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                features.append(pred.to('cpu'))

            results = torch.cat(results, dim=0)
            results = F.interpolate(results, (resolution_vis, resolution_vis))

            if idx % (len(batch_sizes)//100) == 0:
                results = (utils.make_grid(results, nrow=(sample + 1) // rows) + 1) / 2 * 255
                results = (results.numpy().astype('uint8')[[2, 1, 0]]).transpose((1, 2, 0))
                cv2.imwrite('results/fid_sample/%s_%05d.png'%(saveName,idx*batch_size), results)


        features = torch.cat(features, 0)
        return features.reshape(features.shape[0], -1)


    def calc_fid(self, sample_mean, sample_cov, eps=1e-6):
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ self.real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (self.real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - self.real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(self.real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return fid
######################################################

opt = TestOptions().parse()
opt.status = 'test'

length_label,length_img = 0,0
save_root = './results/%s'%opt.name
if 'LMDB' in opt.label_dir:
    import lmdb
    img_list = lmdb.open(
        opt.label_dir,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with img_list.begin(write=False) as txn:
        length_label = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
else:
    styleSeg_root = opt.label_dir
    img_list = sorted(os.listdir(styleSeg_root))
resolution_vis, rows, cols = 512, 2, 3


params = get_params(opt, (512,512))
transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
transform_image = get_transform(opt, params)

model = Pix2PixModel(opt)
model.eval()



if not os.path.exists('ffhq_styles.npy'):
    print('Extracting styles.')
    if 'LMDB' in opt.image_dir:
        styleList = lmdb.open(
            opt.image_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with styleList.begin(write=False) as txn:
            length_img = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        if length_img != length_label:
            print(length_img,length_label,'length_img != length_label')
            exit()

        length_img = min(100000,length_img)
        styles, labels, path = load_style_LMDB(length_img)  #
    else:
        styleImg_root = opt.image_dir
        styleList = sorted(os.listdir(styleImg_root))
        styles, labels, path = load_style(len(styleList))#

    result_file = {'styles': styles, 'labels': labels, 'path': path}
    torch.save(result_file, 'ffhq_styles.npy')
    print('Extract styles done.')
else:
    print('==> Loading styles.')
    styles = torch.load('ffhq_styles.npy')
    styles, labels, path = styles['styles'], styles['labels'], styles['path']


fid = FID()
epochs = list(range(800000,900000,100000))
for epoch in epochs:
    if not os.path.exists('results/fid_sample/%s' % opt.name):
        os.mkdir('results/fid_sample/%s' % opt.name)
    model.netG = util.load_network(model.netG, 'G', str(epoch), opt)

    features = fid.extract_feature_from_samples(model,n_sample=1000,
                                    batch_size=6, saveName='%s/%s' % (opt.name,str(epoch))).numpy()
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    fid_score = fid.calc_fid(sample_mean, sample_cov)

    print('%d fid: %s' % (epoch, fid_score))





