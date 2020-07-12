"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.status = 'test'

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

obj_dic = None

################# Try mean code ######################

# from glob import glob
# import numpy as np
# import torch

# average_style_code_folder = 'styles_test/mean_style_code/mean/'
# obj_dic = {}

# for i in range(19):
#     obj_dic[str(i)] = {}

#     average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
#     average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
#                                 average_category_folder_list]

#     for style_code_path in average_category_list:
#             obj_dic[str(i)][style_code_path] = torch.from_numpy(
#                 np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
                
######################################################

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    if obj_dic is not None:
        print('Load obj_dic')
        data_i['obj_dic'] = obj_dic

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()


