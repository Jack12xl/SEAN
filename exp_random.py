import os

script_path = 'test_random_style.py'
how_many = 10000

label_dir = '/mnt/new_disk2/liury/log/SRNs/test/reproj_celebA_rotate/selected/1233'
image_dir = '/mnt/data/new_disk/chenap/dataset/ffhq/images/0069'
results_dir = './results/Jack12test'
name = 'SEAN-FFHQ'
label_nc = 20
gpu_id = 1

which_epoch = 800000
# for epoch_num in range(2, 10, 2):
command = 'python {} ' \
          '--name {} ' \
          '--no_instance ' \
          '--label_nc {} ' \
          '--dataset_mode custom ' \
          '--use_vae ' \
          '--label_dir {} ' \
          '--image_dir {} ' \
          '--results_dir {} ' \
          '--which_epoch {} ' \
          '--gpu_id {} ' \
          '--how_many {} '.format(script_path, name, label_nc,label_dir, image_dir, results_dir, which_epoch, gpu_id, how_many)

# python
# test.py - -name
# CelebA - HQ_pretrained - -load_size
# 256 - -crop_size
# 256 - -dataset_mode
# custom - -label_dir
# datasets / CelebA - HQ / test / labels - -image_dir
# datasets / CelebA - HQ / test / images - -label_nc
# 19 - -no_instance - -gpu_ids
# 0
print("start to run command {}".format(command))
os.system(command)
