"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from networks.MIRNet_model import MIRNet
from dataloaders.data_rgb import get_test_data
import utils
from utils.bundle_submissions import bundle_submissions_srgb_v1
from skimage import img_as_ubyte


parser = argparse.ArgumentParser(description='RGB denoising evaluation on DND dataset')
parser.add_argument('--input_dir', default='./datasets/dnd/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/denoising/dnd/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/model_denoising.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=16, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir+'matfile')
utils.mkdir(args.result_dir+'png')

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)



model_restoration = MIRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()


with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].cuda()
        filenames = data_test[1]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_noisy)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir + 'png/'+ filenames[batch][:-4] + '.png', denoised_img)
                save_file = os.path.join(args.result_dir+ 'matfile/', filenames[batch][:-4] +'.mat')
                sio.savemat(save_file, {'Idenoised_crop': np.float32(rgb_restored[batch])})

  

bundle_submissions_srgb_v1(args.result_dir+'matfile/', 'srgb_results_for_server_submission/')
os.system("rm {}".format(args.result_dir+'matfile/*.mat'))
