import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        
        self.clean_filenames = [os.path.join(rgb_dir, 'groundtruth', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.clean_filenames = [os.path.join(rgb_dir, 'groundtruth', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2,0,1)

        return noisy, noisy_filename


##################################################################################################

class DataLoaderTestSR(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTestSR, self).__init__()

        self.target_transform = target_transform

        LR_files = sorted(os.listdir(os.path.join(rgb_dir)))


        self.LR_filenames = [os.path.join(rgb_dir, x) for x in LR_files if is_png_file(x)]
        

        self.tar_size = len(self.LR_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        LR = torch.from_numpy(np.float32(load_img(self.LR_filenames[tar_index])))
                
        LR_filename = os.path.split(self.LR_filenames[tar_index])[-1]

        LR = LR.permute(2,0,1)

        return LR, LR_filename
