import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from scipy.io import loadmat, savemat

class Dataset(udata.Dataset):
    def __init__(self,win,path,aug_mode,train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode

        self.LQ_list = []
        self.HQ_list = []

        LQ_dir = os.path.join(path, "input")
        HQ_dir = os.path.join(path, "groundtruth")
        path_tmp = os.listdir(LQ_dir)
        for j in range(len(path_tmp)):
            path_tmp[j] = os.path.join(LQ_dir, path_tmp[j])
        self.LQ_list = path_tmp
        self.LQ_list.sort()

        path_tmp = os.listdir(HQ_dir)
        for j in range(len(path_tmp)):
            path_tmp[j] = os.path.join(HQ_dir, path_tmp[j])
        self.HQ_list = path_tmp
        self.HQ_list.sort()

    def argument(self, lq, hq):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        if hflip:
            hq = torch.flip(hq, dims=[2]).clone()
            lq = torch.flip(lq, dims=[2]).clone()
        if vflip:
            hq = torch.flip(hq, dims=[1]).clone()
            lq = torch.flip(lq, dims=[1]).clone()
        return lq, hq

    def get_patch_random(self, lq, hq):
        win = self.win
        h, w = hq.shape[1:3]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        lq = lq[:,y:y + win, x:x + win]
        hq = hq[:,y:y + win, x:x + win]
        return lq,hq


    def load_file(self,idx):
        LQ_data = cv2.imread(self.LQ_list[idx])
        LQ_data = cv2.cvtColor(LQ_data, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(self.LQ_list[idx])
        HQ_data = cv2.imread(self.HQ_list[idx])
        HQ_data = cv2.cvtColor(HQ_data, cv2.COLOR_BGR2RGB)

        return LQ_data,HQ_data,filename

    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor / 255.
        return img_tensor

    def __len__(self):
        return len(self.HQ_list)

    def __getitem__(self, idx):
        LQ_data, HQ_data, filename = self.load_file(idx)
        LQ_data = self.totensor(LQ_data)
        HQ_data = self.totensor(HQ_data)

        if self.train:
            if self.aug_mode:
                LQ_data, HQ_data = self.argument(LQ_data, HQ_data)
            LQ_data, HQ_data = self.get_patch_random(LQ_data, HQ_data)

        return {"HQ": HQ_data, "LQ": LQ_data, "filename": filename}


class DND_benchmark_Dataset(udata.Dataset):
    def __init__(self,path):
        self.dirpath = os.path.join(path, 'images_srgb')
        infos = h5py.File(os.path.join(path, 'info.mat'), 'r')
        self.info = infos['info']
        self.bb = self.info['boundingboxes']

    def __len__(self):
        return 50


    def __getitem__(self, idx):
        filename = '%04d.mat'%(idx + 1)
        filepath = os.path.join(self.dirpath, filename)
        img = h5py.File(filepath, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        ref = self.bb[0][idx]
        boxes = np.array(self.info[ref]).T

        return {"Inoisy": Inoisy, "boxes": boxes, "filename": filename}

class SIDD_benchmark_Dataset(udata.Dataset):
    def __init__(self,path):
        noisy_data_mat_file = os.path.join(path, 'BenchmarkNoisyBlocksSrgb.mat')
        noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
        self.noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]

    def __len__(self):
        return self.noisy_data_mat.shape[0]

    def block_len(self):
        return self.noisy_data_mat.shape[1]

    def __getitem__(self, idx):
        noisy_image = self.noisy_data_mat[idx, :, :, :, :]
        noisy_image = np.float32(noisy_image / 255.)
        noisy_image = torch.from_numpy(noisy_image.transpose((0, 3, 1, 2)))
        return {"noisy_image": noisy_image}








