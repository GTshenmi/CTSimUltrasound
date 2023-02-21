import argparse
import os
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import datetime
import sys
import scipy.io as scio
import scipy
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def GetRFData(dataPath,opt):

    D = 10  # Sampling frequency decimation factor
    fs = 100e6 / D  # Sampling frequency  [Hz]

    #  Read the data and adjust it in time

    min_sample = 0

    env = []

    env_max_list = []
    env_min_list = []

    for i in range(128):
        dataMatName = f'rf_ln{i + 1}.mat'
        dataFile = os.path.join(dataPath, dataMatName)
        data = scio.loadmat(dataFile)
        rf_data = data["rf_data"]
        rf_data = np.resize(rf_data, (1, len(rf_data)))[0]
        t_start = data["tstart"]

        size_x = int(np.round(t_start * fs - min_sample))

        if size_x > 0:
            rf_data = np.concatenate((np.zeros((size_x)), rf_data))

        rf_env = scipy.signal.hilbert(rf_data)

        rf_env = np.abs(rf_env)

        env_max_list.append(max(rf_env))
        env_min_list.append(min(rf_env))

        env.append(rf_env)

    max_env = max(env_max_list)
    min_env = min(env_min_list)
    log_env = []
    data_len = opt.rfdata_len
    log_env_max_list = []
    log_env_min_list = []

    for i in range(len(env)):

        D = int(np.floor(len(env[i]) / data_len))

        env[i] = env[i] - min_env

        tmp_env = env[i][slice(0, data_len * D, D)]/max_env

        for i in range(len(tmp_env)):
            if tmp_env[i] != 0.0:
                tmp_env[i] = np.log(tmp_env[i])

        log_env_max_list.append(max(tmp_env))
        log_env_min_list.append(min(tmp_env))
        log_env.append(tmp_env)

    log_env_max = max(log_env_max_list)
    log_env_min = min(log_env_min_list)
    # print((log_env_min, log_env_max))
    log_env_max = log_env_max - log_env_min

    disp_env = []

    for i in range(len(log_env)):

        tmp_env = log_env[i]

        tmp_env = ((tmp_env - log_env_min) / log_env_max)

        disp_env.append(tmp_env)


    disp_env = np.asarray(disp_env)

    return disp_env

class MyDataSet(Dataset):
    def __init__(self,root,opt,img_transform=None, rf_transform=None,mode="train",use_input="rf_data",use_embedding = True):

        self.rf_dir = os.path.join(root, "rf_data")
        self.us_dir = os.path.join(root, "us_image")
        self.img_transform = None
        self.rf_transform = None

        if img_transform:
            self.img_transform = transforms.Compose(img_transform)
        if rf_transform:
            self.rf_transform = transforms.Compose(rf_transform)

        if mode == "train":
            f = open(os.path.join(root, "us_image_train.txt"), "r", encoding='utf-8')
            us_train_list = f.read().splitlines()
            f.close()
            self.us_list = us_train_list
        else:
            f = open(os.path.join(root, "us_image_val.txt"), "r", encoding='utf-8')
            us_train_list = f.read().splitlines()
            f.close()
            self.us_list = us_train_list

        self.use_input = use_input

        self.opt = opt

        self.use_embedding = use_embedding

    def __len__(self):
        return len(self.us_list)

    def __getitem__(self, idx):

        loss = 0
        us_name = self.us_list[idx]
        us_path = os.path.join(self.us_dir, us_name)
        us = Image.open(us_path)
        if self.img_transform:
            us = self.img_transform(us)

        rf_name = us_name.replace(".png","")
        rf_path = os.path.join(self.rf_dir, rf_name)

        rf_data = GetRFData(rf_path, self.opt)

        if self.rf_transform:
            rf_data = self.rf_transform(rf_data)

        return {"rf_data": rf_data, "us": us, "name": us_name.replace(".png", "")}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=80, help="interval between image sampling")
    parser.add_argument("--dataset_name", type=str, default="rf2us12", help="name of the dataset")
    parser.add_argument("--rfdata_len", type=int, default=1024, help="length of rf data")
    parser.add_argument("--split_test", type=bool, default=True, help="if split test")
    parser.add_argument("--network", type=str, default="cgan", help="if split test")
    opt = parser.parse_args()

    #print(os.path.join("/home/xuepeng/ultrasound/dataset/train/rf_data/","patient001_frame01_slice9"))

    rf = GetRFData("/home/xuepeng/ultrasound/dataset/train/rf_data/patient001_frame01_slice9",opt=opt,use_embedding=True)
    print(np.shape(rf))

    print(rf)