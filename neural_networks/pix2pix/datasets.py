import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
import scipy.io as scio
import scipy

def GetRFData(dataPath):

    D = 10  # Sampling frequency decimation factor
    fs = 100e6 / D  # Sampling frequency  [Hz]
    c = 1540  # Speed of sound [m]
    no_lines = 128  # Number of lines in image
    image_width = 90 / 180 * np.pi  # Size of image sector [rad]
    dtheta = image_width / no_lines  # Increment for image

    #  Read the data and adjust it in time

    min_sample = 0
    #dataPath = './ultrasound_image_generation/dataset/rf_data/patient001_frame01_slice9/'

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
            # print(size_x)
            rf_data = np.concatenate((np.zeros((size_x)), rf_data))

        rf_env = scipy.signal.hilbert(rf_data)

        rf_env = np.abs(rf_env)

        env_max_list.append(max(rf_env))
        env_min_list.append(min(rf_env))

        env.append(rf_env)

    D = 10
    dB_Range = 50
    max_env = max(env_max_list)
    min_env = min(env_min_list)
    log_env = []
    log_env_max_list = []
    log_env_min_list = []
    D = 10

    for i in range(len(env)):
        dB_Range = 50;

        env[i] = env[i] - min_env

        tmp_env = env[i][slice(0, len(env[i]), D)] / max_env

        #tmp_env = 20 * np.log10(tmp_env)

        tmp_env = 255 / dB_Range * (tmp_env + dB_Range)

        log_env_max_list.append(max(tmp_env))
        log_env_min_list.append(min(tmp_env))
        log_env.append(tmp_env)

    log_env_max = max(log_env_max_list)
    log_env_min = min(log_env_min_list)
    log_env_max = log_env_max - log_env_min
    data_len = 256
    disp_env = []

    for i in range(len(log_env)):
        D = int(np.floor(len(log_env[i]) / data_len))

        tmp_env = log_env[i][slice(0, data_len * D, D)]

        tmp_env = 255 * ((tmp_env - log_env_min) / log_env_max)

        disp_env.append(tmp_env)

    disp_env = np.asarray(disp_env)
    #con_env = np.concatenate((disp_env, disp_env, disp_env, disp_env), axis=0)
    con_env = np.concatenate((disp_env, disp_env), axis=0)

    return con_env

class ImageDataset(Dataset):
    def __init__(self, rf_dir, label_dir, transform=None, target_transform=None,mode="train"):
        self.rf_dir = rf_dir
        self.label_dir = label_dir
        self.transform = None
        self.target_transform = None

        if transform:
            self.transform = transforms.Compose(transform)
        if target_transform:
            self.target_transform = transforms.Compose(target_transform)

        self.rf_list = os.listdir(self.rf_dir)
        self.label_list = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        # image_name = self.image_list[idx]
        # image_path = os.path.join(self.image_dir, image_name)
        # image = read_image(image_path)
        #
        # label_name = image_name
        # label_path = os.path.join(self.label_dir, label_name)
        # label = read_image(label_path)

        label_name = self.label_list[idx]
        label_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_path)

        rf_name = label_name.replace(".png","")
        rf_path = os.path.join(self.rf_dir, rf_name)
        rf_data = GetRFData(rf_path)

        #print(image.size)
        # img = Image.open(self.files[index % len(self.files)])
        # w, h = img.size
        # img_A = img.crop((0, 0, w / 2, h))
        # img_B = img.crop((w / 2, 0, w, h))

        # if np.random.random() < 0.5:
        #     image = Image.fromarray(np.array(image)[:, ::-1, :], "RGB")
        #     label = Image.fromarray(np.array(label)[:, ::-1, :], "RGB")




        if self.transform:
            image = self.transform(rf_data)
        if self.target_transform:
            label = self.target_transform(label)
        #print(idx)

        #return {"A": rf_data, "B": label}
        return {"A": label, "B": rf_data}

# class ImageDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None, target_transform=None,mode="train"):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = None
#         self.target_transform = None
#
#         if transform:
#             self.transform = transforms.Compose(transform)
#         if target_transform:
#             self.target_transform = transforms.Compose(target_transform)
#
#         self.image_list = os.listdir(self.image_dir)
#         self.label_list = os.listdir(self.label_dir)
#
#     def __len__(self):
#         return len(self.label_list)
#
#     def __getitem__(self, idx):
#         # image_name = self.image_list[idx]
#         # image_path = os.path.join(self.image_dir, image_name)
#         # image = read_image(image_path)
#         #
#         # label_name = image_name
#         # label_path = os.path.join(self.label_dir, label_name)
#         # label = read_image(label_path)
#
#         label_name = self.label_list[idx]
#         label_path = os.path.join(self.label_dir, label_name)
#         label = Image.open(label_path)
#
#         image_name = label_name
#         image_path = os.path.join(self.image_dir, image_name)
#         image = Image.open(image_path)
#
#         print(image.size)
#         # img = Image.open(self.files[index % len(self.files)])
#         # w, h = img.size
#         # img_A = img.crop((0, 0, w / 2, h))
#         # img_B = img.crop((w / 2, 0, w, h))
#
#         # if np.random.random() < 0.5:
#         #     image = Image.fromarray(np.array(image)[:, ::-1, :], "RGB")
#         #     label = Image.fromarray(np.array(label)[:, ::-1, :], "RGB")
#
#
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         #print(idx)
#
#         return {"A": image, "B": label}
#         #return {"A": label, "B": image}

# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)
#
#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#         if mode == "train":
#             self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
#
#     def __getitem__(self, index):
#
#         img = Image.open(self.files[index % len(self.files)])
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))
#
#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
#
#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)
#
#         return {"A": img_A, "B": img_B}
#
#     def __len__(self):
#         return len(self.files)
