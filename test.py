import os
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

D=10       #  Sampling frequency decimation factor
fs=100e6/D #  Sampling frequency  [Hz]
c=1540     #  Speed of sound [m]
no_lines=128                  #  Number of lines in image
image_width=90/180*np.pi         #  Size of image sector [rad]
dtheta=image_width/no_lines   #  Increment for image

#  Read the data and adjust it in time

min_sample=0
dataPath = './ultrasound_image_generation/dataset/rf_data/patient001_frame01_slice9/'

env = []

print(len(os.listdir("./dataset/label/")))

env_max_list = []
env_min_list = []

for i in range(128):
    dataMatName = f'rf_ln{i+1}.mat'
    dataFile = os.path.join(dataPath,dataMatName)
    data = scio.loadmat(dataFile)
    rf_data = data["rf_data"]
    rf_data = np.resize(rf_data,(1,len(rf_data)))[0]
    t_start = data["tstart"]


    size_x = int(np.round(t_start*fs-min_sample))

    if size_x > 0:
        #print(size_x)
        rf_data = np.concatenate((np.zeros((size_x)),rf_data))

    rf_env = scipy.signal.hilbert(rf_data)

    rf_env = np.abs(rf_env)

    env_max_list.append(max(rf_env))
    env_min_list.append(min(rf_env))

    env.append(rf_env)

D=10
dB_Range=50
max_env = max(env_max_list)
min_env = min(env_min_list)
print(max_env)
print(min_env)
log_env = []
log_env_max_list = []
log_env_min_list = []
D = 10
for i in range(len(env)):

    dB_Range = 50;

    env[i] = env[i] - min_env

    tmp_env = env[i][slice(0, len(env[i]), D)]/max_env

    tmp_env = 20 * np.log10(tmp_env)

    tmp_env = 255 / dB_Range * (tmp_env + dB_Range)

    log_env_max_list.append(max(tmp_env))
    log_env_min_list.append(min(tmp_env))
    log_env.append(tmp_env)

log_env_max = max(log_env_max_list)
log_env_min = min(log_env_min_list)
print(log_env_min)
print(log_env_max)
log_env_max = log_env_max - log_env_min
data_len = 512
disp_env = []

for i in range(len(log_env)):
    D = int(np.floor(len(log_env[i])/data_len))

    tmp_env = log_env[i][slice(0,data_len*D,D)]

    tmp_env = 255 * ((tmp_env - log_env_min) / log_env_max)

    disp_env.append(tmp_env)

disp_env = np.asarray(disp_env)
con_env = np.concatenate((disp_env,disp_env,disp_env,disp_env),axis=0)

print(np.shape(con_env))

# print(disp_env)
# print(np.shape(disp_env))
# print(type(disp_env))
#env=env-min(min(env))


# log_env=20*log10(env(1:D:max(size(env)),:)/max(max(env)))
# log_env=255/dB_Range*(log_env+dB_Range)
# print(env)
# print(env[0][:])
#env = np.array(env,axis=0)
# for i in range(128):
#     print(len(env[i]))
# env_len = np.size(env[0])
# print(env_len)
# print(type(env))
# print(env[0][1:2:env_len])
# print(env[:][0])
# dB_Range=50
#
# rf_env=rf_env-min(rf_env)
#
# print(rf_env)

# log_env=20*log10(env(1:D:max(size(env)),:)/max(max(env)))
# log_env=255/dB_Range*(log_env+dB_Range)

# [-6.16297582e-33 -8.08890577e-33 -7.51112678e-33 ... -1.07852077e-32
#   4.23704588e-33 -4.42963887e-33]
# [-1.03655951e-32-9.00589733e-26j -4.28700233e-33-9.00960617e-26j
#  -1.07579753e-32-9.01331512e-26j ... -8.57299515e-33-8.99476994e-26j
#   2.63078045e-33-8.99848019e-26j -1.52974046e-34-9.00218736e-26j]
    #np.abs
# label_dir = "./dataset/label/"
# label_list = os.listdir("./dataset/label/")
# label_name = label_list[0]
# label_path = os.path.join(label_dir, label_name)
# label = Image.open(label_path)
# print(len(label.split()))
# print(len(label_list))
# print(os.listdir("./dataset/input/"))

# class MyImageDataSet(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.image_list = os.listdir(self.image_dir)
#         self.label_list = os.listdir(self.label_dir)
#
#     def __len__(self):
#         return len(self.label_list)
#
#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         image_path = os.path.join(self.image_dir, image_name)
#         image = read_image(image_path)
#
#         label_name = image_name
#         label_path = os.path.join(self.label_dir, label_name)
#         label = read_image(label_path)
#
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
# Configure data loader
# dataset = MyImageDataSet(image_dir="./dataset/input/",label_dir="./dataset/label/",
#                            transform=transforms.Compose([
#                                transforms.Resize(opt.image_size),
#                                transforms.CenterCrop(opt.image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]),target_transform=transforms.Compose([
#                                transforms.Resize(opt.image_size),
#                                transforms.CenterCrop(opt.image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
#                                          shuffle=True)