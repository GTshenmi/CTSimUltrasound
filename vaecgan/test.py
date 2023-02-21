from datasetnew import MyDataSet
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=80, help="interval between image sampling")
parser.add_argument("--dataset_name", type=str, default="rf2us1", help="name of the dataset")
parser.add_argument("--rfdata_len", type=int, default=1024, help="length of rf data")
parser.add_argument("--split_test", type=bool, default=True, help="if split test")
parser.add_argument("--network", type=str, default="aecgan", help="if split test")
parser.add_argument("--use_gpu", type=int, default=0, help="use gpu id")

opt = parser.parse_args()

rf_transforms = [
    transforms.ToTensor(),
]
img_transforms = [

    transforms.Resize((opt.img_size, opt.img_size)),

    transforms.ToTensor(),

    transforms.Normalize(0.5, 0.5),
]

dataloader = DataLoader(
    MyDataSet(root="../datasetnew/", opt=opt, img_transform=img_transforms, rf_transform=rf_transforms),
    batch_size=1,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

Tensor = torch.cuda.FloatTensor

for i, batch in enumerate(dataloader):
    # Model inputs
    labels = batch["us"]
    rf_datas = batch["rf_data"]

    batch_size = labels.shape[0]

    # Configure input
    # real_imgs = labels.type(Tensor)
    real_imgs = labels.type(Tensor)
    rf_datas = rf_datas.type(Tensor)

    img = np.array(Tensor.cpu(real_imgs))

    print(img)

    print(np.shape(img))

    print((np.min(img), np.max(img)))
    # print(real_imgs)
    # print("______________")
    # print(rf_datas)
    #
    # print(torch.max(real_imgs,0))
