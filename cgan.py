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
from dataset import MyDataSet
import matplotlib.pyplot as plt

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            #nn.Conv2d(in_size, in_size, 3, 1, 1),
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv2d(out_size, out_size, 3, 1, 1),
                  ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            #nn.ConvTranspose2d(in_size, in_size, 3, 1, 1, bias=False),
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class Generator(nn.Module):
    def __init__(self, in_channels=16, out_channels=1,img_shape=(1, 256, 256)):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.3)
        self.down5 = UNetDown(512, 512, dropout=0.4)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        #self.down7 = UNetDown(512, 512, dropout=0.5)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.4)
        self.up3 = UNetUp(1024, 256, dropout=0.3)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
        # self.up1 = UNetUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        # self.up4 = UNetUp(1024, 512, dropout=0.5)
        # self.up5 = UNetUp(1024, 256)
        # self.up6 = UNetUp(512, 128)
        # self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        self.img_shape = img_shape

    def forward(self,condition):
        # U-Net generator with skip connections from encoder to decoder

        # d1 = self.down1(condition)
        # d2 = self.down2(d1)
        # d3 = self.down3(d2)
        # d4 = self.down4(d3)
        # d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # u1 = self.up1(d8, d7)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u2, d5)
        # u4 = self.up4(u3, d4)
        # u5 = self.up5(u4, d3)
        # u6 = self.up6(u5, d2)
        # u7 = self.up7(u6, d1)
        # img = self.final(u7)

        d1 = self.down1(condition)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        img = self.final(u5)

        img_shape = (1,256,256) #remember to edit

        img = img.view(img.size(0), *img_shape)

        # print("gen")
        # print(img.shape)

        return img

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                #nn.Conv2d(in_filters, in_filters, 3, stride=1, padding=1),
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 17, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self,condition,img):
        # Concatenate image and condition image by channels to produce input

        img_input = torch.cat((img,condition), 1)

        outpuut = self.model(img_input)

        return outpuut

class ConditionGan():
    def __init__(self,opt):

        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.device = torch.device("cuda:" + str(opt.use_gpu) if self.cuda else "cpu")

        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.input_shape = (128, opt.rfdata_len)
        self.patch = (1, opt.img_size // 2 ** 4, opt.img_size // 2 ** 4)

        os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
        os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

        self.rf_transforms = [
            transforms.ToTensor(),
        ]
        self.img_transforms = [

            transforms.Resize((opt.img_size, opt.img_size)),

            transforms.ToTensor(),

            transforms.Normalize(0.5,0.5),
        ]
        self.dataloader = DataLoader(
            MyDataSet(root="./dataset/",opt = opt, img_transform=self.img_transforms,rf_transform=self.rf_transforms,use_embedding=True),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )

        self.val_dataloader = DataLoader(
            MyDataSet(root="./dataset/",opt = opt, img_transform=self.img_transforms,rf_transform=self.rf_transforms,mode="test"),
            batch_size=10,
            shuffle=True,
            num_workers=1,
        )
        # Initialize generator and discriminator
        self.generator = Generator(img_shape=self.img_shape)
        self.discriminator = Discriminator()
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.lambda_pixel = 100
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.criterion_GAN.to(self.device)
        self.criterion_pixelwise.to(self.device)

        self.opt = opt

    def sample_images(self,batches_done,train_real,train_fake):
        """Saves a generated sample from the validation set"""
        datas = next(iter(self.val_dataloader))

        # real_imgs = Variable(datas["us"].type(Tensor))
        # conditions = Variable(datas["rf_data"].type(Tensor))

        real_imgs = datas["us"].type(self.Tensor)
        conditions = datas["rf_data"].type(self.Tensor)

        #batch_size = datas["us"].shape[0]

        #z = Variable(Tensor(np.random.normal(0, 1, (batch_size, *img_shape))))

        fake_imgs = self.generator(conditions)

        img_sample = torch.cat((fake_imgs.data,real_imgs.data), -2)
        img_sample_train = torch.cat((train_fake.data,train_real.data), -2)
        #img_sample = torch.cat((fake_imgs.data, imgs.data), -2)

        save_image(img_sample, "images/%s/test_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)
        save_image(img_sample_train,"images/%s/train_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

    def TrainModel(self):

        prev_time = time.time()
        print(self.generator)
        print(self.discriminator)
        for epoch in range(self.opt.n_epochs):
            for i, batch in enumerate(self.dataloader):

                # Model inputs
                labels = batch["us"]
                rf_datas = batch["rf_data"]

                batch_size = labels.shape[0]


                #Adversarial ground truths
                valid =self.Tensor(np.ones((batch_size, *self.patch)))
                fake = self.Tensor(np.zeros((batch_size, *self.patch)))

                # Configure input
                real_imgs = labels.type(self.Tensor)
                conditions = rf_datas.type(self.Tensor)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                #z = Variable(Tensor(np.random.normal(0, 1, (batch_size, *img_shape))))

                fake_imgs = self.generator(conditions)

                pred_fake = self.discriminator(conditions,fake_imgs)
                loss_GAN = self.criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

                # Total loss
                loss_G = loss_GAN + self.lambda_pixel * loss_pixel

                loss_G.backward()

                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.discriminator(conditions,real_imgs)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = self.discriminator(conditions,fake_imgs.detach())
                loss_fake = self.criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                batches_left = self.opt.n_epochs * len(self.dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s\r\n"
                    % (
                        epoch,
                        self.opt.n_epochs,
                        i,
                        len(self.dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_GAN.item(),
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % self.opt.sample_interval == 0:
                    self.sample_images(batches_done,real_imgs,fake_imgs)

            if epoch <= 120:
                if ((epoch + 1) % 5 == 0):
                    torch.save(self.generator, "./saved_models/{}/generator_{}.pth".format(self.opt.dataset_name, (epoch + 1)))
            else:
                if ((epoch + 1) % 2 == 0):
                    torch.save(self.generator, "./saved_models/{}/generator_{}.pth".format(self.opt.dataset_name, (epoch + 1)))

    def TestModelLoss(self,modelname):

        train_loss = []
        test_loss = []

        rootpath = os.path.join("./saved_models/", modelname)
        model_list = os.listdir(rootpath)

        with torch.no_grad():
            for model in model_list:

                tmp_loss = []
                generator = torch.load(os.path.join(rootpath, model))
                generator.to(self.device)
                generator.eval()

                for i, batch in enumerate(self.dataloader):
                    real_imgs = batch["us"].type(self.Tensor)
                    conditions = batch["rf_data"].type(self.Tensor)
                    name = batch["name"][0]

                    fake_imgs = generator(conditions)

                    loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

                    tmp_loss.append(self.Tensor.cpu(loss_pixel))

                    sys.stdout.write(
                        "\r[%s] [%s] [Batch %d/%d] [loss: %f]\r\n"
                        % (
                            model,
                            name,
                            i,
                            len(self.dataloader),
                            loss_pixel,
                        )
                    )

                train_loss.append(np.mean(tmp_loss))
                tmp_loss = []

                for i, batch in enumerate(self.val_dataloader):
                    real_imgs = batch["us"].type(self.Tensor)
                    conditions = batch["rf_data"].type(self.Tensor)

                    name = batch["name"][0]

                    fake_imgs = generator(conditions)

                    loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

                    tmp_loss.append(self.Tensor.cpu(loss_pixel))

                    sys.stdout.write(
                        "\r[%s] [%s] [Batch %d/%d] [loss: %f]\r\n"
                        % (
                            model,
                            name,
                            i,
                            len(self.val_dataloader),
                            loss_pixel,
                        )
                    )

                test_loss.append(np.mean(tmp_loss))

            np.save(f'./loss/{modelname}_testloss.npy', test_loss)
            np.save(f'./loss/{modelname}_trainloss.npy', train_loss)
            np.save(f'./loss/{modelname}_modelnames', model_list)

    def ShowModelLoss(self,model_name):
        train_loss = np.load(f'./loss/{model_name}_trainloss.npy')
        test_loss = np.load(f'./loss/{model_name}_testloss.npy')
        model_names = np.load(f'./loss/{model_name}_modelnames.npy')

        # fig = plt.figure(1)  # 如果不传入参数默认画板1
        # # 第2步创建画纸，并选择画纸1
        # ax1 = plt.subplot(2, 1, 1)
        # ax1.set_title("train loss")
        # # 在画纸1上绘图
        # plt.plot(train_loss)
        # # 选择画纸2
        # ax2 = plt.subplot(2, 1, 2)
        # ax2.set_title("test loss")
        # # 在画纸2上绘图
        # plt.plot(test_loss)
        # # 显示图像
        # plt.show()

        index = np.where(train_loss == np.min(train_loss))
        print(f'train min loss : {model_names[index]}')
        index = np.where(test_loss == np.min(test_loss))
        print(f'test min loss : {model_names[index]}')

        return model_names[index][0]


    def GengerateImg(self,model_name):
        rootpath = "./saved_models"
        best_model = self.ShowModelLoss(model_name)
        # best_model = "generator_110.pth"
        name = model_name
        # model_path = os.path.join(rootpath,best_model)
        print(best_model)
        save_path_train = os.path.join("./images/",name,"train")
        save_path_test = os.path.join("./images/",name,"test")

        os.makedirs(save_path_train, exist_ok=True)
        os.makedirs(save_path_test, exist_ok=True)

        train_loss = []
        test_loss = []

        with torch.no_grad():
            generator = torch.load(os.path.join(rootpath,name,best_model))
            generator.cuda(self.opt.use_gpu)
            generator.eval()

            for i, batch in enumerate(self.dataloader):
                real_imgs = batch["us"].type(self.Tensor)
                conditions = batch["rf_data"].type(self.Tensor)
                name = batch["name"][0]

                fake_imgs = generator(conditions)

                loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

                img_sample = torch.cat((fake_imgs.data, real_imgs.data), -2)

                save_image(img_sample, "%s/test_%s.png" % (save_path_train, i), nrow=5, normalize=True)

                train_loss.append(self.Tensor.cpu(loss_pixel))

                sys.stdout.write(
                    "\r[Batch %d/%d] [loss: %f]\r\n"
                    % (
                        i,
                        len(self.dataloader),
                        loss_pixel,
                    )
                )

            for i, batch in enumerate(self.val_dataloader):
                real_imgs = batch["us"].type(self.Tensor)
                conditions = batch["rf_data"].type(self.Tensor)

                name = batch["name"][0]

                fake_imgs = generator(conditions)

                loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

                img_sample = torch.cat((fake_imgs.data, real_imgs.data), -2)

                save_image(img_sample, "%s/test_%s.png" % (save_path_test, i), nrow=5, normalize=True)

                test_loss.append(self.Tensor.cpu(loss_pixel))

                sys.stdout.write(
                    "\r[Batch %d/%d] [loss: %f]\r\n"
                    % (
                        i,
                        len(self.val_dataloader),
                        loss_pixel,
                    )
                )

            # fig = plt.figure(1)  # 如果不传入参数默认画板1
            # # 第2步创建画纸，并选择画纸1
            # ax1 = plt.subplot(2, 1, 1)
            # ax1.set_title("pre_train")
            # # 在画纸1上绘图
            # plt.plot(train_loss)
            # # 选择画纸2
            # ax2 = plt.subplot(2, 1, 2)
            # ax2.set_title("pre_test")
            # # 在画纸2上绘图
            # plt.plot(test_loss)
            # # 显示图像
            # plt.show()

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=80, help="interval between image sampling")
    parser.add_argument("--dataset_name", type=str, default="rf2us22", help="name of the dataset")
    parser.add_argument("--rfdata_len", type=int, default=1024, help="length of rf data")
    parser.add_argument("--split_test", type=bool, default=True, help="if split test")
    parser.add_argument("--network", type=str, default="cgan", help="if split test")
    parser.add_argument("--use_encoder", type=str, default="ae6/autoencoder_150.pth", help="if split test")
    parser.add_argument("--use_gpu", type=int, default=0, help="use gpu id")
    opt = parser.parse_args()
    print(opt)
    with torch.cuda.device(opt.use_gpu):
        cgan = ConditionGan(opt)
        cgan.TestModelLoss("rf2us21")
        cgan.GengerateImg("rf2us21")
        #GengerateImg()
