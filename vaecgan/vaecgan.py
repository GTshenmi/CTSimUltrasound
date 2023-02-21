from autoencoder import AutoEncoder
from gan import Generator,Discriminator
from datasetnew import MyDataSet

import os
import numpy as np
import time
import datetime
import sys
import argparse

from itertools import chain
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

class VAECGan():
    def __init__(self,opt):

        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.device = torch.device("cuda:" + str(opt.use_gpu) if self.cuda else "cpu")

        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.input_shape = (128, opt.rfdata_len)
        self.patch = (1, opt.img_size // 2 ** 4, opt.img_size // 2 ** 4)

        os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
        os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
        os.makedirs("loss/%s" % opt.dataset_name, exist_ok=True)

        self.rf_transforms = [
            transforms.ToTensor(),
        ]
        self.img_transforms = [

            transforms.Resize((opt.img_size, opt.img_size)),

            transforms.ToTensor(),

            transforms.Normalize(0.5,0.5),
        ]

        self.dataloader = DataLoader(
            MyDataSet(root="../datasetnew/",opt = opt, img_transform=self.img_transforms,rf_transform=self.rf_transforms),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            drop_last = True,
        )

        self.val_dataloader = DataLoader(
            MyDataSet(root="../datasetnew/",opt = opt, img_transform=self.img_transforms,rf_transform=self.rf_transforms,mode="test"),
            batch_size=2,
            shuffle=True,
            num_workers=1,
        )

        # Initialize generator and discriminator
        self.generator = Generator(img_shape=self.img_shape)
        self.autoencoder = AutoEncoder()
        self.discriminator = Discriminator()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(params=chain(self.generator.parameters(),
                                                self.autoencoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.lambda_pixel = 100
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()
        self.criterion_AE = torch.nn.MSELoss()

        self.autoencoder.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.criterion_GAN.to(self.device)
        self.criterion_pixelwise.to(self.device)

        self.opt = opt

        self.loss = []

        self.train_loss = np.zeros(5)
        self.test_loss = np.zeros(5)

    def SampleImg(self,batches_done,train_real,train_fake):
        """Saves a generated sample from the validation set"""
        datas = next(iter(self.val_dataloader))

        test_real = datas["us"].type(self.Tensor)
        rf_datas = datas["rf_data"].type(self.Tensor)

        encoder_rf,_ = self.autoencoder(rf_datas)

        test_fake = self.generator(encoder_rf)

        img_sample_test = torch.cat((test_fake.data,test_real.data), -2)
        img_sample_train = torch.cat((train_fake.data,train_real.data), -2)

        save_image(img_sample_test, "images/%s/test_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True,range = (-1.0,1.0))
        save_image(img_sample_train,"images/%s/train_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True,range = (-1.0,1.0))

    def SaveModel(self,epoch):

        if epoch <= 120:
            if ((epoch + 1) % 5 == 0):
                torch.save(self.generator,"./saved_models/{}/generator_{}.pth".format(self.opt.dataset_name, (epoch + 1)))
                torch.save(self.autoencoder,"./saved_models/{}/autoencoder_{}.pth".format(self.opt.dataset_name, (epoch + 1)))
                torch.save(self.discriminator,"./saved_models/{}/discriminator_{}.pth".format(self.opt.dataset_name, (epoch + 1)))
        else:
            torch.save(self.generator, "./saved_models/{}/generator_{}.pth".format(self.opt.dataset_name, (epoch + 1)))
            torch.save(self.autoencoder,"./saved_models/{}/autoencoder_{}.pth".format(self.opt.dataset_name, (epoch + 1)))
            torch.save(self.discriminator,"./saved_models/{}/discriminator_{}.pth".format(self.opt.dataset_name, (epoch + 1)))

    def CalTestLoss(self,generator,autoencoder,discriminator):

        test_loss = np.zeros(4)

        for i, datas in enumerate(self.val_dataloader):
            real_imgs = datas["us"].type(self.Tensor)
            rf_datas = datas["rf_data"].type(self.Tensor)
            batch_size = real_imgs.shape[0]

            valid = self.Tensor(np.ones((batch_size, *self.patch)))

            encoder_rf, decoder_rf = autoencoder(rf_datas)

            fake_imgs = generator(encoder_rf)

            pred_fake = discriminator(encoder_rf, fake_imgs)

            loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

            loss_AE = self.criterion_AE(decoder_rf, rf_datas)

            loss_GAN = self.criterion_GAN(pred_fake, valid)

            loss_G = loss_GAN + self.lambda_pixel * loss_pixel + loss_AE

            current_loss_list = []

            current_loss_list.append(self.Tensor.cpu(loss_G))
            current_loss_list.append(self.Tensor.cpu(loss_AE))
            current_loss_list.append(self.Tensor.cpu(loss_pixel))
            current_loss_list.append(self.Tensor.cpu(loss_GAN))

            current_loss = np.array(current_loss_list)

            test_loss = test_loss + current_loss

        test_loss = test_loss/len(self.val_dataloader)

        self.test_loss = np.vstack((self.test_loss, test_loss))

    def SaveLoss(self):

        np.save("loss/%s/train_loss.npy" % (opt.dataset_name),self.train_loss)
        np.save("loss/%s/test_loss.npy" % (opt.dataset_name), self.test_loss)

    def TrainModel(self):

        prev_time = time.time()
        print(self.generator)
        print(self.discriminator)
        print(self.autoencoder)
        real_imgs = None
        fake_imgs = None
        for epoch in range(self.opt.n_epochs):
            train_loss = np.zeros(5)
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
                rf_datas = rf_datas.type(self.Tensor)

                # -----------------
                #  Train Generator And AutoEncoder
                # -----------------

                self.optimizer_G.zero_grad()

                encoder_rf,decoder_rf = self.autoencoder(rf_datas)

                fake_imgs = self.generator(encoder_rf)

                # print(fake_imgs.shape)
                # print(encoder_rf.shape)

                pred_fake = self.discriminator(encoder_rf,fake_imgs)

                loss_GAN = self.criterion_GAN(pred_fake, valid)

                # Pixel-wise loss
                loss_pixel = self.criterion_pixelwise(fake_imgs, real_imgs)

                loss_AE = self.criterion_AE(decoder_rf,rf_datas)

                # Total loss
                loss_G = 10 * loss_GAN + self.lambda_pixel * loss_pixel + 10 * loss_AE

                loss_G.backward()

                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.discriminator(encoder_rf.detach(),real_imgs)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = self.discriminator(encoder_rf.detach(),fake_imgs.detach())
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
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, AE loss: %f, pixel: %f, adv: %f] ETA: %s\r\n"
                    % (
                        epoch,
                        self.opt.n_epochs,
                        i,
                        len(self.dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_AE.item(),
                        loss_pixel.item(),
                        loss_GAN.item(),
                        time_left,
                    )
                )

                current_loss_list = []

                current_loss_list.append(loss_D.item())
                current_loss_list.append(loss_G.item())
                current_loss_list.append(loss_AE.item())
                current_loss_list.append(loss_pixel.item())
                current_loss_list.append(loss_GAN.item())

                current_loss = np.array(current_loss_list)

                train_loss = train_loss + current_loss

            train_loss = train_loss/len(self.dataloader)

            self.train_loss = np.vstack((self.train_loss,train_loss))

            #self.CalTestLoss()

            self.SampleImg(epoch,real_imgs,fake_imgs)

            self.SaveModel(epoch)

        np.save("loss/%s/train_loss.npy" % (opt.dataset_name), self.train_loss)


if __name__ == '__main__':
    os.makedirs("../images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=80, help="interval between image sampling")
    parser.add_argument("--dataset_name", type=str, default="rf2us2", help="name of the dataset")
    parser.add_argument("--rfdata_len", type=int, default=1024, help="length of rf data")
    parser.add_argument("--split_test", type=bool, default=True, help="if split test")
    parser.add_argument("--network", type=str, default="aecgan", help="if split test")
    parser.add_argument("--use_gpu", type=int, default=0, help="use gpu id")

    opt = parser.parse_args()
    print(opt)

    with torch.cuda.device(opt.use_gpu):
        vaecgan = VAECGan(opt)
        vaecgan.TrainModel()

#decode:upsample+conv2d 组合代替 transposed_conv2d，可以减少 checkerboard 的产生,可以采用 pixelshuffle
#gated-conv2d
#multi-stage discriminator
#Coarse2fine
#pix2pixHD

