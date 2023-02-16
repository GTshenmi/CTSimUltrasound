import torch
#import visdom
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
import argparse
import os
import time
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=60, help="interval between image sampling")
parser.add_argument("--dataset_name", type=str, default="ae7", help="name of the dataset")
parser.add_argument("--rfdata_len", type=int, default=1024, help="length of rf data")
parser.add_argument("--split_test", type=bool, default=True, help="if split test")
parser.add_argument("--network", type=str, default="ae", help="network")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--device",type = str,default="gpu",help="train device")

opt = parser.parse_args()

#opt.device = "cpu"

print(opt)
input_shape = (128,opt.rfdata_len)
output_shape = (opt.img_size,opt.img_size)

cuda = True if (torch.cuda.is_available() and opt.device == "gpu") else False

device = torch.device("cuda" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 128, 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 128, 1024
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor = 2,mode = "bilinear"),# b, 128, 256, 2048
            nn.MaxPool2d((1,2)),  # b, 128, 256, 1024

            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b, 64, 256, 1024
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,8, 3, stride=1, padding=1),  # b, 4, 256, 1024
            nn.InstanceNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((1, 4)),  # b, 4, 256, 256
        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(8, 64,(3,4), stride=(1,4), padding=(1,0)),  # b, 64, 256, 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),  # b, 128, 256, 1024
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, (3,4), stride=(1,2), padding=(1,1)),  # b, 128, 256, 2048
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor = 0.5,mode = "bilinear",recompute_scale_factor = True),# b, 128, 128, 1024

            # nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 64, 128, 1024
            # nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),  # b, 1, 128, 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
        )


    def forward(self, x):
        enc = self.encoder(x)
        #print(x.shape)
        dec = self.decoder(enc)
        #print(x.shape)
        return enc,dec


if __name__ == '__main__':
    #TrainModelAE()
    #device = torch.device('cpu')
    from dataset import MyDataSet

    os.makedirs("images", exist_ok=True)
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)


    rf_transforms = [
        transforms.ToTensor(),

        # transforms.Normalize(0.5,0.5),
    ]

    img_transforms = [

        transforms.Resize((opt.img_size, opt.img_size)),

        transforms.ToTensor(),

        # transforms.Normalize(0.5,0.5),
    ]
    dataloader = DataLoader(
        MyDataSet(root="./dataset/", opt=opt, img_transform=img_transforms, rf_transform=rf_transforms,use_embedding=False),
        batch_size=opt.batch_size,
        #batch_size=1,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    val_dataloader = DataLoader(
        MyDataSet(root="./dataset/", opt=opt, img_transform=img_transforms, rf_transform=rf_transforms, mode="test",use_embedding=False),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )


    model = autoencoder()
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print(model)

    # viz = visdom.Visdom()
    if cuda:
        model.cuda()
        criteon.cuda()


    def sample_result(batches_done):

        """Saves a generated sample from the validation set"""
        datas = next(iter(val_dataloader))

        rf_data = datas["rf_data"].type(Tensor)

        _,result = model(rf_data)

        loss = criteon(result, rf_data)

        return loss


    def TrainModelAE():
        test_loss = 0
        prev_time = time.time()
        for epoch in range(opt.n_epochs):
            for i, batch in enumerate(dataloader):
                rf_data = batch["rf_data"].type(Tensor)

                # torch.set_printoptions(precision=20)
                # print(rf_data)

                _,x_hat = model(rf_data)
                loss = criteon(x_hat, rf_data)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Train loss: %f] [Test loss: %f] ETA: %s\r\n"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss.item(),
                        test_loss,
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % opt.sample_interval == 0:
                    test_loss = sample_result(batches_done)

            if epoch <= 120:
                if ((epoch + 1) % 5 == 0):
                    torch.save(model, "./saved_models/{}/AutoEncoder_{}.pth".format(opt.dataset_name, (epoch + 1)))
            else:
                if ((epoch + 1) % 2 == 0):
                    torch.save(model, "./saved_models/{}/AutoEncoder_{}.pth".format(opt.dataset_name, (epoch + 1)))

    def TestModel():
        pre_train = []
        pre_test = []
        with torch.no_grad():
            autoencoder = torch.load("./saved_models/ae6/AutoEncoder_150.pth")
            #autoencoder.to(device)
            autoencoder.eval()

            for i, batch in enumerate(dataloader):
                rf_data = batch["rf_data"].type(Tensor)

                name = batch["name"][0]
                save_path = os.path.join("./dataset/train/rf_embedding/",name+".npy")

                rf_embedding,decoder_rf = autoencoder(rf_data)

                #print(rf_embedding)

                loss = criteon(decoder_rf, rf_data)
                pre_train.append(Tensor.cpu(loss))

                rf_embedding = rf_embedding.view(8,opt.img_size,opt.img_size)

                rf_embedding = torch.permute(rf_embedding,(1, 2, 0))

                rf_embedding = Tensor.numpy(Tensor.cpu(rf_embedding))

                np.save(save_path,rf_embedding)

                sys.stdout.write(
                    "\r[%s] [Batch %d/%d] [loss: %f]\r\n"
                    % (
                        name,
                        i,
                        len(dataloader),
                        loss,
                    )
                )

            for i, batch in enumerate(val_dataloader):
                rf_data = batch["rf_data"].type(Tensor)

                name = batch["name"][0]
                save_path = os.path.join("./dataset/test/rf_embedding/", name + ".npy")

                rf_embedding, decoder_rf = autoencoder(rf_data)

                loss = criteon(decoder_rf, rf_data)
                pre_test.append(Tensor.cpu(loss))

                rf_embedding = rf_embedding.view(8, opt.img_size, opt.img_size)

                rf_embedding = torch.permute(rf_embedding,(1, 2, 0))

                rf_embedding = Tensor.numpy(Tensor.cpu(rf_embedding))

                np.save(save_path, rf_embedding)

                sys.stdout.write(
                    "\r[%s] [Batch %d/%d] [loss: %f]\r\n"
                    % (
                        name,
                        i,
                        len(val_dataloader),
                        loss,
                    )
                )

            fig = plt.figure(1) #如果不传入参数默认画板1
            #第2步创建画纸，并选择画纸1
            ax1=plt.subplot(2,1,1)
            ax1.set_title("pre_train")
            #在画纸1上绘图
            plt.plot(pre_train)
            #选择画纸2
            ax2=plt.subplot(2,1,2)
            ax2.set_title("pre_test")
            #在画纸2上绘图
            plt.plot(pre_test)
            #显示图像
            plt.show()

    def TestModelLoss(modelname):
        train_loss = []
        test_loss = []
        #rootpath = "./saved_models/ae6"
        rootpath = os.path.join("./saved_models/",modelname)
        #rootpath = "C:/Users/IzumiSagiri/Desktop/ae6/"

        model_list = os.listdir(rootpath)

        with torch.no_grad():
            for model in model_list:

                tmp_loss = []
                autoencoder = torch.load(os.path.join(rootpath,model))
                autoencoder.eval()

                for i, batch in enumerate(dataloader):
                    rf_data = batch["rf_data"].type(Tensor)

                    name = batch["name"][0]
                    save_path = os.path.join("./dataset/train/rf_embedding/",name+".npy")

                    _,decoder_rf = autoencoder(rf_data)

                    loss = criteon(decoder_rf, rf_data)
                    tmp_loss.append(Tensor.cpu(loss))

                    sys.stdout.write(
                        "\r[%s] [%s] [Batch %d/%d] [loss: %f]\r\n"
                        % (
                            model,
                            name,
                            i,
                            len(dataloader),
                            loss,
                        )
                    )


                train_loss.append(np.mean(tmp_loss))
                tmp_loss = []

                for i, batch in enumerate(val_dataloader):
                    rf_data = batch["rf_data"].type(Tensor)

                    name = batch["name"][0]
                    save_path = os.path.join("./dataset/test/rf_embedding/", name + ".npy")

                    _, decoder_rf = autoencoder(rf_data)

                    loss = criteon(decoder_rf, rf_data)
                    tmp_loss.append(Tensor.cpu(loss))

                    sys.stdout.write(
                        "\r[%s] [%s] [Batch %d/%d] [loss: %f]\r\n"
                        % (
                            model,
                            name,
                            i,
                            len(val_dataloader),
                            loss,
                        )
                    )

                test_loss.append(np.mean(tmp_loss))

            np.save(f'./loss/{modelname}_testloss.npy', test_loss)
            np.save(f'./loss/{modelname}_trainloss.npy', train_loss)
            np.save(f'./loss/{modelname}_modelnames',model_list)

            fig = plt.figure(1) #如果不传入参数默认画板1
            #第2步创建画纸，并选择画纸1
            ax1=plt.subplot(2,1,1)
            ax1.set_title("train loss")
            #在画纸1上绘图
            plt.plot(train_loss)
            #选择画纸2
            ax2=plt.subplot(2,1,2)
            ax2.set_title("test loss")
            #在画纸2上绘图
            plt.plot(test_loss)
            #显示图像
            plt.show()

            index = np.where(train_loss == np.min(train_loss))
            print(f'train min loss : {model_list[index]}')
            index = np.where(test_loss == np.min(test_loss))
            print(f'test min loss : {model_list[index]}')


    TestModel()
    #TrainModelAE()
    #TestModelLoss("ae6")



