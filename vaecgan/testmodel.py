import torch
import os
from datasetnew import MyDataSet
import argparse
import sys
import numpy as np

from itertools import chain
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
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
parser.add_argument("--use_gpu", type=int, default=1, help="use gpu id")
parser.add_argument("--resume", type=bool, default=True, help="if resume train")

opt = parser.parse_args()

# def GengerateImg(self, model_name):
#     rootpath = "./saved_models"
#     best_model = ShowModelLoss(model_name)
#     # best_model = "generator_110.pth"
#     name = model_name
#     # model_path = os.path.join(rootpath,best_model)
#     print(best_model)
#     save_path_train = os.path.join("../images/", name, "train")
#     save_path_test = os.path.join("../images/", name, "test")
#
#     os.makedirs(save_path_train, exist_ok=True)
#     os.makedirs(save_path_test, exist_ok=True)
#
#     train_loss = []
#     test_loss = []
#
#     with torch.no_grad():
#         generator = torch.load(os.path.join(rootpath, name, best_model))
#         generator.cuda(opt.use_gpu)
#         generator.eval()
#
#         for i, batch in enumerate(dataloader):
#             real_imgs = batch["us"].type(Tensor)
#             conditions = batch["rf_data"].type(Tensor)
#             name = batch["name"][0]
#
#             fake_imgs = generator(conditions)
#
#             loss_pixel = criterion_pixelwise(fake_imgs, real_imgs)
#
#             img_sample = torch.cat((fake_imgs.data, real_imgs.data), -2)
#
#             save_image(img_sample, "%s/train_%s.png" % (save_path_train, i), nrow=5, normalize=True)
#
#             train_loss.append(Tensor.cpu(loss_pixel))
#
#             sys.stdout.write(
#                 "\r[Batch %d/%d] [loss: %f]\r\n"
#                 % (
#                     i,
#                     len(dataloader),
#                     loss_pixel,
#                 )
#             )
#
#         for i, batch in enumerate(val_dataloader):
#             real_imgs = batch["us"].type(Tensor)
#             conditions = batch["rf_data"].type(Tensor)
#
#             name = batch["name"][0]
#
#             fake_imgs = generator(conditions)
#
#             loss_pixel = criterion_pixelwise(fake_imgs, real_imgs)
#
#             img_sample = torch.cat((fake_imgs.data, real_imgs.data), -2)
#
#             save_image(img_sample, "%s/test_%s.png" % (save_path_test, i), nrow=5, normalize=True)
#
#             test_loss.append(Tensor.cpu(loss_pixel))
#
#             sys.stdout.write(
#                 "\r[Batch %d/%d] [loss: %f]\r\n"
#                 % (
#                     i,
#                     len(val_dataloader),
#                     loss_pixel,
#                 )
#             )
#
#         # fig = plt.figure(1)  # 如果不传入参数默认画板1
#         # # 第2步创建画纸，并选择画纸1
#         # ax1 = plt.subplot(2, 1, 1)
#         # ax1.set_title("pre_train")
#         # # 在画纸1上绘图
#         # plt.plot(train_loss)
#         # # 选择画纸2
#         # ax2 = plt.subplot(2, 1, 2)
#         # ax2.set_title("pre_test")
#         # # 在画纸2上绘图
#         # plt.plot(test_loss)
#         # # 显示图像
#         # plt.show()

def GenerateImg(modelname):

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
        batch_size=4,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        MyDataSet(root="../datasetnew/", opt=opt, img_transform=img_transforms, rf_transform=rf_transforms,
                  mode="test"),
        batch_size=4,
        shuffle=True,
        num_workers=4,
    )

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = torch.device("cuda:" + str(opt.use_gpu) if cuda else "cpu")
    lambda_pixel = 100
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_AE = torch.nn.MSELoss()
    patch = (1, opt.img_size // 2 ** 4, opt.img_size // 2 ** 4)

    train_loss = []
    test_loss = []

    #name = modelname

    rootpath = os.path.join("./saved_models/", modelname)
    model_list = os.listdir(rootpath)

    with torch.no_grad():
        for model in model_list:
            if "generator" in model:
                save_path_train = os.path.join("./images/", modelname,model.replace(".pth",""), "train")
                save_path_test = os.path.join("./images/", modelname,model.replace(".pth",""), "test")

                os.makedirs(save_path_train, exist_ok=True)
                os.makedirs(save_path_test, exist_ok=True)
                #print(model)
                generator = torch.load(os.path.join(rootpath, model))
                autoencoder = torch.load(os.path.join(rootpath, model.replace("generator","autoencoder")))
                discriminator = torch.load(os.path.join(rootpath, model.replace("generator", "discriminator")))

                generator.to(device)
                autoencoder.to(device)
                discriminator.to(device)

                tmp_loss = []

                generator.eval()
                autoencoder.eval()
                discriminator.eval()

                for i, batch in enumerate(val_dataloader):
                    real_imgs = batch["us"].type(Tensor)
                    rf_datas = batch["rf_data"].type(Tensor)
                    dataname = batch["name"][0]

                    batch_size = real_imgs.shape[0]

                    valid = Tensor(np.ones((batch_size, *patch)))

                    encoder_rf, decoder_rf = autoencoder(rf_datas)

                    fake_imgs = generator(encoder_rf)

                    pred_fake = discriminator(encoder_rf, fake_imgs)

                    loss_pixel = criterion_pixelwise(fake_imgs, real_imgs)

                    loss_AE = criterion_AE(decoder_rf, rf_datas)

                    loss_GAN = criterion_GAN(pred_fake, valid)

                    loss_G = loss_GAN + lambda_pixel * loss_pixel + loss_AE

                    tmp_loss.append(Tensor.cpu(loss_G))

                    img_sample = torch.cat((fake_imgs.data, real_imgs.data), -2)

                    save_image(img_sample, "%s/test_%s.png" % (save_path_test, i), nrow=5, normalize=True)

                    sys.stdout.write(
                        "\r[%s] [%s] [Batch %d/%d] [loss: %f]\r\n"
                        % (
                            model,
                            dataname,
                            i,
                            len(dataloader),
                            loss_G,
                        )
                    )

                test_loss.append(np.mean(tmp_loss))

                np.save(f'./loss/{modelname}/testloss.npy', test_loss)

            np.save(f'./loss/{modelname}/model_name.npy', model_list)

print(opt)
torch.cuda.set_device(opt.use_gpu)
with torch.cuda.device(opt.use_gpu):
    GenerateImg("rf2us1")
