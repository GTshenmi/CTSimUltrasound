#with torch.cuda.device(opt.use_gpu):
    # Configure dataloaders
    # rf_transforms = [
    #     transforms.ToTensor(),
    #
    #     #transforms.Normalize(0.5,0.5),
    # ]



    # img_transforms = [
    #
    #     transforms.Resize((opt.img_size, opt.img_size)),
    #
    #     transforms.ToTensor(),
    #
    #     transforms.Normalize(0.5,0.5),
    # ]

    # dataloader = DataLoader(
    #     MyDataSet(root="./dataset/",opt = opt, img_transform=img_transforms,rf_transform=rf_transforms,use_embedding=True),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.n_cpu,
    # )
    #
    # val_dataloader = DataLoader(
    #     MyDataSet(root="./dataset/",opt = opt, img_transform=img_transforms,rf_transform=rf_transforms,mode="test"),
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=1,
    # )

    # # Initialize generator and discriminator
    # generator = Generator()
    # discriminator = Discriminator()
    # # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    #
    # lambda_pixel = 100
    # criterion_GAN = torch.nn.MSELoss()
    # criterion_pixelwise = torch.nn.L1Loss()
    #
    # generator.to(device)
    # discriminator.to(device)
    # criterion_GAN.to(device)
    # criterion_pixelwise.to(device)

# if cuda:
#     generator.cuda(opt.use_gpu)
#     discriminator.cuda(opt.use_gpu)
#     criterion_GAN.cuda(opt.use_gpu)
#     criterion_pixelwise.cuda(opt.use_gpu)
#
#
# def sample_images(batches_done,train_real,train_fake):
#     """Saves a generated sample from the validation set"""
#     datas = next(iter(val_dataloader))
#
#     # real_imgs = Variable(datas["us"].type(Tensor))
#     # conditions = Variable(datas["rf_data"].type(Tensor))
#
#     real_imgs = datas["us"].type(Tensor)
#     conditions = datas["rf_data"].type(Tensor)
#
#     #batch_size = datas["us"].shape[0]
#
#     #z = Variable(Tensor(np.random.normal(0, 1, (batch_size, *img_shape))))
#
#     fake_imgs = generator(conditions)
#
#     img_sample = torch.cat((fake_imgs.data,real_imgs.data), -2)
#     img_sample_train = torch.cat((train_fake.data,train_real.data), -2)
#     #img_sample = torch.cat((fake_imgs.data, imgs.data), -2)
#
#     save_image(img_sample, "images/%s/test_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)
#     save_image(img_sample_train,"images/%s/train_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)
#
# # ----------
# #  Training
# # ----------
#
# def TrainModel():
#     prev_time = time.time()
#     print(generator)
#     print(discriminator)
#     for epoch in range(opt.n_epochs):
#         for i, batch in enumerate(dataloader):
#
#             # Model inputs
#             labels = batch["us"]
#             rf_datas = batch["rf_data"]
#
#             # print("rf_datas")
#             # print(rf_datas.shape)
#
#             batch_size = labels.shape[0]
#
#             #print(batch_size)
#
#             #Adversarial ground truths
#             # valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
#             # fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
#
#             # valid = Variable(Tensor(np.ones((batch_size, *patch))), requires_grad=False)
#             # fake = Variable(Tensor(np.zeros((batch_size, *patch))), requires_grad=False)
#             valid =Tensor(np.ones((batch_size, *patch)))
#             fake = Tensor(np.zeros((batch_size, *patch)))
#
#             # Configure input
#             # real_imgs = Variable(labels.type(Tensor))
#             # conditions = Variable(rf_datas.type(Tensor))
#             real_imgs = labels.type(Tensor)
#             conditions = rf_datas.type(Tensor)
#             #inputs = Variable(imgs.type(Tensor))
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#
#             optimizer_G.zero_grad()
#
#             #z = Variable(Tensor(np.random.normal(0, 1, (batch_size, *img_shape))))
#
#             fake_imgs = generator(conditions)
#
#             pred_fake = discriminator(conditions,fake_imgs)
#             loss_GAN = criterion_GAN(pred_fake, valid)
#             # Pixel-wise loss
#             loss_pixel = criterion_pixelwise(fake_imgs, real_imgs)
#
#             # Total loss
#             loss_G = loss_GAN + lambda_pixel * loss_pixel
#
#             loss_G.backward()
#
#             optimizer_G.step()
#
#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#
#             optimizer_D.zero_grad()
#
#             # Real loss
#             pred_real = discriminator(conditions,real_imgs)
#             loss_real = criterion_GAN(pred_real, valid)
#
#             # Fake loss
#             pred_fake = discriminator(conditions,fake_imgs.detach())
#             loss_fake = criterion_GAN(pred_fake, fake)
#
#             # Total loss
#             loss_D = 0.5 * (loss_real + loss_fake)
#
#             loss_D.backward()
#             optimizer_D.step()
#
#             # --------------
#             #  Log Progress
#             # --------------
#
#             # Determine approximate time left
#             batches_done = epoch * len(dataloader) + i
#             batches_left = opt.n_epochs * len(dataloader) - batches_done
#             time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
#             prev_time = time.time()
#
#             # Print log
#             sys.stdout.write(
#                 "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s\r\n"
#                 % (
#                     epoch,
#                     opt.n_epochs,
#                     i,
#                     len(dataloader),
#                     loss_D.item(),
#                     loss_G.item(),
#                     loss_pixel.item(),
#                     loss_GAN.item(),
#                     time_left,
#                 )
#             )
#
#             # If at sample interval save image
#             if batches_done % opt.sample_interval == 0:
#                 sample_images(batches_done,real_imgs,fake_imgs)
#
#         if epoch <= 120:
#             if ((epoch + 1) % 5 == 0):
#                 torch.save(generator, "./saved_models/{}/generator_{}.pth".format(opt.dataset_name, (epoch + 1)))
#         else:
#             if ((epoch + 1) % 2 == 0):
#                 torch.save(generator, "./saved_models/{}/generator_{}.pth".format(opt.dataset_name, (epoch + 1)))




def GetRFData(dataPath,opt,use_embedding):

    D = 10  # Sampling frequency decimation factor
    fs = 100e6 / D  # Sampling frequency  [Hz]
    c = 1540  # Speed of sound [m]
    no_lines = 128  # Number of lines in image
    image_width = 90 / 180 * np.pi  # Size of image sector [rad]
    dtheta = image_width / no_lines  # Increment for image

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

    D = 10
    dB_Range = 50
    max_env = max(env_max_list)
    min_env = min(env_min_list)
    log_env = []
    log_env_max_list = []
    log_env_min_list = []
    D = 10

    for i in range(len(env)):
        dB_Range = 50

        env[i] = env[i] - min_env

        tmp_env = env[i][slice(0, len(env[i]), D)] / max_env

        tmp_env = 255 / dB_Range * (tmp_env + dB_Range)

        # log_env_max_list.append(max(tmp_env))
        # log_env_min_list.append(min(tmp_env))
        log_env.append(tmp_env)

    # print(log_env)
    # log_env_max = max(log_env_max_list)
    # log_env_min = min(log_env_min_list)
    # log_env_max = log_env_max - log_env_min
    # print((log_env_min,log_env_max))
    data_len = opt.rfdata_len
    disp_env = []
    disp_env_max_list = []
    disp_env_min_list = []
    disp_env_nor = []

    for i in range(len(log_env)):
        D = int(np.floor(len(log_env[i]) / data_len))

        tmp_env = log_env[i][slice(0, data_len * D, D)]
        disp_env_max_list.append(max(tmp_env))
        disp_env_min_list.append(min(tmp_env))
        # print(tmp_env)

        #tmp_env = 255 * ((tmp_env - log_env_min) / log_env_max)
        #tmp_env = ((tmp_env - log_env_min) / log_env_max)

        disp_env.append(tmp_env)



    disp_env_max = max(disp_env_max_list)
    disp_env_min = min(disp_env_min_list)
    disp_env_max = disp_env_max - disp_env_min
    print(disp_env_max)
    print(disp_env_min)

    for i in range(len(disp_env)):
        tmp_env = disp_env[i]
        tmp_env_nor = ((tmp_env - disp_env_min) / disp_env_max)
        disp_env_nor.append(tmp_env_nor)

    print(disp_env_nor)

    disp_env_nor = np.asarray(disp_env_nor)

    #print(disp_env)

    if use_embedding:
        with torch.no_grad():
            criteon = nn.MSELoss()
            autoencoder = torch.load("./saved_models/ae1/AutoEncoder01_300.pth")
            autoencoder.eval()
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            disp_env_nor = transform(disp_env_nor).type(Tensor)
            disp_env_nor = disp_env_nor.view(1, 1, 128, data_len)
            rf_embedding = autoencoder(disp_env_nor)
            loss = criteon(rf_embedding, disp_env_nor)
            rf_embedding = rf_embedding.view(1, 128, data_len)
            rf_embedding = Tensor.numpy(Tensor.cpu(rf_embedding))
            loss = Tensor.numpy(Tensor.cpu(loss))
            # print(type(rf_embedding))
            # print(type(loss))
            return rf_embedding,loss

    return disp_env_nor