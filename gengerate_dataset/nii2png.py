import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像


def nii_to_image(rootpath,niiname):

    num = niiname[0:3]
    # print(num)

    mask_dir = "./MRI_volunteer_dataset/MRI_thyroid+jugular+carotid_label"

    mask_list = os.listdir(mask_dir)

    mask_name = None
    mask = None

    for m in mask_list:
        # print(m)
        if (niiname.replace(".nii","")) in m:
            mask_name = m
            mask = nib.load(os.path.join(mask_dir,mask_name))
            break

    if mask is None:
        print("mask not found.")
        return
    # else:
    #     print(mask.get_fdata())

    slice_trans = []

    save_dir = "./MRI_volunteer_dataset/MRI_PNG/" + num

    nii_path = os.path.join(rootpath,niiname)

    img = nib.load(nii_path)  # 读取nii

    img_fdata = img.get_fdata()
    mask_fdata = mask.get_fdata()

    max_val = np.max(img_fdata)
    min_val = np.min(img_fdata)

    img_fdata = (img_fdata - min_val) * 255/(max_val - min_val)
    img_data = img_fdata.astype(np.uint8)
    mask_data = mask_fdata.astype(np.uint8)

    # print(np.max(mask_data))
    # print(np.min(mask_data))
    print(np.shape(mask_data))

    os.makedirs(save_dir, exist_ok=True)

    (x, y, z) = img.shape

    for i in range(x):
        silce_x = img_data[i, :, :]
        mask_x = mask_data[i, :, :]
        w_mask = np.uint8(mask_x > 0) * 255

        # break
        # print(np.shape(mask_x))
        if 2 in mask_x or 4 in mask_x:
            # print("Find Mask")
            pic_name = f'{niiname.replace(".nii", "")}_x_silce{i}.png'
            imageio.imwrite(os.path.join(save_dir, pic_name), silce_x)
            mask_name_save = f'{mask_name.replace(".nii", "")}_x_silce{i}.png'
            imageio.imwrite(os.path.join(save_dir, mask_name_save), w_mask)

    for i in range(y):
        silce_y = img_data[:, i, :]
        mask_y = mask_data[:, i, :]
        w_mask = np.uint8(mask_y > 0) * 255

        # break
        # print(np.shape(mask_x))
        if 2 in mask_y or 4 in mask_y:
            # print("Find Mask")
            pic_name = f'{niiname.replace(".nii", "")}_y_silce{i}.png'
            imageio.imwrite(os.path.join(save_dir, pic_name), silce_y)
            mask_name_save = f'{mask_name.replace(".nii", "")}_y_silce{i}.png'
            imageio.imwrite(os.path.join(save_dir, mask_name_save), w_mask)


    for i in range(z):
        silce_z = img_data[:, :, i]
        mask_z = mask_data[:, :, i]
        w_mask = np.uint8(mask_z > 0) * 255

        # break
        # print(np.shape(mask_x))
        if 2 in mask_z or 4 in mask_z:
            # print("Find Mask")
            pic_name = f'{niiname.replace(".nii", "")}_z_silce{i}.png'
            imageio.imwrite(os.path.join(save_dir, pic_name), silce_z)
            mask_name_save = f'{mask_name.replace(".nii", "")}_z_silce{i}.png'
            imageio.imwrite(os.path.join(save_dir, mask_name_save), w_mask)

    # for i in range(y):
    #
    #     slice_y = img_data[:, i, :]  # 选择哪个方向的切片都可以
    #
    #
    #     pic_name = f'{niiname.replace(".nii", "")}_y_silce{i}.png'
    #     imageio.imwrite(os.path.join(save_path, pic_name), slice_y)



# if __name__ == '__main__':
#
#     rootpath = './MRI_volunteer_dataset/MRI/'
#     filenames = os.listdir(rootpath)
#
#     for file in filenames:
#         nii_to_image(rootpath,file)


    #处理单个文件
    #filepath = './training/training/patient001'
    # imgfile = './iamge'
    # print(filepath)
    # nii_to_image(filepath)

import cv2
from PIL import Image

# bmpimg = Image.open('./001.bmp')
pngimg = Image.open('./001.png').convert("RGB")
#
# print(bmpimg)
# print(pngimg)


# pngimg.resize((256,256),Image.ANTIALIAS)

pngimg = pngimg.resize((256,256),Image.BICUBIC)

pngimg = pngimg.convert('L')

pngimg.save("./gray001.bmp", 'bmp')


# if __name__ == '__main__':
#     rootDir = "."
#     imgStorePath = rootDir
#
#     if (not os.path.exists(imgStorePath)):
#         os.makedirs(imgStorePath) #创建目录
#
#     # a_list = fnmatch.filter(os.listdir(rootDir),'*.png')
#
#     path = rootDir + '/' + "001.png"
#     # 开始读取
#     img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#     # 直接调用.tofile(_path),我这里显示权限拒绝，所用系统自带的文件写入
#     img_encode = cv2.imencode('.bmp', img)[1]
#
#     # img = cv2.imread(path)
#
#     with open(imgStorePath + '/001.bmp', 'wb') as f:  # 写入
#         f.write(img_encode)

    # for i in range(len(a_list)):
    #     path = rootDir+'/'+a_list[i]
    #     # 开始读取
    #     img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    #     # 直接调用.tofile(_path),我这里显示权限拒绝，所用系统自带的文件写入
    #     img_encode = cv2.imencode('.bmp',img)[1]
    #
    #     # img = cv2.imread(path)
    #
    #     t = a_list[i]
    #     t = t[:-4]  # 拿到图片名
    #     with open(imgStorePath + t + '.bmp', 'wb') as f: #写入
    #         f.write(img_encode)
