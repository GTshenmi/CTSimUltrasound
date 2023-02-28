import os
import random
import shutil
# def GengerateTest():
#
#     label_dir = "./dataset/train/label/"
#     input_dir = "./dataset/train/input/"
#     label_list = os.listdir(label_dir)
#
#     label_len = len(label_list)
#
#     test_len = int(label_len*0.3)
#
#     print(len(label_list))
#
#     sample = random.sample(label_list, test_len)
#
#     for picname in sample:
#         input_path = os.path.join(input_dir,picname)
#         label_path = os.path.join(label_dir,picname)
#         shutil.move(input_path,input_path.replace("train","test"))
#         shutil.move(label_path,label_path.replace("train","test"))
#
#     label_list = os.listdir(label_dir)
#     test_list = os.listdir(label_dir.replace("train","test"))
#     print(len(label_list))
#     print(len(test_list))
#
# def GengerateRFTest():
#
#     label_dir = "./dataset/train/rf_label/"
#     rf_dir = "./dataset/train/rf_data/"
#     label_list = os.listdir(label_dir)
#
#     label_len = len(label_list)
#
#     test_len = int(label_len*0.3)
#
#     print(len(label_list))
#
#     sample = random.sample(label_list, test_len)
#
#     for rfname in sample:
#         rf_path = os.path.join(rf_dir,rfname.replace(".png",""))
#         label_path = os.path.join(label_dir,rfname)
#         shutil.move(rf_path,rf_path.replace("train","test"))
#         shutil.move(label_path,label_path.replace("train","test"))
#
#     label_list = os.listdir(label_dir)
#     test_list = os.listdir(label_dir.replace("train","test"))
#     print(len(label_list))
#     print(len(test_list))


def GengerateTest():

    label_dir = "./dataset/train/label/"
    rf_dir = "./dataset/train/rf_data/"
    input_dir = "./dataset/train/input/"

    label_list = os.listdir(label_dir)

    label_len = len(label_list)

    test_len = int(label_len*0.3)

    print(len(label_list))

    sample = random.sample(label_list, test_len)

    for labelname in sample:
        rf_path = os.path.join(rf_dir,labelname.replace(".png",""))
        label_path = os.path.join(label_dir,labelname)
        input_path = os.path.join(input_dir, labelname)

        shutil.move(rf_path,rf_path.replace("train","test"))
        shutil.move(label_path,label_path.replace("train","test"))
        shutil.move(input_path, input_path.replace("train", "test"))

    label_list = os.listdir(label_dir)
    test_list = os.listdir(label_dir.replace("train","test"))
    print(len(label_list))
    print(len(test_list))
    label_list = os.listdir(rf_dir)
    test_list = os.listdir(rf_dir.replace("train","test"))
    print(len(label_list))
    print(len(test_list))
    label_list = os.listdir(input_dir)
    test_list = os.listdir(input_dir.replace("train","test"))
    print(len(label_list))
    print(len(test_list))



import numpy as np
# a= ["aaaaa","bbbbb",5]
# a.append(6)
# np.save('a.npy',a)   # 保存为.npy格式
#
# a=np.load('a.npy')
# print(a)
# name = "rf2us20"
#
# print(os.path.join("./images/",name,"train"))
# epoch = 100
# name = "ae01"
# print("./saved_models/{}/AutoEncoder01_{}.pth".format(name,(epoch + 1)))

# loss = np.array([1,2,3,4,5])
#
# train_loss = np.array([6,7,8,9,10])
#
#
# loss = np.vstack((loss,train_loss))
# loss = np.vstack((loss,train_loss+5))
# loss = np.vstack((loss,train_loss+10))
#
# print(loss)

# f = open("./datasetnew/us_image_val.txt", "r",encoding='utf-8')
# train_list = f.read().splitlines()
# #print(train_list)
# f.close()
#
# for train in train_list:
#     if "patient117_frame01_slice9" in train:
#         print(True)

array = np.array([0,1,2,3,4])
array = np.vstack((array,array+5))
array = np.vstack((array,array+5))

print(0 in array)
print(5 in array)
print(15 in array)
print(100 in array)

#
# print(np.isin(array,0))
# print(np.isin(array,5))
# print(np.isin(array,15))
# print(np.isin(array,100))