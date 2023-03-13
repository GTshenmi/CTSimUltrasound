import numpy as np
from PIL import Image

# arr1 = np.zeros((10,256,256))
#
# arr2 = arr1[0]
# for i in range(1,10):
#     arr2 = np.concatenate((arr2,arr1[i]),axis=1)
#
# print(arr2.shape)
model_path="/home/xuepeng/ultrasound/neural_networks/iddpm/run_record/openai-2023-03-03-17-23-23-938923/ema_0.9999_200000.pt"

print(model_path.split("/")[-1].replace(".pt",""))
# arr1 = np.zeros((1,256,256))
# print(arr1.shape)
# arr2 = np.ones((1,256,256))
#
# arr3 = np.concatenate((arr1,arr2), axis=0)
#
# print(arr3.shape)
#
# arr4 = np.concatenate((arr3[0],arr3[1]), axis=1)
# # print(arr4)
# print(arr4.shape)
#
# print(type((arr1,arr2)))



# arr4 = np.concatenate(arr3,axis=2)
# print(arr4.shape)

# arr1 = np.zeros((256,256,1))
# print(arr1.shape)
# arr2 = np.ones((256,256,1))
#
# arr3 = np.concatenate((arr1,arr2), axis=2)
#
# print(arr3.shape)
#
# arr4 = np.concatenate(arr3,axis=2)
# print(arr4.shape)

# samplefile = np.load("./run_record/openai-2023-03-07-14-45-37-359032/samples_1x256x256x1.npz")
#
# print(samplefile.files)
#
# # print(samplefile['arr_0'].resize())
#
# imgs = np.squeeze(samplefile['arr_0'],0)
#
# imgs = np.squeeze(imgs,2)
#
# print(imgs.shape)
#
# imgs = Image.fromarray(imgs)
# imgs.save("test.png")

