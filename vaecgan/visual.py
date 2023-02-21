import numpy as np
import matplotlib.pyplot as plt

def ShowModelLoss():

    train_loss = np.load("./loss/rf2us1/train_loss.npy")

    loss_D = []
    loss_G = []

    for loss in train_loss:
        loss_D.append(loss[0])
        loss_G.append(loss[1])

    fig = plt.figure(1) #如果不传入参数默认画板1
    #第2步创建画纸，并选择画纸1
    ax1=plt.subplot(2,1,1)
    ax1.set_title("Loss D")
    #在画纸1上绘图
    plt.plot(loss_D)
    #选择画纸2
    ax2=plt.subplot(2,1,2)
    ax2.set_title("Loss G")
    #在画纸2上绘图
    plt.plot(loss_G)
    #显示图像
    plt.show()

    # index = np.where(train_loss == np.min(train_loss))
    # print(f'train min loss : {model_names[index]}')
    # index = np.where(test_loss == np.min(test_loss))
    # print(f'test min loss : {model_names[index]}')

if __name__ == '__main__':
    ShowModelLoss()
