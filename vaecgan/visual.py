import numpy as np
import matplotlib.pyplot as plt

def ShowModelLoss():

    datasetname = "rf2us6"

    train_loss = np.load(f'./loss/{datasetname}/train_loss.npy')

    loss_D = []
    loss_G = []
    loss_P = []
    loss_GAN = []

    index = 0

    for loss in train_loss:
        loss_D.append(loss[0])
        loss_G.append(loss[1])
        loss_P.append(loss[3])
        loss_GAN.append(loss[4])
        index = index + 1
        # if index >= 150:
        #     break

    fig = plt.figure(1)
    fig.suptitle(datasetname)

    ax1=plt.subplot(2,2,1)
    ax1.set_title("Loss D")
    plt.plot(loss_D)
    ax2=plt.subplot(2,2,2)
    ax2.set_title("Loss G")
    plt.plot(loss_G)

    ax3=plt.subplot(2,2,3)
    ax3.set_title("Loss Pix")
    plt.plot(loss_P)

    ax4=plt.subplot(2,2,4)
    ax4.set_title("Loss Gan")
    plt.plot(loss_GAN)

    #显示图像
    plt.show()

    # index = np.where(train_loss == np.min(train_loss))
    # print(f'train min loss : {model_names[index]}')
    # index = np.where(test_loss == np.min(test_loss))
    # print(f'test min loss : {model_names[index]}')

if __name__ == '__main__':
    ShowModelLoss()
