import numpy as np
import matplotlib.pyplot as plt

def ShowModelLoss():
    # train_loss = np.load("./loss/ae6_trainloss.npy")
    # test_loss = np.load("./loss/ae6_testloss.npy")
    # model_names = np.load("./loss/ae6_modelnames.npy")
    train_loss = np.load("./loss/rf2us20_trainloss.npy")
    test_loss = np.load("./loss/rf2us20_testloss.npy")
    model_names = np.load("./loss/rf2us20_modelnames.npy")

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
    print(f'train min loss : {model_names[index]}')
    index = np.where(test_loss == np.min(test_loss))
    print(f'test min loss : {model_names[index]}')

# def GengerateImg(model_path):

if __name__ == '__main__':
    ShowModelLoss()