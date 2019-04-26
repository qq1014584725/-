import random
import matplotlib.pyplot as plt
import numpy as np

#构造数据
def data_set():
    y = []
    labels = []
    random.seed(10)
    for x in range(10):
        nosie = random.uniform(-1, 1)
        y.append([x*0.1, 2 * x * 0.1 + 1 + nosie])
        if nosie >= 0:
            labels.append(1)
        else:
            labels.append(-1)

    return (y,labels)

#训练模型
def train(data):
    X, labels = data

    #初始化参数 w,b
    w_len = len(X[0])
    W = np.zeros((w_len, 1))
    b = 0

    #设置迭代参数
    count = 0             #记录迭代次数
    out_count2 = 0         #记录一轮若所有点都满足条件则跳出循环
    limit_count = 10000       #极限迭代论述若极限论数未迭代完成退出循环
    aph = 0.05              #学习率

    #开始训练迭代
    while True:
        out_count2 += 1
        out_count1 = 0        #记录一轮迭代次数便于观察模型训练完毕退出循环
        for x,label in zip(X,labels):
            x = np.array(x)

            #分类错误就开始迭代
            if (np.dot(x, W) + b)*label <= 0:
                count += 1

                #重新计算w,b
                if label < 0:
                    W = W + aph * (-1) * x.reshape(-1, 1)
                    b = b + aph * (-1)
                else:
                    W = W + aph * (1) * x.reshape(-1, 1)
                    b = b + aph * 1

            else:
                out_count1 += 1

        #若所有点满足条件循环
        if out_count1 == len(X):
            break
        #若到达极限论数跳出循环
        if out_count2 > limit_count:
            break

    print(count)
    print(out_count1)
    print(out_count2)

    return (W,b)

if __name__ == '__main__':
    y,labels = data_set()
    print(y)
    print(labels)
    #
    # #数据散点图
    y = np.array(y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(y[:, 0], y[:, 1])
    plt.plot(y[:, 0], y[:, 0] * 2 + 1)
    # plt.show()

    W,b = train(data_set())
    print(W,b)

    X = np.arange(0,10)
    X = X*0.1
    print(X)
    Y = ((-b) - (W[0])*X)/W[1]
    plt.plot(X,Y,'r')
    plt.show()