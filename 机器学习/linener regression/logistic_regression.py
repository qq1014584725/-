import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

#训练迭代
def Train(datas, labels):
        labels = np.reshape(labels,(-1,1))

        #初始化参数beta = [0,0,1]
        m,n = np.shape(data)
        beta = np.zeros((n,1))
        beta = beta + [[0],[0],[0.1]]

        #迭代参数设置
        train_limit_count = 150
        train_count = 0
        a = 0.05

        # 迭代
        while True:
                train_count += 1
                if train_count > train_limit_count:
                        break

                #[m,1]
                z = np.dot(datas, beta)

                # shape [m, 1]
                p1 = np.exp(z) / (1 + np.exp(z))
                # shape [m, m]
                p = np.diag((p1 * (1 - p1)).reshape(m))
                # shape [1, 3]
                first_order = -np.sum(datas * (labels - p1), 0, keepdims=True)

                # update
                beta -= np.reshape(first_order,(-1,1)) * a

                l = np.sum(labels * z + np.log(1 + np.exp(z)))
                print(l)

        print("总共迭代次数为",train_count)

        return beta


if __name__ == '__main__':
    # 输入数据 并调整数据格式
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data = [[0.697, 0.46],
            [0.774, 0.376],
            [0.634, 0.264],
            [0.608, 0.318],
            [0.556, 0.215],
            [0.403, 0.237],
            [0.481, 0.149],
            [0.437, 0.211],
            [0.666, 0.091],
            [0.243, 0.0267],
            [0.245, 0.057],
            [0.343, 0.099],
            [0.639, 0.161],
            [0.657, 0.198],
            [0.36, 0.37],
            [0.593, 0.042],
            [0.719, 0.103]]

    #数据处理 y = wx + b 转化为 y = bx 所以所有数据后面加一个1
    data = np.array(data)
    m,n = np.shape(data)
    data = np.insert(data, n, values=1, axis=1)
    # print(data)

    #根据求导的公式进行迭代
    beta = Train(data, labels)
    print(beta)

    #画出分类好的图像
    plt.xlabel('dimention')
    plt.ylabel('sugger')
    for i in range(len(labels)):
            if labels[i]:
                plt.scatter(data[i, 0], data[i, 1], c='green')
            else:
                plt.scatter(data[i, 0], data[i, 1], c= 'red')

    plt.plot([-0.2,1],[-(beta[2,0] - 0.2 * beta[0,0])/beta[1,0], -(beta[2,0] + 1 * beta[0,0])/beta[1,0]])
    plt.show()