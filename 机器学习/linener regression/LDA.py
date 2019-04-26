import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 协方差矩阵计算 是几维的随机变量就是几乘几的矩阵
def Cov_Matrix(data, u):
    matrix = []
    for x in (data - u):
        s = x.reshape((-1, 1)) * x
        matrix.append(s)

    return sum(matrix)


# 根据数据2阶矩阵求逆函数     若果是3阶以上可以用SVD求解
def Matrix_Transform(matrix):
    Sw = np.mat(matrix)
    U, sigma, V = np.linalg.svd(Sw)
    Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T

    return Sw_inv

#投影点到直线
def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if   k == 0:      return [a, t]
    elif k == np.inf: return [0, b]
    x = (a+k*b-k*t) / (k*k+1)
    y = k*x + t
    return [x, y]

# 计算w值
def Train(train_datas, train_labels):
    postive = []
    negtive = []

    for i, j in zip(train_datas, train_labels):
        if j:
            postive.append(i)
        else:
            negtive.append(i)

    u_postive = sum(postive) / len(postive)
    u_negtive = sum(negtive) / len(negtive)

    Cov_postive = Cov_Matrix(postive, u_postive)
    Cov_negtive = Cov_Matrix(negtive, u_negtive)

    Scatter_Matrix = Cov_postive + Cov_negtive

    Scatter_Matrix_ = Matrix_Transform(Scatter_Matrix)

    w = np.dot(Scatter_Matrix_, (u_negtive - u_postive).reshape((-1, 1)))

    return (w,u_postive,u_negtive)

#预测
def Predict(test_datas, w, u_postive,  u_negtive):
    predict = []
    u0 = np.dot(u_negtive, w)
    u1 = np.dot(u_postive, w)

    for x in range(len(test_labels)):
        yi = np.dot(test_datas[x], w)
        if abs(yi-u0) < abs(yi-u1):
            predict.append(0)
        else:
            predict.append(1)

    return predict

if __name__ == '__main__':
    # 输入数据 并调整数据格式
    lables = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

    data = np.array(data)
    # print(data)

    # 绘制图形散点图
    plt.scatter(data[:8, 0], data[:8, 1], c='green')
    plt.scatter(data[8:, 0], data[8:, 1], c='red')
    plt.xlabel('dimention')
    plt.ylabel('sugger')
    plt.show()

    # 将数据集百分之30作为测试集
    train_datas, test_datas, train_labels, test_labels = train_test_split(data, lables, test_size=0.30,
                                                                          random_state=23345)

    w, u_postive,  u_negtive = Train(data, lables)
    print(w)
    test_predict = Predict(test_datas, w, u_postive,  u_negtive)

    score = accuracy_score(test_labels, test_predict)
    print(score)

    p0x = -0.2
    p0y = p0x * (w[1, 0] / w[0, 0])
    p1x = 1
    p1y = p1x * (w[1, 0] / w[0, 0])

    #规定作图范围
    plt.xlim(-0.2, 1)
    plt.ylim(-0.5, 0.7)

    plt.scatter(data[:8, 0], data[:8, 1], c='green')
    plt.scatter(data[8:, 0], data[8:, 1], c='red')
    plt.xlabel('dimention')
    plt.ylabel('sugger')
    plt.plot([p0x, p1x], [p0y, p1y])

    # 绘制点的投影线
    m, n = np.shape(data)
    for i in range(m):
        x_p = GetProjectivePoint_2D([data[i, 0], data[i, 1]], [w[1, 0] / w[0, 0], 0])
        if lables[i] == 0:
            plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
        if lables[i] == 1:
            plt.plot(x_p[0], x_p[1], 'go', markersize=5)
        plt.plot([x_p[0], data[i, 0]], [x_p[1], data[i, 1]], 'c--', linewidth=0.3)

    plt.show()

###############################################################
    #观察图形后发现有一个点影响分类较严重,去掉后重新画图  而且数据过少所以全部用于训练
    lables.pop(14)
    data = np.delete(data, [14], axis=0)
    # print(data)

    plt.scatter(data[:8, 0], data[:8, 1], c='green')
    plt.scatter(data[8:, 0], data[8:, 1], c='red')
    plt.xlabel('dimention')
    plt.ylabel('sugger')
    plt.show()


    w, u_postive, u_negtive = Train(data, lables)
    test_predict = Predict(test_datas, w, u_postive, u_negtive)
    print(w)

    score = accuracy_score(test_labels, test_predict)
    print(score)

    p0x = -0.2
    p0y = p0x * (w[1, 0] / w[0, 0])
    p1x = 1
    p1y = p1x * (w[1, 0] / w[0, 0])
    # 规定作图范围
    plt.xlim(-0.2, 1)
    plt.ylim(-0.5, 0.7)

    plt.scatter(data[:8, 0], data[:8, 1], c='green')
    plt.scatter(data[8:, 0], data[8:, 1], c='red')
    plt.xlabel('dimention')
    plt.ylabel('sugger')
    plt.plot([p0x, p1x], [p0y, p1y])

    #绘制点的投影线
    m,n = np.shape(data)
    for i in range(m):
        x_p = GetProjectivePoint_2D([data[i, 0], data[i,1]], [w[1, 0]/w[0, 0], 0])
        if lables[i] == 0:
            plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
        if lables[i] == 1:
            plt.plot(x_p[0], x_p[1], 'go', markersize=5)
        plt.plot([x_p[0], data[i, 0]], [x_p[1], data[i, 1]], 'c--', linewidth=0.3)

    plt.show()
