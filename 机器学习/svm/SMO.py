import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer

#核函数计算
def kernelTrans(X, A, model):  # 通过数据计算转换后的核函数
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if model[0] == 'lin':  # 线性核函数
        K = X * A.T

    elif model[0] == 'rbf':  # 高斯核
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * model[1] ** 2))

    # # 错误核函数
    else:
        raise ValueError('{} is not a kernal function'.format(model[0]))

    return K

#计算SVM运算是的数据结构
# X 输入数据 每一行代表一个实例（矩阵Matrix）
# labelMat 输入数据的标签（列向量Vector）
# C 惩罚系数（一般需要调试最优的值）
# toler 软间隔分割的克赛 另一种说法就是函数间隔
# m 输入数据的实例个数
# alphas 优化参数α（列向量Vector）
# b 偏置参数
# eCache 为缓存ei的值避免重复计算
class svmStruct:
    def __init__(self, dataMatIn, classLables, C, toler, model):
        self.X = dataMatIn
        self.labelMat = classLables
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.model = model
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], self.model)


#SVM类
class SVM:
    # 初始化参数
    def __init__(self):
        self.alphas = None
        self.b = None
        self.w = None
        self.model = None

    #用静态方法将其封装为类内函数  作用是可以不用实例化的调用此函数
    @staticmethod
    # 计算Ei值及（f（x）-y）的大小
    def calcEi(opt, i):
        fxi = float(np.multiply(opt.alphas, opt.labelMat).T * (opt.K[:, i])) + opt.b
        Ei = fxi - float(opt.labelMat[i])

        return Ei

    @staticmethod
    # 随机产生第二个α的索引值
    def selecJrand(i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    @staticmethod
    # 根据\Ei-Ej\的大小选择第二个α的索引
    def selectJ(i, opt, ei):
        #存储最大\Ei-Ej\的索引值和Ej值
        max_k = -1
        max_delta_e = 0

        ej = 0
        #更新ei进缓存
        opt.eCache[i] = [1, ei]
        #找出计算过ei的索引
        valid_e_cache_list = np.nonzero(opt.eCache[:,0].A)[0]
        #若未计算过ei
        if len(valid_e_cache_list) > 1:
            for k in valid_e_cache_list:
                if k == i:
                    continue
                ek = SVM.calcEi(opt, k)
                delta_e = abs(ei - ek)
                if delta_e > max_delta_e:
                    max_k = k
                    max_delta_e = delta_e
                    ej = ek

            return max_k, ej
        #若未计算仍和ei
        else:
            j = SVM.selecJrand(i, opt.m)
            ej = SVM.calcEi(opt, j)
            return j, ej

    @staticmethod
    #更新ei缓存值
    def updateEi(opt, i):
        ei = SVM.calcEi(opt, i)
        opt.eCache[i] = [1, ei]

    @staticmethod
    #根据计算的最优αj的边界值取αj的最优点
    def clip_alpha(alpha_j, high, low):
        if alpha_j > high:
            alpha_j = high
        elif alpha_j < low:
            alpha_j = low
        return alpha_j

    @staticmethod
    #计算αj的上下界
    def calcLimt(opt, i, j):
        if opt.labelMat[i] == opt.labelMat[j]:
            high = min(opt.alphas[i] + opt.alphas[j], opt.C)
            low = max(0, opt.alphas[i] + opt.alphas[j] - opt.C)
        else:
            high = min(opt.C + opt.alphas[j] - opt.alphas[i], opt.C)
            low = max(0, opt.alphas[j] - opt.alphas[i])

        return high, low

    #在已经选定出第一个迭代参数后，找第二个迭代参数并计算更形后的参数
    def inner_l(self, i, opt):
        #计算ei
        ei = SVM.calcEi(opt, i)
        #查看是否满足KKT条件
        if ((opt.labelMat[i] * ei < -opt.tol) and (opt.alphas[i] < opt.C)) or ((opt.labelMat[i] * ei > opt.tol) and (opt.alphas[i] > 0)):
            #选取第二个调优参数 以及复制α
            j, ej = SVM.selectJ(i, opt, ei)
            alpha_iold = opt.alphas[i].copy()
            alpha_jold = opt.alphas[j].copy()
            #计算αj的上下界
            high, low = SVM.calcLimt(opt, i, j)

            #上下界相等时不进行更新
            if low == high:
                print('low == high')
                return 0

            #计算eta = Kii + Kjj - 2Kij 此处为线性核函数 Kij = （Xi）.T * Xj
            eta = 2.0 * opt.K[i,j] - opt.K[i,i] - opt.K[j,j]

            #若eta >= 0 说明开口向下最小值出现在上下界
            if eta >= 0:
                print('eta >= 0')
                return 0

            #计算更新后的αj
            opt.alphas[j] -= opt.labelMat[j] * (ei - ej) / eta
            #判断其是否在可行域范围内并找出最小点
            opt.alphas[j] = SVM.clip_alpha(opt.alphas[j], high, low)
            SVM.updateEi(opt, j)

            #如果αj的变化非常小那么停止更新αi 因为αi靠差做更新
            if abs(opt.alphas[j] - alpha_jold) < 0.00001:
                print('j not moving enough')
                return 0

            #更新αi
            opt.alphas[i] += opt.labelMat[i] * opt.labelMat[j] * (alpha_jold - opt.alphas[j])
            SVM.updateEi(opt, i)

            #更新b
            b1 = opt.b - ei - opt.labelMat[i] * (opt.alphas[i] - alpha_iold) * opt.K[i,i] - opt.labelMat[j] * (opt.alphas[j] - alpha_jold) * opt.K[i,j]
            b2 = opt.b - ej - opt.labelMat[i] * (opt.alphas[i] - alpha_iold) * opt.K[i,j] - opt.labelMat[j] * (opt.alphas[j] - alpha_jold) * opt.K[j,j]
            # 步骤 8：求解 b
            if (opt.alphas[i] > 0) and (opt.alphas[i] < opt.C):
                opt.b = b1
            elif (opt.alphas[j] > 0) and (opt.alphas[j] < opt.C):
                opt.b = b2
            else:
                opt.b = (b1 + b2) / 2.0
            return 1

        else:
            return 0

    #smo算法
    def smo(self, input_data, lable_data, C, toler, max_iter, model=('linear', 0)):
        #初始化w, b
        self.w = None
        self.b = None

        #初始化需要用到的数据结构
        opt = svmStruct(np.mat(input_data), np.mat(lable_data).transpose(), C, toler, model)
        self.model = opt.model

        #记录迭代的参数
        iter = 0
        #记录轮次的参数
        entire_set = True
        #记录alpha对的改变次数
        alpha_pairs_changed = 0

        #循环迭代求解参数
        # 第一轮求解走if下条件，并且计算完后可以找出所有属于[0，C]的α，下一次迭代进入else下条件
        # 直到在else条件下α无变化及 alpha_pairs_changed = 0 后 在进入if条件
        while (iter < max_iter) and ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(opt.m):
                    alpha_pairs_changed += self.inner_l(i, opt)
                    print('fullSet, iter: {} i: {},paris changed {}'.format(iter, i, alpha_pairs_changed))
                iter += 1
            else:
                #找出所有属于[0， C]α的索引值
                non_bound_i = np.nonzero((opt.alphas.A > 0) * (opt.alphas.A < opt.C))[0]

                for i in non_bound_i:
                    alpha_pairs_changed += self.inner_l(i, opt)
                    print('non-bound, iter: {} i: {},pairs changed {}'.format(iter, i, alpha_pairs_changed))
                iter += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print('iteration number: {}'.format(iter))

        self.alphas = opt.alphas
        self.b = opt.b
        return opt.b, opt.alphas

    def calc_w(self, input_data, label_data):
        X = np.mat(input_data)
        label_mat = np.mat(label_data).transpose()
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(self.alphas[i] * label_mat[i], X[i, :].T)
        self.w = w
        return w

    def predict(self, input_feature, X=None, label=None):
        #线性核函数预测 需要先计算w
        if self.model[0] == 'linear':
            pred = np.mat(input_feature) * self.w + self.b
            for i in range(len(pred)):
                if pred[i] > 0:
                    pred[i] = 1
                else:
                    pred[i] = -1
            return pred

        #高斯核函数预测
        elif self.model[0] == 'rbf':
            input_feature = np.mat(input_feature)
            m,n = np.shape(input_feature)
            label = np.mat(label).transpose()
            nonzero_list = np.nonzero(self.alphas.A > 0)[0]
            fx = np.zeros((m,1))

            for i in range(m):
                kernelEval = kernelTrans(X[nonzero_list], input_feature[i, :], self.model)
                predict = kernelEval.T * np.multiply(label[nonzero_list], self.alphas[nonzero_list]) + self.b
                fx[i,0] = predict

            return fx
        else:
            raise ValueError('{} is not a kernal function'.format(self.model))

#读取数据集
def load_data_set(file_name):
    """
    从文件中获取数据集
    :param file_name: 文件名
    :return: 返回从文件中获取的数据集
             input_data 存储特征值，label_data 存储标签值
    """
    input_data, label_data = [], []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        input_data.append([float(cur_line[0]), float(cur_line[1])])
        label_data.append(float(cur_line[2]))
    return input_data, label_data

#读取UCI数据集
def read_data():
    data_set = load_breast_cancer()
    X = data_set.data
    feature_names = data_set.feature_names
    y = data_set.target
    target_names = data_set.target_names

    return X, y, feature_names, target_names


if __name__ == '__main__':
    # 读取数据
    X, y = load_data_set('textSet.txt')

    # 读取UCI数据
    # X, y, feature_names, target_names = read_data()
    # X = X[20:70, 0:2]
    # y = y[20:70]
    # for i in range(len(y)):
    #     if y[i] == 0:
    #         y[i] = -1


    # # 选取前两个特征绘制散点图
    # f1 = plt.figure(1)
    # data01_x1, data01_x2, data02_x1, data02_x2 = [], [], [], []
    # for i in range(len(X)):
    #     if y[i] > 0:
    #         data01_x1.append(X[i][0])
    #         data01_x2.append(X[i][1])
    #     else:
    #         data02_x1.append(X[i][0])
    #         data02_x2.append(X[i][1])
    # plt.scatter(data01_x1, data01_x2, color='r')
    # plt.scatter(data02_x1, data02_x2, color='g')
    # plt.legend(loc='upper right')
    # plt.grid(True, linewidth=0.3)
    # plt.show()
    #
    # # 用smo算法的SVM训练
    # svm = SVM()
    # svm.smo(X, y, 0.6, 0.001, 40)
    # svm.calc_w(X, y)
    #
    # #打印出支持向量
    # for i in range(len(svm.alphas)):
    #     if svm.alphas[i] > 0.0:
    #         print('suppor vector:{} ,{}; alpha = {}'.format(X[i], y[i], svm.alphas[i]))
    #
    # #打印出权重和偏置的参数
    # print(svm.w)
    # print(svm.b)
    #
    # #画出超平面
    # #画出散点图
    # plt.scatter(data01_x1, data01_x2, s=30)
    # plt.scatter(data02_x1, data02_x2, s=30)
    #
    # #超平面计算
    # x1_max = max(X)[0]
    # x1_min = min(X)[0]
    # x = np.arange(x1_min, x1_max, 0.1)
    # y = (-float(svm.b) - float(svm.w[0])*x) / float(svm.w[1])
    # plt.plot(x,y)
    #
    # #标记支持向量
    # for i, alpha in enumerate(svm.alphas):
    #     if alpha > 0:
    #         x1, x2 = X[i]
    #         plt.scatter([x1], [x2], s=100, c='none', linewidths=1.5, edgecolors='red')
    #
    # plt.show()
    #
    # #对数据进行预测
    # pred = svm.predict(X)

    #用核函数进行分类
    svm = SVM()
    svm.smo(X, y, 1, 0.001, 40, ('rbf', 1))

    # 选取前两个特征绘制散点图
    f1 = plt.figure(1)
    data01_x1, data01_x2, data02_x1, data02_x2 = [], [], [], []
    for i in range(len(X)):
        if y[i] > 0:
            data01_x1.append(X[i][0])
            data01_x2.append(X[i][1])
        else:
            data02_x1.append(X[i][0])
            data02_x2.append(X[i][1])
    plt.scatter(data01_x1, data01_x2, color='r')
    plt.scatter(data02_x1, data02_x2, color='g')
    plt.legend(loc='upper right')
    plt.grid(True, linewidth=0.3)

    #可视化高斯核下的图像
    X = np.mat(X)
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    #生成网格
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    #用预测函数算出预测值
    Z = svm.predict(np.c_[XX.reshape(-1,1), YY.reshape(-1,1)], X, y)
    Z = Z.reshape(np.shape(XX))
    # print(Z)

    plt.contour(XX, YY, Z, 0)

    plt.show()