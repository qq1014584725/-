from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def watermalonDataSet():
    # 读取西瓜数据
    sheet = pd.read_excel(io='data.xlsx')

    # 数据处理
    data = []
    for v in range(17):
        data.append(list(sheet.ix[v].values))

    # 读取数据值
    dataSet = []
    for i in data:
        mid = i[7:9]
        dataSet.append(mid)

    # 读取标记
    Labels = []
    for i in data:
        if i[9] == '是':
            Labels.append(1)
        else:
            Labels.append(0)


    return dataSet, Labels

if __name__ == '__main__':
    datasets, labels = watermalonDataSet()
    # print(datasets, labels)
    datasets = np.array(datasets)
    labels = np.array(labels)

    #采取两种不同的核函数进行分类
    #linear为线性核函数及x.T * x   rbf为高斯核函数
    for fig_num, kernel in enumerate(('linear', 'rbf')):
        #初始化svm分类器
        svc = svm.SVC(C=1000, kernel=kernel, gamma='auto')
        #训练数据集
        svc.fit(datasets, labels)
        #获得支持向量
        sv = svc.support_vectors_

        #画图
        plt.figure(fig_num)
        plt.clf()

        #绘制散点图 以及支持高亮支持向量的点
        plt.scatter(datasets[:, 0], datasets[:, 1], edgecolors='k', c=labels, cmap=plt.cm.Paired, zorder=10)
        plt.scatter(sv[:, 0], sv[:, 1], edgecolors='k', facecolors='none', s=80, linewidths=2, zorder=10)

        #绘制分割平面以及区域上色
        #找到x轴和y轴的极限位置
        x_min, x_max = datasets[:, 0].min() - 0.2, datasets[:, 0].max() + 0.2
        y_min, y_max = datasets[:, 1].min() - 0.2, datasets[:, 1].max() + 0.2

        #根据最大值和最小找以0.02为间隔生成网格矩阵
        XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        #用np.c_[]组合每个网格点的x值和y值
        #在用svc.decision_function计算每个点到分割平面的距离
        Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])

        #绘图前先将Z的形式改得跟XX一致
        Z = Z.reshape(XX.shape)
        print(np.shape(XX))
        print(Z)
        print(np.shape(Z))

        #pcolormesh绘制上色 plt.绘制等高线
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])

        plt.title(kernel)
        plt.axis('tight')

    plt.show()
