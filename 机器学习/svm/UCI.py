from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#读取数据集
def read_data():
    data_set = load_breast_cancer()
    X = data_set.data
    feature_names = data_set.feature_names
    y = data_set.target
    target_names = data_set.target_names

    return X, y, feature_names, target_names

if __name__ == '__main__':
    X, y, feature_names, target_names = read_data()
    print(X,y)
    print(feature_names)
    print(target_names)

    #选取前两个特征绘制散点图
    f1 = plt.figure(1)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='r', label=target_names[0])
    plt.scatter(X[y==1, 0], X[y==1, 1], color='g', label=target_names[1])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(loc='upper right')
    plt.grid(True, linewidth=0.3)
    plt.show()

    #数据归一化  就是L2正则化（每个实例的每个值除以它的2范数）
    normalized_X = preprocessing.normalize(X)

    #分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.5, random_state=0)
    for fig_num, kernel in enumerate(('linear', 'rbf')):
        accuracy = []
        c = []
        #由于不知道合适的惩罚系数 所以从1到1000中遍历选择最优的
        for C in range(1, 1000, 1):
            #初始化svm的训练器
            clf = svm.SVC(C=C, kernel=kernel, gamma='auto')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy.append(metrics.accuracy_score(y_test, y_pred))
            c.append(C)

        print('max accuracy of %s kernel SVM: %.3f' % (kernel, max(accuracy)))

        #画出图形关于C和准确率的图形
        f2 = plt.figure(kernel)
        plt.plot(c, accuracy)
        plt.xlabel('penalty paramter')
        plt.ylabel('accuracy')

        plt.show()