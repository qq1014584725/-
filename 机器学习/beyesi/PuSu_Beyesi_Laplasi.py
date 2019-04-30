import numpy as np
import pandas as pd

#读取西瓜数据
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
        mid = i[1: 7]
        dataSet.append(mid)

    # 读取标记
    Labels = []
    for i in data:
        if i[9] == '是':
            Labels.append(1)
        else:
            Labels.append(0)


    return dataSet, Labels

def train(datasets, labels):
    classfication = set(labels)

    # 计算P(c)概率 及每一个类的概率
    P_c = []
    for i in classfication:
        count = 0
        for j in labels:
            if i == j:
                count += 1

        # P_c.append(count/len(labels))             # 不带拉普拉斯修正
        P_c.append((count + 1) / (len(labels) + len(classfication)))              #不带拉普拉斯修正

    #计算P(x\c)概率 及后验概率 并制作出后验概率表

    c = []
    for i in classfication:
        l = [[i]]
        for x in range(len(labels)):
            if labels[x] == i:
                l.append(x)
        c.append(l)                         #将数据按类分隔开

    #计算P(x\c)
    data = np.array(datasets)
    probobility_list = []
    for i in c:
        same_class = []
        for j in range(len(datasets[0])):
            same_character = []
            for k in set(data[:,j]):
                xi = [k for x in i[1:] if data[x, j] == k]
                same_character.append([k, (len(xi)+1)/(len(i)-1+len(set(data[:,j])))])
            same_class.append(same_character)
        probobility_list.append([i[0],same_class])
    print(probobility_list)


if __name__ == '__main__':
    datasets, labels = watermalonDataSet()

    # print(datasets)
    # print(labels)

    train(datasets, labels)
