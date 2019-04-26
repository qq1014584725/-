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
        mid = i[1: 9]
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

    print(datasets)
    print(labels)

