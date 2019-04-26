import pandas as pd
import numpy as np
import copy

def Train(data, chara_value, character, classfication=None, count=0):
    # 返回条件
    if len(data) < 1:  # 若果分支的实例为空返回
        return print('是')
    if len(character) == 0:  # 若果特征已经分类完返回
        return print('特征为空')

    # 如果分支实例全为一类返回
    comper = data[0][-1]
    for j in range(len(data)):
        if data[j][-1] != comper:
            break
        elif j+1 == len(data):
            return print(comper)
        else:
            pass

    ##################不返回继续计算
    #计算信息熵
    postive = 0
    all = 0
    for i in data:
        all += 1
        if i[-1] == '是':
           postive += 1

    P = postive/all
    H = -(P * np.log2(P) + (1-P) * np.log2(1-P))
    # print(H)

    #计算不同特征下的条件熵
    rem_all = []
    for i in range(len(chara_value)):
        if chara_value[i] != 0:
            rem1 = []
            H_ = 0
            for j in range(len(chara_value[i])):
                rem2 = []
                P = 0
                for k in data:
                    if k[i+1] == j+1:
                        rem2.append(k)
                        if k[-1] == '是':
                             P += 1

                if len(rem2) == 0:
                    P_ = 0
                else:
                    P_ = P/len(rem2)

                if P_ == 0 or P_ == 1:
                    H_ += 0
                else:
                    H_ +=  len(rem2) * (P_ * np.log2(P_)+ (1-P_) * np.log2(1-P_))

                rem1.append(rem2)
            rem1.append(H + H_/len(data))

            rem_all.append(rem1)

    #找出信息增益最大的一个属性对其进行递归
    maxer = 0
    index = 0
    for i in range(len(rem_all)):
        if rem_all[i][-1] > maxer:
            maxer = rem_all[i][-1]
            index = i

    #将新分支放入递归过程
    character_ = copy.copy(character)
    classfication = character_.pop(index)
    chara_value_ = copy.copy(chara_value)
    chara_value_.pop(index)
    chara_value_.insert(index, 0)
    count += 1

    for i in rem_all[index][:-1]:
        Character.append(character_)
        Chara_Value.append(chara_value_)
        Count.append(count)

    # print(count)
    print(classfication)
    for i in rem_all[index][:-1]:
        Train(i,Chara_Value.pop(), Character.pop(), classfication, Count.pop())



if __name__ == '__main__':
    #读取西瓜数据
    sheet = pd.read_excel(io='data.xlsx')
    # print(sheet)

    #数据处理
    data = []
    for v in range(17):
        data.append(sheet.ix[v].values)

    # print(data)

    #读取特征
    character = []
    for i,k in sheet.items():
        character.append(k.name)

    #去除第一列和最后三列
    character = character[1:-3]
    chara_value = []
    for i in character:
        chara_value.append(list(set(sheet.ix[:,i].values)))
    # print(chara_value)

    #将文字特征转化为数字
    for i in data:
        for j in range(len(chara_value)):
            for k in range(len(chara_value[j])):
                if i[j+1] == chara_value[j][k]:
                    i[j+1] = k+1
    # print(data)

    #生成决策树
    Character = []
    Chara_Value = []
    Count = []
    Train(data, chara_value, character)