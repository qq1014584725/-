import cv2
import pandas as pd
import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('hog.xml')

    for img in trainset:
        img = np.reshape(img, (28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features

def Train(trainset, train_labels):
    #获取训练长度
    trainset_size = len(train_labels)
    features_length = len(trainset[0])

    #初始化w和b
    w = np.zeros((features_length, 1))
    b = 0

    study_count = 0         #学习次数记录，分类错误时几一次
    nochange_count = 0      #记录连续分类正确的次数，若分类错误则归为0
    nochange_limit = 100000      #连续分类正确多少次后跳出循环
    study_step = 0.001           #训练步长
    study_total = 10000          #最多迭代多少次退出循环

    while True:
        nochange_count += 1
        if nochange_count > nochange_limit:
            break

        #随机选取数据
        index = random.randint(0, trainset_size-1)
        img = trainset[index]
        label = train_labels[index]

        #计算yi(wx+b)
        yi = int(label) * 2 - 1
        result = yi * (np.dot(img, w) + b)

        #如果异号则更新w和b
        if result <= 0:
            img = np.reshape(trainset[index], (features_length, 1))           #调整其为一列的向量

            w += img * yi * study_step
            b += yi * study_step

            study_count += 1

            if study_count > study_total:
                break

            nochange_count = 0

    return (w,b)

def Predict(testset, w, b):
    predict = []
    for img in testset:
        result = np.dot(img,w) + b
        result = result > 0

        predict.append(result)

    return np.array(predict)

if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[:, 1:]
    labels = data[:, 0]

    features = get_hog_features(imgs)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print('read data cost ',time_2 - time_1,' second','\n')

    print('Start training')
    w,b = Train(train_features,train_labels)
    time_3 = time.time()
    print('training cost',time_3 - time_2,' second','\n')

    print('Start predict')
    test_predict = Predict(test_features, w, b)
    time_4 = time.time()
    print('predicting cost',time_4 - time_3, 'second','\n')

    score = accuracy_score(test_labels, test_predict)
    print('The accuracy score is', score)


