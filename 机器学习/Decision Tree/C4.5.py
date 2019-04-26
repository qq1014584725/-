import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def createDataSet():
    """
       创建数据集
       """
    dataSet = [['青年', '否', '否', '一般', '拒绝'],
               ['青年', '否', '否', '好', '拒绝'],
               ['青年', '是', '否', '好', '同意'],
               ['青年', '是', '是', '一般', '同意'],
               ['青年', '否', '否', '一般', '拒绝'],
               ['中年', '否', '否', '一般', '拒绝'],
               ['中年', '否', '否', '好', '拒绝'],
               ['中年', '是', '是', '好', '同意'],
               ['中年', '否', '是', '非常好', '同意'],
               ['中年', '否', '是', '非常好', '同意'],
               ['老年', '否', '是', '非常好', '同意'],
               ['老年', '否', '是', '好', '同意'],
               ['老年', '是', '否', '好', '同意'],
               ['老年', '是', '否', '非常好', '同意'],
               ['老年', '否', '否', '一般', '拒绝'],
               ]
    featureName = ['年龄', '有工作', '有房子', '信贷情况']

    return dataSet, featureName

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
        mid = i[1:]
        dataSet.append(mid)

    # 读取特征
    Labels = []
    for i, k in sheet.items():
        Labels.append(k.name)

    Labels = Labels[1:9]

    return dataSet, Labels

#数据分割 将某一维特征的某一特征值下的实例全部取出 并去除改维数据以便下一步递归
def splitDataSet(dataSet, axis, value):
    retData = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retData.append(reduceFeatVec)

    return retData

#连续变量的分割函数
def splitDataSetByValue(dataSet, i, split):
    retData1 = []
    retData2 = []
    for featVec in dataSet:
        #小于分割点的
        if featVec[i] < float(split):
            reduceFeatVec1 = featVec[:i]
            reduceFeatVec1.extend(featVec[i + 1:])
            retData1.append(reduceFeatVec1)
        #大于分割点的
        else:
            reduceFeatVec2 = featVec[:i]
            reduceFeatVec2.extend(featVec[i + 1:])
            retData2.append(reduceFeatVec2)

    return [retData1,retData2]

#计算信息熵
def calcShannonEnt(dataSet):
    #计算总体实例个数
    numEntries = len(dataSet)

    #此处用字典存储每个标签出现的次数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        #若果遍历到新标签加入到标签字典中
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        #遍历出的标签在其相应的value下加1
        labelCounts[currentLabel] += 1

    #计算实例不同标签下的概率
    shannonEnt = 0.0
    for key in labelCounts:
        P = float(labelCounts[key])/numEntries
        shannonEnt -= P * np.log2(P)

    return shannonEnt

#计算条件熵
def calcConditionalEntropy(dataSet, uniqueVals, i):
    ce = 0.0

    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        ce += len(subDataSet) * calcShannonEnt(subDataSet)

    ce = ce/len(dataSet)

    return ce

#连续值变量的最佳切分点计算 以及其最大的信息增益
def calcRelativeValueEntropy(dataSet, uniqueVals, i):
    ce = 0.0
    rangeList = []

    #遍历所有点寻找最佳且分点
    for index in range(len(uniqueVals)-1):
        split = (uniqueVals[index] + uniqueVals[index+1])/2
        subDataSet = splitDataSetByValue(dataSet, i, split)
        ce = (len(subDataSet[0]) * calcShannonEnt(subDataSet[0]) + len(subDataSet[1]) * calcShannonEnt(subDataSet[1]))/len(dataSet)
        rangeList.append((ce,split))

    rangeList = sorted(rangeList, key=lambda x:x[0])

    return rangeList[0]

#计算信息增益
def calcInformationGain(dataSet, baseEntropy, i):

    # 提取第i维特征的全部数据
    featList = [example[i] for example in dataSet]
    # 提取该特征的全部取值
    uniqueVals = set(featList)

    #若特征为离散变量
    if type(dataSet[0][i]).__name__ == 'str':
        #计算条件熵
        newEntropy = calcConditionalEntropy(dataSet, uniqueVals, i)
        t = None

    #若特征连续变量
    else:
        uniqueVals = sorted(uniqueVals)
        newEntropy, t = calcRelativeValueEntropy(dataSet, uniqueVals, i)

    infoGain = baseEntropy - newEntropy
    return infoGain, t

#计算固有值函数
def calcIntrinsicValue(dataSet, i):
    #计算每个属性值的次数
    intrinsicDict = {}
    for featVec in dataSet:
        if featVec[i] not in intrinsicDict:
            intrinsicDict[featVec[i]] = 0
        intrinsicDict[featVec[i]] += 1

    HA = 0.0
    for key in intrinsicDict:
        P = float(intrinsicDict[key])/len(dataSet)
        HA -= P * np.log2(P)

    return HA

#信息增益率的计算函数
def calcInformationGainRatio(dataSet, baseEntropy, i):
    #提取第i维特征的全部数据
    featList = [example[i] for example in dataSet]
    #提取所有的取值
    uniqueVals = set(featList)

    #若特征为离散变量
    if type(dataSet[0][i]).__name__ == 'str':
        #计算条件熵
        newEntropy = calcConditionalEntropy(dataSet, uniqueVals, i)
        infoGain = baseEntropy - newEntropy
        infoGainratio = infoGain / calcIntrinsicValue(dataSet, i)
        t = None

    # 若特征连续变量
    else:
        uniqueVals = sorted(uniqueVals)
        newEntropy, t = calcRelativeValueEntropy(dataSet, uniqueVals, i)
        infoGain = baseEntropy - newEntropy
        count = 0
        for feaVec in dataSet:
            if feaVec[i] < t:
                count += 1
        P = count/len(dataSet)
        IntrinsicValue = 0 - P * np.log2(P) - (1-P) * np.log2(1-P)
        infoGainratio = infoGain/IntrinsicValue

    return infoGainratio, t

#ID3算法框架
def chooseBestFeatureByID3(dataSet):
    #读取最后一列的分类
    numFeatures = len(dataSet[0]) - 1

    #计算信息熵H（X）
    baseEntropy = calcShannonEnt(dataSet)

    #计算所剩特征的最大信息增益
    bestInfoGain = 0.0
    bestFeature = -1
    split = None

    #遍历所有维度 并计算其信息增益
    for i in range(numFeatures):
        infoGain, t = calcInformationGain(dataSet, baseEntropy, i)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
            split = t

    return bestFeature, split

#C45算法框架
def chooseBestFeatureByC45(dataSet):
    #读取最后一列的分类
    numFeatures = len(dataSet[0]) - 1

    #计算信息熵H（X）
    baseEntropy = calcShannonEnt(dataSet)

    #计算所剩特征的最大行西增益率
    bestInfoGainRatio = 0.0
    bestFeature = -1
    split = None

    #遍历所有维度 并计算其信息增益率
    for i in range(numFeatures):
        infoGainradio, t =calcInformationGainRatio(dataSet, baseEntropy, i)
        if infoGainradio > bestFeature:
            bestInfoGainRatio = infoGainradio
            bestFeature = i
            split = t

    return bestFeature, split

#找出类实例最多的一类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1

    #对字典中item对象里面的每个value值进行降序排列
    sortedClassCount = sorted(classCount, key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def createTree(dataSet, featureName, chooseBestFeatureFunc = chooseBestFeatureByC45):

    #读取所有数据里面的分类情况
    classList = [example[-1] for example in dataSet]

    #设置返回条件
    if classList.count(classList[0]) == len(classList):                #如果分类全为一类标签
        return classList[0]
    if len(dataSet[0]) == 0:                                            #若果分类特征已经没有了  返回类最多的一项
        return majorityCnt(classList)

    # 找出最佳分类特征的索引值 及其特征
    bestFeat, t = chooseBestFeatureFunc(dataSet)
    bestFeatLabel = featureName[bestFeat]

    #map结构，且key为featureLabel
    myTree = {bestFeatLabel:{}}
    del (featureName[bestFeat])

    #找出最佳分类特征的特征值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    #如果最佳分类特征为离散变量
    if type(dataSet[0][bestFeat]).__name__=='str':
        #对不同的value进行递归
        for value in uniqueVals:
            subLabels = featureName[:]          #复制一个新的label作为传递参数传递给下一层决策树
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    else:
        subLabels = featureName[:]
        myTree[bestFeatLabel][t,'<'] = createTree(splitDataSetByValue(dataSet, bestFeat,t)[0], subLabels)
        subLabels = featureName[:]
        myTree[bestFeatLabel][t,'>='] = createTree(splitDataSetByValue(dataSet, bestFeat, t)[1], subLabels)


    return myTree

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#获取决策树叶子节点的个数
def getNumLeafs(myTree):
    numLeafs = 0                   #初始化叶子
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    # 可以使用list(myTree.keys())[0]
    #将树变为迭代对象然后用next函数读取他的key值
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]                      #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':     #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs

#获取决策树的层数
def getTreeDepth(myTree):
    maxDepth = 0                                  #初始化决策树深度
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    # 可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]                          #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':         #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth      #更新层数
    return maxDepth


"""
函数说明:绘制结点
Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
"""


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=font)


"""
函数说明:标注有向边属性值
Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
"""


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置中文字体
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30, fontproperties=font)


"""
函数说明:绘制决策树
Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
"""


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


"""
函数说明:创建绘制面板
Parameters:
    inTree - 决策树(字典)
"""


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()

#////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    # # 读取数据
    # dataSet,featureName = createDataSet()
    #
    # #生成决策树
    # myTree = createTree(dataSet, featureName)
    # print(myTree)

    #西瓜数据测试
    dataSet, featureName = watermalonDataSet()

    #生成决策树
    myTree = createTree(dataSet, featureName)
    print(myTree)


    #可视化决策树
    # 定义文本框和箭头格式
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='0.8')
    arrow_args = dict(arrowstyle='<-')
    # 设置中文字体
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    createPlot(myTree)





