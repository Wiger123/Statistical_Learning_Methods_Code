# coding = utf-8
# Author: Wiger
# Date: 2020-11-10
# Email: wiger@mail.ustc.edu.cn

'''
例:
C1 C2 ... Ck: 决定是否贷款
    C1: 贷款
    C2: 不贷款
D1 D2 ... Dn: 某一特征上的分类
    A1: 年龄
        D1: 青年
        D2: 中年
        D3: 老年
    A2: 工作
        D1: 有工作
        D2: 无工作
    A3: 房屋
        D1: 有房
        D2: 无房
'''

import numpy as np

def majorClass(labelArr):
    '''
    寻找当前标签集合中数目最多的标签
    :param labelArr: 标签集合
    :return: 数目最多的标签
    '''

    # 建立字典, 统计各类标签
    classDict = {}

    # 遍历标签
    for i in range(len(labelArr)):
        # 字典中存在该标签
        if labelArr[i] in classDict.keys():
            # 标签数目加一
            classDict[labelArr[i]] += 1
        # 无该标签
        else:
            # 创建该标签
            classDict[labelArr[i]] = 1

    # 字典进行降序排序
    classSort = sorted(classDict.items(), key = lambda x : x[1], reverse = True)
    
    # 返回第一项(数目最多)的标签
    return classSort[0][0]

def calc_H_D(trainLabelArr):
    '''
    计算数据集 D 的经验熵, 熵只与分布概率分布有关, 与数据本身的值无关
    :param trainLabelArr: 训练标签集合
    :return: 经验熵
    '''

    # 经验熵初始化
    H_D = 0

    # 标签放入集合中
    trainLabelSet = set([label for label in trainLabelArr])

    # 计算每一个出现过的标签
    for i in trainLabelSet:
        # 计算概率
        # trainLabelArr == i: 当前标签集中为该标签的的位置
        # 例: a = [1, 0, 0, 1], c = (a == 1): c == [True, false, false, True]
        # trainLabelArr[trainLabelArr == i]: 获得指定标签的样本
        # trainLabelArr[trainLabelArr == i].size: 获得指定标签的样本的大小, 即 |Ck|
        # trainLabelArr.size: 整个标签集的数量(样本集的数量), 即 |D|
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size

        # 经验熵累加求和
        H_D += -1 * p * np.log2(p)

    # 返回经验熵
    return H_D

def calcH_D_A(trainDataArr_DevFeature, trainLabelArr):
    '''
    计算经验条件熵
    :param trainDataArr_DevFeature: 切割后只有 feature 那列数据的数组
    :param trainLabelArr: 标签集数组
    :return: 经验条件熵
    '''

    # 初始为 0
    H_D_A = 0

    # 在 featue 那列放入集合中, 是为了根据集合中的数目知道该 feature 目前可取值数目是多少
    trainDataSet = set([label for label in trainDataArr_DevFeature])

    # 对于每一个特征取值遍历计算条件经验熵的每一项
    for i in trainDataSet:
        # 计算 H(D|A)
        # trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size: |Di| / |D|
        # calc_H_D(trainLabelArr[trainDataArr_DevFeature == i]): H(Di)
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size * calc_H_D(trainLabelArr[trainDataArr_DevFeature == i])

    #返回得出的条件经验熵
    return H_D_A

def calcBestFeature(trainDataList, trainLabelList):
    '''
    计算信息增益最大的特征
    :param trainDataList: 当前数据集
    :param trainLabelList: 当前标签集
    :return: 信息增益最大的特征和最大信息增益值
    '''

    # 数据集转为数组
    trainDataArr = np.array(trainDataList)

    # 标签集转为数组
    # 标签集需要转置
    # 借用 dodo 一个很好的例子:
    # a = np.array([1, 2, 3]); b = np.array([1, 2, 3]).T
    # a[0] = [1, 2, 3]; b[0] = 1, b[1] = 2
    # 为了调用每一个元素, 标签集需要转置
    trainLabelArr = np.array(trainLabelList).T

    # 特征数目
    # shape[0]: 样本数目; shape[1]: 特征数目
    featureNum = trainDataArr.shape[1]

    # 初始化最大信息增益
    maxG_D_A = -1

    # 初始化最大信息增益的特征
    maxFeature = -1

    # 对所有特征遍历计算
    for feature in range(featureNum):

        # 1. 计算数据集 D 的经验熵 H(D)
        H_D = calc_H_D(trainLabelArr)

        # 2. 提取列表中 feature 列的数据
        # trainDataArr[:, feature]: 所有样本第 feature 个特征那的一列数据
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)

        # 3. 计算条件经验熵 H(D|A)
        H_D_A = calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)

        # 4. 计算信息增益率: G(D,A) = H(D) - H(D|A)
        G_D_A = H_D - H_D_A

        # 5. 更新最大的信息增益与对应的特征
        if G_D_A > maxG_D_A:
            # 更新最大信息增益
            maxG_D_A = G_D_A

            # 更新对应特征
            maxFeature = feature

    # 返回最大信息增益的特征和最大信息增益
    return maxFeature, maxG_D_A