# coding = utf-8
# Author: Wiger
# Date: 2020-10-30
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000
测试集数量: 10000
---------------
运行结果:
非 kd 树
测试集: 200
k: 25
欧式距离:
    正确率: 97%
    运行时长: 547.80s
曼哈顿距离:
    正确率: 95.5%
    运行时长: 471.04s
'''

import pandas as pd
import numpy as np
import time
import knn.kd_tree as kd

def loadData(fileName):
    '''
    加载 Mnist 数据集
    :param fileName: 数据集路径
    :return: 数据 list 和 标签 list
    '''

    # 开始读取数据
    print('START TO READ DATA')

    # 存放数据集和标签的 list
    dataArr = [];
    labelArr = [];

    # pandas 读取, 更为简便

    # 打开文件, 按行读取
    df = pd.read_csv(fileName, header = None)

    # 获取数据 list
    dataArr = df.iloc[:, 1:]

    # 获取标签 list
    labelArr = df.iloc[:, 0]

    # 数据读取结束
    print('END READING DATA')

    return dataArr, labelArr

def calcDist(x1, x2):
    '''
    计算两样本向量之间距离
    :param x1: 向量一
    :param x2: 向量二
    :return: 向量之间距离
    '''

    # 欧式距离
    return np.sqrt(np.sum(np.square(x1 - x2)))

    # 曼哈顿距离
    # return np.sum(np.abs(x1 - x2))

# 纯 numpy 写法
def getNearest(trainDataMat, trainLabelMat, x, topK):
    '''
    通过距离向量 x 最近的 topK 个点的投票, 获得样本 x 的标签
    :param trainDataMat: 训练数据集
    :param trainLabelMat: 训练标签集
    :param x: 预测样本 x
    :param topK: 选择最邻近的样本数目
    :return: 预测样本 x 的标签
    '''

    # 建立用于存放向量 x 与训练集中样本距离的列表
    distList = [0] * len(trainLabelMat)

    # 遍历样本点, 计算与 x 的距离
    for i in range(len(trainDataMat)):

        # 获取训练集中样本 xi
        xi = trainDataMat[i]

        # 计算向量 x 与样本 xi 距离
        curDist = calcDist(x, xi)

        # 距离存入列表
        distList[i] = curDist

    # 获取与样本 x 最邻近的 topK 个样本
    topKList = np.argsort(np.array(distList))[:topK]

    # 投票列表
    # 由于类型 0 - 9, 故列表长度设为 10
    labelList = [0] * 10

    # 对 distList 进行遍历
    for index in topKList:

        # 根据标记数值在 distList 中投票
        labelList[int(trainLabelMat[index])] += 1

    # 返回票数最多的标签
    return labelList.index(max(labelList))

def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK = 25):
    '''
    测试正确率
    :param trainDataArr: 训练数据集
    :param trainLabelArr: 训练标签集
    :param testDataArr: 测试数据集
    :param testLabelArr: 测试标签集
    :param topK: 选择邻近点数目, 默认 25
    :return: 正确率
    '''

    # 开始训练
    print('START TRAINING')

    # 数据转为矩阵形式便于运算
    trainDataMat = np.mat(trainDataArr)
    testDataMat = np.mat(testDataArr)

    # 标签转为矩阵后转置便于计算
    # 转置是因为计算中取 label 的某一元素时, 1 * N 的矩阵无法用 label[i] 方式读取
    trainLabelMat = np.mat(trainLabelArr).T
    testLabelMat = np.mat(testLabelArr).T

    # 错误分类样本计数
    errorCount = 0

    # # 不借助 kd 树, 仅测试 200 个样本点
    # for i in range(200):
    #
    #     # 当前测试样本向量
    #     x = testDataMat[i]
    #
    #     # 预测当前向量标记
    #     y = getNearest(trainDataMat, trainLabelMat, x, topK)
    #
    #     # 若测试标签与实际不符, 错误样本计数 + 1
    #     if y != testLabelMat[i]:
    #         errorCount += 1
    #
    #     # 打印测试进度
    #     print('TEST %d:%d' % (i, 200))

    # kd 树
    kdRoot = kd.KDTree(trainDataMat, trainLabelMat)

    kd.KDTree.find_nearest_neighbour(kdRoot)

    # 返回正确率
    return 1 - (errorCount / 200)

# 主函数
if __name__ == '__main__':

    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取训练数据
    trainData, trainLabel = loadData("D:\\Mnist\\mnist_train.csv")

    # 获取测试数据
    testData, testLabel = loadData("D:\\Mnist\\mnist_test.csv")

    # 测试获得正确率
    accuracyRate = test(trainData, trainLabel, testData, testLabel, 25)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示正确率
    print('Accuracy Rate:', accuracyRate)

    # 显示用时时长
    print('Time Span:', endTime - startTime)


