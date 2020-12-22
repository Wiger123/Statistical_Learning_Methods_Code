# coding = utf-8
# Author: Wiger
# Date: 2020-12-22
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000 (实际使用: 20000)
测试集数量: 10000
---------------
运行结果:
正确率: %
运行时长: s
'''

import numpy as np
import time

def loadData(fileName):
    '''
    加载 Mnist 数据集
    :param fileName: 数据集路径
    :return: 数据 list 和 标签 list
    '''

    # 存放数据集和标签的 list
    dataList = [];
    labelList = [];

    # 打开文件
    fr = open(fileName, 'r')

    # 按行读取文件
    for line in fr.readlines():
        # 每一行数据按 ',' 进行切割, 返回字段列表
        curLine = line.strip().split(',')

        # 前几节以 5 作为分割效果较好, 这里还是按照 0 分类
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)

        # 数据 01 二值化
        dataList.append([int(int(num) > 128) for num in curLine[1:]])

    return dataList, labelList

class maxEnt:
    '''
    最大熵类
    '''
    def __init__(self, trainDataList, trainLabelList, testDataList, testLabelList):
        '''
        参数初始化
        :param trainDataList: 训练数据集
        :param trainLabelList: 训练标签集
        :param testDataList: 测试数据集
        :param testLabelList: 测试标签集
        '''
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.testDataList = testDataList
        self.testLabelList = testLabelList

        # 特征数目 (784)
        self.featureNum = len(trainDataList[0])

        # 训练集长度 (20000)
        self.N = len(trainDataList)

        # 训练集中 (xi, y) 对数目
        self.n = 0

        # 所有 (x, y) 对出现的次数
        self.fixy = self.calc_fixy()

        # 改进的迭代尺度算法 IIS 中的参数 M
        self.M = 10000
        
        # Pw(y|x) 中的 w
        self.w = [0] * self.n

        # (x, y)->id 和 id->(x, y) 的搜索字典
        self.xy2idDict, self.id2xyDict = self.createSearchDict()

        # Ep_xy 期望值
        self.Ep_xy = self.calcEp_xy()

if __name__ == '__main__':
    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取训练集及标签
    print('START TO READ DATA')
    trainData, trainLabel = loadData("D:\\Mnist\\mnist_train.csv")

    # 获取测试集及标签
    # 获取测试数据
    testData, testLabel = loadData("D:\\Mnist\\mnist_test.csv")
    print('END READING DATA')

    print(trainData)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示用时时长
    print('Time Span:', endTime - startTime)
