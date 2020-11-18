# coding = utf-8
# Author: Wiger
# Date: 2020-11-17
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000 (实际使用: )
测试集数量: 10000 (实际使用: )
---------------
运行结果:
正确率:
运行时长:
'''

import time
import numpy as np
import math
import random

def loadData(fileName):
    '''
    加载 Mnist 数据集
    :param fileName: 数据集路径
    :return: 数据 list 和 标签 list
    '''

    # 开始读取数据
    print('START TO READ DATA')

    # 数据数组
    dataArr = []

    # 标签数组
    labelArr = []

    # 打开文件
    fr = open(fileName)

    # 遍历文件中的每一行
    for line in fr.readlines():
        # 获取当前行, 并按 "," 分割成字段放入列表中
        # strip: 去掉每行字符串首尾指定的字符(默认空格或换行符)
        # split: 按照指定的字符将字符串切割成每个字段, 返回列表形式
        curLine = line.strip().split(',')

        # 将每行中除标签外的数据放入数据集中(curLine[0]为标签信息)
        # 在放入的同时将字符串形式的数据转换为 0 - 1 的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1 : ]])

        # 将标签信息放入标签集中
        # 放入的同时将标签转换为整型
        # 数字 0: 1
        if int(curLine[0]) == 0:
            labelArr.append(1)

        # 数字 1 - 9: -1
        else:
            labelArr.append(-1)

    # 数据读取结束
    print('END READING DATA')

    # 返回数据集和标签
    return dataArr, labelArr

class SVM:
    '''
    SVM (支持向量机)类
    '''
    def __init__(self, trainDataList, trainLabelList, sigma = 10, C = 200, tolerance = 0.001):
        '''
        SVM 参数初始化
        :param trainDataList: 训练数据集
        :param trainLabelList: 训练标签集
        :param sigma: 高斯核分母中的 σ
        :param C: 软间隔中的惩罚系数
        :param tolerance: 松弛变量
        '''
        # 训练数据集矩阵
        self.trainDataMat = np.mat(trainDataList)

        # 训练标签集矩阵, 转置为列向量便于计算
        self.trainLabelMat = np.mat(trainLabelList).T

        # m: 训练样本数目
        # n: 样本特征数目
        self.m, self.n = np.shape(self.trainDataMat)

        # 高斯核分母中的 σ
        self.sigma = sigma

        # 惩罚参数
        self.C = C

        # 松弛变量
        self.tolerance = tolerance

        # 核函数 (初始化时计算)
        self.k = self.calcKernel()

        # SVM 中偏置量 b
        self.b = 0

        # a: 拉格朗日乘子向量
        self.alpha = [0] * self.trainDataMat.shape[0]

        # SMO 中的 Ei
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]

        # 支持向量索引列表
        self.supportVecIndex = []

    def calcKernel(self):
        '''
        计算高斯核函数: K(x,z) = exp(- (||x-z||^2) / (2 * σ^2))
        :return: 高斯核矩阵
        '''

        # 初始化高斯核结果矩阵: 训练样本数目 m * 训练样本数目 m
        # k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        # 大循环遍历获取 Xi (核函数中的 x)
        for i in range(self.m):
            # 每 100 个打印一次
            if i % 100 == 0:
                print('Kernel: ', i, self.m)

            # 获取单个样本作为 x
            X = self.trainDataMat[i, :]

            # 小遍历获取 Xj (核函数中的 z)
            # 遍历从 i 开始 (矩阵 k[i][j] = k[j][i], 从 i 开始避免重复运算)
            for j in range(i, self.m):
                # 获取单个样本作为 z
                Z = self.trainDataMat[j, :]

                # 计算分子: ||x-z||^2
                result = (X - Z) * (X - Z).T

                # 分子除以分母后取指数
                result = np.exp(-1 * result / (2 * self.sigma * self.sigma))

                # 高斯核保存
                k[i][j] = result
                k[j][i] = result

        # 返回高斯核矩阵
        return k