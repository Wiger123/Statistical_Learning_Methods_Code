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
        self.k = self.calc_kernel()

        # SVM 中偏置量 b
        self.b = 0

        # a: 拉格朗日乘子向量
        self.alpha = [0] * self.trainDataMat.shape[0]

        # SMO 中的 Ei
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]

        # 支持向量索引列表
        self.supportVecIndex = []

    def calc_kernel(self):
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

    def calc_gxi(self, i):
        '''
        g(xi) = Σ aj * yj * K(xi, xj) + b
        :param i: x 的下标
        :return: g(xi) 的值
        '''

        # 初始化 g(xi)
        gxi = 0

        # 根据书中 a 不等于 0 才参与计算
        # index 为非零 的 a 下标列表
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]

        # 遍历每一个非零 a
        for j in index:
            # 计算 g(xi)
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]

        # 求和结束后加上偏置 b
        gxi += self.b

        # 返回 g(xi)
        return gxi

    def satisfy_KKT(self, i):
        '''
        判断第 i 个 a 是否满足 KKT 条件
        :param i: a 的下标
        :return: 是否满足 KKT 条件
            True: 满足
            False: 不满足
        '''

        # g(x): 统计学习方法 7.104
        gxi = self.calc_gxi(i)

        # yi: 标签
        yi = self.trainLabelMat[i]

        # 检验是在松弛变量的范围内进行的, 故需要对 a 和 tolerance 进行判断
        # KKT 条件: 统计学习方法 7.111, 7.112, 7.113
        # 7.111: yi * g(xi) >= 1
        # -tolerance < a < tolerance
        if yi * gxi >= 1 and math.fabs(self.alpha[i]) < self.tolerance:
            # 满足 KKT
            return True

        # 7.112: yi * g(xi) = 1, 考虑到松弛变量, 采用绝对值减法
        # -tolerance < a < C + tolerance
        elif math.fabs(yi * gxi - 1) < self.tolerance and self.alpha[i] > -self.toler and self.alpha[i] < (self.C + self.toler):
            # 满足 KKT
            return True

        # 7.113: yi * g(xi) <= 1
        # C - tolerance < a < C + tolerance
        elif yi * gxi <= 1 and math.fabs(self.alpha[i] - self.C) < self.tolerence:
            # 满足 KKT
            return True

        # 其他情况均属于不满足 KKT, 返回 False
        return False

    def calc_Ei(self, i):
        '''
        根据 7.105 计算 Ei: 函数 g(x) 对输入 xi 的预测值与真实输出 yi 之差
        :param i: E 下标
        :return: 计算得到的 E 值
        '''

        # 计算 g(xi)
        gxi = self.calc_gxi(i)

        # Ei = g(xi) - yi
        return gxi - self.trainLabelMat[i]

    def get_alphaJ(self, E1, i):
        '''
        SMO 中选择第二个变量
        :param E1: 第一个变量的 E1
        :param i: 第一个变量的 a 的下标
        :return: E2, a2 的下标
        '''

        # 初始化 E2
        E2 = 0

        # 初始化 |E1 - E2| = -1
        max_E1_E2 = -1

        # 初始化第二个变量下标为 -1
        max_index = -1