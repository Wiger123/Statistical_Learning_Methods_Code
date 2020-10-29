# coding = utf-8
# Author: Wiger
# Date: 2020-10-29
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000
测试集数量: 10000
---------------
运行结果:
正确率: 81.72%
运行时长: 61.47s
'''

import pandas as pd
import numpy as np
import time

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

    # 打开文件
    fr = open(fileName, 'r')

    # 逐行读取

    # # 按行读取文件
    # for line in fr.readlines():
    #
    #     # 每一行数据按 ',' 进行切割, 返回字段列表
    #     curLine = line.strip().split(',')
    #     # 0-9: 标记, 由于是二分任务, 将 >= 5 的作为 1, < 5 为 -1
    #     if int(curLine[0]) >= 5:
    #         labelArr.append(1)
    #     else:
    #         labelArr.append(-1)
    #     # 存放标记
    #     # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素(标签)外将所有元素转换成 int 类型
    #     # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(不归一化, 成功率出现 bug, 一直为 100%)
    #     dataArr.append([int(num) / 255 for num in curLine[1:]])

    # pandas 读取, 更为简便

    # 打开文件, 按行读取
    df = pd.read_csv(fileName, header=None)

    # 获取数据 list
    # 此处需要整除255进行归一化
    dataArr = df.iloc[:, 1:] / 255

    # 获取标签 list
    # 0-9: 标记, 由于是二分任务, 将 >= 5 的作为 1, < 5 为 -1
    labelArr = pd.Series((num >= 5) for num in df.iloc[:, 0]).map({True: 1, False: -1})

    # 数据读取结束
    print('END READING DATA')

    return dataArr, labelArr

def perceptron(dataArr, labelArr, iter = 50):
    '''
    感知机训练
    :param dataArr: 训练集数据
    :param labelArr: 训练集标签
    :param iter: 迭代次数, 默认 50
    :return: 训练后的 w 和 b
    '''

    # 开始训练
    print('START TRAINING')

    # 数据转为矩阵形式便于运算
    # 转换后的数据每一个样本向量均为横向
    dataMat = np.mat(dataArr)

    # 标签转为矩阵后转置便于计算
    # 转置是因为计算中取 label 的某一元素时, 1 * N 的矩阵无法用 label[i] 方式读取
    labelMat = np.mat(labelArr).T

    # 获取矩阵大小
    m, n = np.shape(dataMat)

    # 初始化权重 w 为 n 个值为 0 的 1 * n 矩阵
    w = np.zeros((1, n))

    # 初始化偏置 b = 0
    b = 0

    # 初始化步长, 即梯度下降过程中的 n, 控制梯度下降速率
    h = 0.0001

    # Gram 矩阵为 n * n 矩阵, 可以用于加速计算, 不过会占用一定存储空间
    # iter 次迭代运算
    for k in range(iter):

        # 对于每一个样本进行梯度下降
        # 随机梯度下降: 每计算一个样本就针对该样本进行一次梯度下降
        for i in range(m):

            # 获取当前样本的向量
            xi = dataMat[i]

            # 获取当前样本对应的标签
            yi = labelMat[i]

            # 判断是否误分类: -yi(w * xi + b) >= 0
            # 此处 = 0, 即样本点在超平面上, 仍可以优化超平面, 故可视作误分类
            if -1 * yi * (w * xi.T + b) >= 0:

                # 对于误分类样本, 进行梯度下降更新参数
                w = w + h * yi * xi
                b = b + h * yi

        # 打印训练进度
        print('ROUND %d:%d TRAINING' % (k, iter))

    # 训练结束
    print('END TRAINING')

    # 返回 w, b
    return w, b

def test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr: 测试数据集
    :param labelArr: 测试标签集
    :param w: 训练获得的权重 w
    :param b: 训练获得的偏置 b
    :return: 正确率
    '''

    # 开始计算测试准确率
    print('START TESTING')

    # 数据转为矩阵形式便于运算
    dataMat = np.mat(dataArr)

    # 标签转为矩阵后转置便于计算
    labelMat = np.mat(labelArr).T

    # 获取矩阵大小
    m, n = np.shape(dataMat)

    # 错误分类样本计数
    errorCount = 0

    # 对所有样本进行测试
    for i in range(m):

        # 获取当前样本的向量
        xi = dataMat[i]

        # 获取当前样本对应的标签
        yi = labelMat[i]

        # 判断是否误分类: -yi(w * xi + b) >= 0
        if -1 * yi * (w * xi.T + b) >= 0 :
            errorCount += 1

    # 计算正确率
    accuracyRate = 1 - (errorCount / m)

    # 返回正确率
    return accuracyRate

# 主函数
# if __name__ == '__main__':
#     # 获取当前时间, 作为起始时间
#     startTime = time.time()
#
#     # 获取训练数据
#     trainData, trainLabel = pd.loadData("G:\\Statistical-Learning-Method_Code-master\\Mnist\\mnist_train.csv")
#
#     # 获取测试数据
#     testData, testLabel = pd.loadData("G:\\Statistical-Learning-Method_Code-master\\Mnist\\mnist_test.csv")
#
#     # 训练获得权重和正确率
#     w, b = pd.perceptron(trainData, trainLabel, 30)
#
#     # 测试获得正确率
#     accuracyRate = pd.test(testData, testLabel, w, b)
#
#     # 获取当前时间, 作为结束时间
#     endTime = time.time()
#
#     # 显示正确率
#     print('Accuracy Rate:', accuracyRate)
#
#     # 显示用时时长
#     print('Time Span:', endTime - startTime)
