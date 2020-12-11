# coding = utf-8
# Author: Wiger
# Date: 2020-12-10
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000
测试集数量: 10000
---------------
运行结果:
迭代次数: 50
正确率: 87.14%
运行时长: 411.27s
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

    # 存放数据集和标签的 list
    dataArr = [];
    labelArr = [];

    # 经典逐行读取

    # 打开文件
    fr = open(fileName, 'r')

    # 按行读取文件
    for line in fr.readlines():
        # 每一行数据按 ',' 进行切割, 返回字段列表
        curLine = line.strip().split(',')

        # 0-9: 标记, 由于是二分任务, 将 >= 5 的作为 1, < 5 为 0
        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(0)

        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素(标签)外将所有元素转换成 int 类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(不归一化, 成功率出现 bug, 一直为 100%)
        dataArr.append([int(num) / 255 for num in curLine[1:]])

    return dataArr, labelArr

def predict(w, x):
    '''
    预测样本的标签
    :param w: 需要通过训练得到的表达式中 w
    :param x: 预测的样本向量
    :return: 预测结果
    '''
    # dot 为向量点积, 结果为 w * x
    wx = np.dot(w, x)

    # 计算标签为 1 的概率
    P1 = np.exp(wx) / (1 + np.exp(wx))

    # 若标签为 1 的概率大于 0.5, 则返回 1
    if P1 >= 0.5:
        return 1
    # 否则返回0
    else:
        return 0

def logistic_regression(trainDataArr, trainLabelArr, iter = 200):
    '''
    逻辑斯蒂回归训练
    :param trainDataArr: 训练数据集
    :param trainLabelArr: 训练标签集
    :param iter: 迭代次数
    :return: 训练得到的 w
    '''
    # dodo 按照书中 6.5 的表达方式, 在 w 向量后加了一列 b, 在 x 向量后加了一列 1
    # w = (w1, w2, w3, ... wn, b).T
    # x = (x1, x2, x3, ... xn, 1).T
    # 通过这种计算方式 w * x 直接代替了原来的 w * x + b
    for i in range(len(trainDataArr)):
        trainDataArr[i].append(1)

    # 列表转为数组形式, 便于计算
    # 由于 python 运行内存上限, 这里不进行批量转换, 在下面的实际训练过程中, 每次迭代对每次样本进行转换
    # trainDataArr = np.array(trainDataArr)

    # 初始化 w, 同时要考虑到 b, 长度也要增加一项
    # w = np.zeros(trainDataArr.shape[1])
    w = np.zeros(len(trainDataArr[0]))

    # 训练步长
    h = 0.001

    # 迭代 iter 次进行随机梯度下降
    for i in range(iter):
        # 每次遍历所有样本, 进行随机梯度下降
        # for j in range(trainDataArr.shape[0]):
        for j in range(len(trainDataArr)):
            # 极大化似然函数 -> 对似然函数求和部分的每一项单独求导 w -> 梯度上升
            wx = np.dot(w, trainDataArr[j])

            # 样本对应标签
            # yi = trainLabelArr[j]
            yi = np.array(trainLabelArr[j])

            # 样本向量
            # xi = trainDataArr[j]
            xi = np.array(trainDataArr[j])

            # 梯度上升
            # 单步步长乘上求导后的结果即为训练单次变化
            w += h * (xi * yi - (np.exp(wx) * xi) / (1 + np.exp(wx)))

            # 打印训练进度
            if j % 1000 == 0:
                print("第 {} 次迭代, 已训练完 {} 个样本".format(i, j))

    # 返回训练后的 w
    return w


def test(testDataArr, testLabelArr, w):
    '''
    测试准确率
    :param testDataArr: 测试数据集
    :param testLabelArr: 测试标签集
    :param w: 训练获得的权重 w
    :return: 正确率
    '''

    # 与训练过程一致, 所有样本增加一维, 值为 1
    for i in range(len(testDataArr)):
        testDataArr[i].append(1)

    # 错误分类样本计数
    errorCount = 0

    # 对所有样本进行测试
    for i in range(len(testDataArr)):
        # 判断预测结果与标记是否一致
        if testLabelArr[i] != predict(w, testDataArr[i]):
            errorCount += 1

    # 计算正确率
    accuracyRate = 1 - (errorCount / len(testDataArr))

    # 返回正确率
    return accuracyRate

# 主函数
if __name__ == '__main__':

    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取训练数据
    print('START TO READ DATA')
    trainData, trainLabel = loadData("D:\\Mnist\\mnist_train.csv")

    # 获取测试数据
    testData, testLabel = loadData("D:\\Mnist\\mnist_test.csv")
    print('END READING DATA')

    # 训练获得权重
    w = logistic_regression(trainData, trainLabel, 50)

    # 测试获得正确率
    print('START TESTING')
    accuracyRate = test(testData, testLabel, w)

    # 显示正确率
    print('Accuracy Rate:', accuracyRate)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示用时时长
    print('Time Span:', endTime - startTime)
