# coding = utf-8
# Author: Wiger
# Date: 2020-11-04
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000
测试集数量: 10000
---------------
运行结果:
正确率:
运行时长:
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
        # 在放入的同时将原先字符串形式的数据转换为整型
        # 此外将数据进行了二值化处理, 大于 128 的转换成 1, 小于 128 的转换成 0, 方便后续计算
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])

        # 将标签信息放入标签集中
        # 放入的同时将标签转换为整型
        labelArr.append(int(curLine[0]))

    # 数据读取结束
    print('END READING DATA')

    # 返回数据集和标签
    return dataArr, labelArr

def calcProbability(trainDataArr, trainLabelArr):
    '''
    通过训练集计算先验概率分布和条件概率分布
    :param trainDataArr: 训练数据集
    :param trainLabelArr: 训练标记集
    :return: 先验概率分布和条件概率分布
    '''

    # 特征数目
    feature_count = 784

    # 标签类别数目
    label_count = 10

    # 每一标签先验概率
    # Py[i]: P(Y = i) 的概率
    # 数据: 10 行 1 列
    Py = np.zeros((label_count, 1))

    # 对于每一个标签类, 逐一计算先验概率
    for i in range(label_count):
        # np.mat(trainLabelArr) == i: 标签转为矩阵形式, 每一位与 i 进行比较, 相同则该位变为 True, 反之变为 False
        # np.sum(np.mat(trainLabelArr) == i): 计算矩阵中 True 个数, 即统计训练数据中标签 i 的总数目
        # (np.sum(np.mat(trainLabelArr) == i)) + 1: 贝叶斯估计: 为防止概率为 0 影响后续计算产生偏差, 分子加上 λ = 1 (拉普拉斯平滑)
        # len(trainLabelArr) + 10: 贝叶斯估计: 为防止概率为 0 影响后续计算产生偏差, 分母加上 Kλ = 10 * 1 (拉普拉斯平滑)
        # ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10): 每一标签先验概率的贝叶斯估计
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)

    # 取对数讲连乘转为求和
    Py = np.log(Py)

    # 条件概率
    # Px_y[label][j][x[j]]: P(Xj = x[j] | Y = label) 同样可表示为 P(X1 = x[1], x2 = x[2], ... | Y = label)
    # 初始化为全 0 矩阵, 用于存放所有情况下的条件概率
    # 由于已经归一化, 故特征值仅为 0 和 1
    Px_y = np.zeros((label_count, feature_count, 2))

    # 对标记集进行遍历, 统计计数
    for i in range(len(trainLabelArr)):
        # 获取当前循环所使用的标记
        label = trainLabelArr[i]

        # 获取当前要处理的样本
        x = trainDataArr[i]

        # 对该样本的每一维特征进行遍历
        for j in range(feature_count):
            # 在矩阵中对应位置加1
            # 这里还没有计算条件概率, 先把所有数累加, 在后续步骤中再求对应的条件概率
            Px_y[label][j][x[j]] += 1

    # 对标记集进行遍历, 计算条件概率
    for label in range(label_count):
        # 循环每一个标记对应的每一个特征
        for j in range(feature_count):
            # 获取 y = label, 第 j 个特征为 0 的个数
            Px_y0 = Px_y[label][j][0]

            # 获取 y = label, 第 j 个特征为 1 的个数
            Px_y1 = Px_y[label][j][1]

            # 贝叶斯估计: 分子 + λ; 分母 + λ * Sj（为每个特征可取值个数）;
            # λ = 1; Sj =2;
            # P(X[j] = 0 | Y = label) 的条件概率
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))

            # P(X[j] = 1 | Y = label) 的条件概率
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    # 返回先验概率与条件概率
    return Py, Px_y

def naiveBayes(Py, Px_y, x):
    '''
    通过朴素贝叶斯进行概率估计
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param x: 估计样本
    :return: 所有标签的估计概率
    '''

    # 特征数目
    feature_count = 784

    # 标签类别数目
    label_count = 10

    # 所有标签的估计概率数组
    P = [0] * label_count

    # 根据朴素贝叶斯, 计算出 10 个标签类的概率, 选出能使表达式(先验概率 * 条件概率连乘)概率最大的 y, 即为估计结果
    for i in range(label_count):
        # 训练过程中概率进行了 log 对数处理, 连乘转换为连加
        # 求和项
        sum = 0

        # 对每一个特征的条件概率
        for j in range(feature_count):
            # 累加: 标签: i;  特征: j; 估计样本该特征的值: x[j] (即样本该特征值为 x[j] 的概率);
            sum += Px_y[i][j][x[j]]

        # 与先验概率相加
        P[i] = sum + Py[i]

    # 返回概率最大的标签
    return P.index(max(P))

def test(Py, Px_y, testDataArr, testLabelArr):
    '''
    对测试集进行测试
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param testDataArr: 测试集数据
    :param testLabelArr: 测试集标记
    :return: 准确率
    '''

    # 错误样本计数
    errorCount = 0

    # 对所有样本进行测试
    for i in range(len(testDataArr)):
        # 获取预测值
        predict = naiveBayes(Py, Px_y, testDataArr[i])

        # 与答案比较
        if predict != testLabelArr[i]:
            # 错误则计数加一
            errorCount += 1

    # 计算正确率
    accuracyRate = 1 - (errorCount / len(testDataArr))

    # 返回正确率
    return accuracyRate

# 主函数
if __name__ == '__main__':
    # 获取训练数据
    trainDataArr, trainLabelArr = loadData("D:\\Mnist\\mnist_train.csv")

    # 获取测试数据
    testDataArr, testLabelArr = loadData("D:\\Mnist\\mnist_test.csv")

    # 开始读取数据
    print('START TO TRAIN')

    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 开始训练
    Py, Px_y = calcProbability(trainDataArr, trainLabelArr)

    # 测试准确率
    accuracyRate = test(Py, Px_y, testDataArr, testLabelArr)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 开始读取数据
    print('END TRAINING')

    # 显示正确率
    print('Accuracy Rate:', accuracyRate)

    # 显示用时时长
    print('Time Span:', endTime - startTime)
