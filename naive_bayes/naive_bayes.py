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

def naive_bayes(Py, Px_y, x):
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


# 主函数
if __name__ == '__main__':
    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取训练数据
    trainDataArr, trainLabelArr = loadData("D:\\Mnist\\mnist_train.csv")

    # 获取测试数据
    testDataArr, testLabelArr = loadData("D:\\Mnist\\mnist_test.csv")

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示正确率
    # print('Accuracy Rate:', accuracyRate)

    # 显示用时时长
    print('Time Span:', endTime - startTime)
