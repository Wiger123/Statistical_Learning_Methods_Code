# coding = utf-8
# Author: Wiger
# Date: 2020-11-10
# Email: wiger@mail.ustc.edu.cn

'''
决策树算法: ID3, C4.5, CART
算法内容: 特征选择, 树的生成, 树的剪枝
ID3:
    特征选择: 信息增益最大
C4.5:
    特征选择: 信息增益比最大
CART:
    特征选择: 基尼系数最小
    剪枝: 计算损失函数: 固定 a, 获取使损失函数最小的子树
---------------
训练集: Mnist
训练集数量: 60000
测试集数量: 10000
---------------
运行结果:
ID3 (未剪枝)
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