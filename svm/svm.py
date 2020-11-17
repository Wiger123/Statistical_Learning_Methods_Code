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
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        #获取当前行，并按“，”切割成字段放入列表中
        #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        #split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        #在放入的同时将原先字符串形式的数据转换为0-1的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        #将标记信息放入标记集中
        #放入的同时将标记转换为整型
        #数字0标记为1  其余标记为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    #返回数据集和标记
    return dataArr, labelArr

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
        # 数字 1 - 9: -1
        if int(curLine[0]) == 0:
            labelArr.append(1)

        else:
            labelArr.append(-1)

    # 数据读取结束
    print('END READING DATA')

    # 返回数据集和标签
    return dataArr, labelArr