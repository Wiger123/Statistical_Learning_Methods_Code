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
正确率:
运行时长:
'''

import pandas as pd
import time

def loadData(fileName):
    '''
    加载 Mnist 数据集
    :param fileName: 数据集路径
    :return: list 形式的数据集和标记
    '''

    print('START TO READ DATA')

    # 存放数据集和标签的 list
    dataArr = [];
    labelArr = [];

    # 打开文件, 按行读取
    df = pd.read_csv(fileName, header=None)
    # 获取数据 list
    dataArr = df.iloc[:, 1:]
    # 获取标签 list
    # 0-9: 标记, 由于是二分任务, 将 >= 5 的作为 1, < 5 为 -1
    labelArr = pd.Series((num >= 5) for num in df.iloc[:, 0]).map({True: 1, False: -1})

    return dataArr, labelArr
