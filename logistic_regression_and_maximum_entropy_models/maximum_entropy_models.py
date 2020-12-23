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
from collections import defaultdict

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

    def calcEp_xy(self):
        '''
        参考 82 页最下方公式
        计算特征函数 f(x, y) 关于经验分布 P_(x, y) 的期望值
        :return: Ep_(f)
        '''

        # 初始化 Ep_xy 列表, 长度为 n
        Ep_xy = [0] * self.n

        # 遍历每一个特征
        for feature in range(self.featureNum):
            # 遍历每个特征中的 (x, y) 对
            for (x, y) in self.fixy[feature]:
                # 获得其 id
                id = self.xy2idDict[feature][(x, y)]
                # 将计算得到的 Ep_xy 写入对应的位置中
                # fixy 中存放所有对在训练集中出现过的次数, 除以训练集总长度 N 就是概率
                Ep_xy[id] = self.fixy[feature][(x, y)] / self.N

        # 返回期望
        return Ep_xy

    def calcEpxy(self):
        '''
        参考 83 页最上方公式
        计算特征函数 f(x, y) 关于模型 P(y|x) 与经验分布 P_(x) 的期望值
        :return: Ep(f)
        '''

        # 初始化 Epxy 列表, 长度为 n
        Epxy = [0] * self.n

        #对于每一个样本进行遍历
        for i in range(self.N):
            # 初始化公式中的 P(y|x) 列表
            Pwxy = [0] * 2

            # 计算 P(y = 0 | X)
            # 注: X 表示是一个样本的全部特征, x 表示单个特征, 这里是全部特征的一个样本
            Pwxy[0] = self.calcPwy_x(self.trainDataList[i], 0)

            # 计算 P(y = 1 | X)
            Pwxy[1] = self.calcPwy_x(self.trainDataList[i], 1)

            # 遍历每一个特征
            for feature in range(self.featureNum):
                # 对 y = 0 和 y = 1 进行计算
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]

        # 返回期望
        return Epxy

    def createSearchDict(self):
        '''
        创建查询字典
        xy2idDict：通过(x,y)对找到其id,所有出现过的xy对都有一个id
        id2xyDict：通过id找到对应的(x,y)对
        '''
        # 设置xy搜多id字典
        # 这里的x指的是单个的特征，而不是某个样本，因此将特征存入字典时也需要存入这是第几个特征
        # 这一信息，这是为了后续的方便，否则会乱套。
        # 比如说一个样本X = (0, 1, 1) label =(1)
        # 生成的标签对有(0, 1), (1, 1), (1, 1)，三个(x，y)对并不能判断属于哪个特征的，后续就没法往下写
        # 不可能通过(1, 1)就能找到对应的id，因为对于(1, 1),字典中有多重映射
        # 所以在生成字典的时总共生成了特征数个字典，例如在mnist中样本有784维特征，所以生成784个字典，属于
        # 不同特征的xy存入不同特征内的字典中，使其不会混淆
        xy2idDict = [{} for i in range(self.featureNum)]
        # 初始化id到xy对的字典。因为id与(x，y)的指向是唯一的，所以可以使用一个字典
        id2xyDict = {}
        # 设置缩影，其实就是最后的id
        index = 0
        # 对特征进行遍历
        for feature in range(self.featureNum):
            # 对出现过的每一个(x, y)对进行遍历
            # fixy：内部存放特征数目个字典，对于遍历的每一个特征，单独读取对应字典内的(x, y)对
            for (x, y) in self.fixy[feature]:
                # 将该(x, y)对存入字典中，要注意存入时通过[feature]指定了存入哪个特征内部的字典
                # 同时将index作为该对的id号
                xy2idDict[feature][(x, y)] = index
                # 同时在id->xy字典中写入id号，val为(x, y)对
                id2xyDict[index] = (x, y)
                # id加一
                index += 1
        # 返回创建的两个字典
        return xy2idDict, id2xyDict

    def calc_fixy(self):
        '''
        计算(x, y)在训练集中出现过的次数
        :return:
        '''
        # 建立特征数目个字典，属于不同特征的(x, y)对存入不同的字典中，保证不被混淆
        fixyDict = [defaultdict(int) for i in range(self.featureNum)]
        # 遍历训练集中所有样本
        for i in range(len(self.trainDataList)):
            # 遍历样本中所有特征
            for j in range(self.featureNum):
                # 将出现过的(x, y)对放入字典中并计数值加1
                fixyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] += 1
        # 对整个大字典进行计数，判断去重后还有多少(x, y)对，写入n
        for i in fixyDict:
            self.n += len(i)
        # 返回大字典
        return fixyDict

    def calcPwy_x(self, X, y):
        '''
        计算“6.23 最大熵模型的学习” 式6.22
        :param X: 要计算的样本X（一个包含全部特征的样本）
        :param y: 该样本的标签
        :return: 计算得到的Pw(Y|X)
        '''
        # 分子
        numerator = 0
        # 分母
        Z = 0
        # 对每个特征进行遍历
        for i in range(self.featureNum):
            # 如果该(xi,y)对在训练集中出现过
            if (X[i], y) in self.xy2idDict[i]:
                # 在xy->id字典中指定当前特征i，以及(x, y)对：(X[i], y)，读取其id
                index = self.xy2idDict[i][(X[i], y)]
                # 分子是wi和fi(x，y)的连乘再求和，最后指数
                # 由于当(x, y)存在时fi(x，y)为1，因为xy对肯定存在，所以直接就是1
                # 对于分子来说，就是n个wi累加，最后再指数就可以了
                # 因为有n个w，所以通过id将w与xy绑定，前文的两个搜索字典中的id就是用在这里
                numerator += self.w[index]
            # 同时计算其他一种标签y时候的分子，下面的z并不是全部的分母，再加上上式的分子以后
            # 才是完整的分母，即z = z + numerator
            if (X[i], 1 - y) in self.xy2idDict[i]:
                # 原理与上式相同
                index = self.xy2idDict[i][(X[i], 1 - y)]
                Z += self.w[index]
        # 计算分子的指数
        numerator = np.exp(numerator)
        # 计算分母的z
        Z = np.exp(Z) + numerator
        # 返回Pw(y|x)
        return numerator / Z

    def maxEntropyTrain(self, iter=500):
        # 设置迭代次数寻找最优解
        for i in range(iter):
            # 单次迭代起始时间点
            iterStart = time.time()
            # 计算“6.2.3 最大熵模型的学习”中的第二个期望（83页最上方哪个）
            Epxy = self.calcEpxy()
            # 使用的是IIS，所以设置sigma列表
            sigmaList = [0] * self.n
            # 对于所有的n进行一次遍历
            for j in range(self.n):
                # 依据“6.3.1 改进的迭代尺度法” 式6.34计算
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / Epxy[j])
            # 按照算法6.1步骤二中的（b）更新w
            self.w = [self.w[i] + sigmaList[i] for i in range(self.n)]
            # 单次迭代结束
            iterEnd = time.time()
            # 打印运行时长信息
            print('iter:%d:%d, time:%d' % (i, iter, iterStart - iterEnd))

    def predict(self, X):
        '''
        预测标签
        :param X:要预测的样本
        :return: 预测值
        '''
        # 因为y只有0和1，所有建立两个长度的概率列表
        result = [0] * 2
        # 循环计算两个概率
        for i in range(2):
            # 计算样本x的标签为i的概率
            result[i] = self.calcPwy_x(X, i)
        # 返回标签
        # max(result)：找到result中最大的那个概率值
        # result.index(max(result))：通过最大的那个概率值再找到其索引，索引是0就返回0，1就返回1
        return result.index(max(result))

    def test(self):
        '''
        对测试集进行测试
        :return:
        '''
        # 错误值计数
        errorCnt = 0
        # 对测试集中所有样本进行遍历
        for i in range(len(self.testDataList)):
            # 预测该样本对应的标签
            result = self.predict(self.testDataList[i])
            # 如果错误，计数值加1
            if result != self.testLabelList[i]:   errorCnt += 1
        # 返回准确率
        return 1 - errorCnt / len(self.testDataList)

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

    # 初始化最大熵类
    maxEnt = maxEnt(trainData[:20000], trainLabel[:20000], testData, testLabel)

    # 开始训练
    print('START TO TRAIN')
    maxEnt.maxEntropyTrain()

    # 开始测试
    print('START TO TEST')
    accuracy = maxEnt.test()
    print('ACCURACY:', accuracy)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示用时时长
    print('Time Span:', endTime - startTime)
