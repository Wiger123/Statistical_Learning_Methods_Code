# coding = utf-8
# Author: Wiger
# Date: 2020-11-17
# Email: wiger@mail.ustc.edu.cn

'''
训练集: Mnist
训练集数量: 60000 (实际使用: 1000)
测试集数量: 10000 (实际使用: 100)
---------------
运行结果:
正确率: 97%
运行时长: 57.47s
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

    def isSatisfyKKT(self, i):
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
        elif math.fabs(yi * gxi - 1) < self.tolerance and self.alpha[i] > -self.tolerance and self.alpha[i] < (self.C + self.tolerance):
            # 满足 KKT
            return True

        # 7.113: yi * g(xi) <= 1
        # C - tolerance < a < C + tolerance
        elif yi * gxi <= 1 and math.fabs(self.alpha[i] - self.C) < self.tolerance:
            # 满足 KKT
            return True

        # 其他情况均属于不满足 KKT, 返回 False
        return False

    def calcEi(self, i):
        '''
        根据 7.105 计算 Ei: 函数 g(x) 对输入 xi 的预测值与真实输出 yi 之差
        :param i: E 下标
        :return: 计算得到的 E 值
        '''

        # 计算 g(xi)
        gxi = self.calc_gxi(i)

        # Ei = g(xi) - yi
        return gxi - self.trainLabelMat[i]

    def getAlphaJ(self, E1, i):
        '''
        SMO 中选择第二个变量
        :param E1: 第一个变量的 E1
        :param i: 第一个变量的 a 的下标
        :return: E2, a2 的下标
        '''

        # 初始化 E2
        E2 = 0

        # 初始化 |E1 - E2| = -1
        maxE1_E2 = -1

        # 初始化第二个变量下标为 -1
        maxIndex = -1

        # 获得 Ei 非 0 的对应索引组成的列表, 列表内容为非 0 的 Ei 下标 i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]

        # 对每个非零 Ei 的下标 i 进行遍历
        for j in nozeroE:
            # 计算 E2
            E2_tmp = self.calcEi(j)

            # 如果 |E1-E2| 大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                # 更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)

                # 更新最大值 E2
                E2 = E2_tmp

                # 更新最大值 E2 的索引 j
                maxIndex = j

        # 如果列表中没有非 0 元素了(对应程序最开始运行时的情况)
        if maxIndex == -1:
            maxIndex = i

            while maxIndex == i:
                # 获得随机数,如果随机数与第一个变量的下标 i 一致则重新随机
                maxIndex = int(random.uniform(0, self.m))

            # 获得 E2
            E2 = self.calcEi(maxIndex)

        # 返回第二个变量的 E2 值以及其索引
        return E2, maxIndex

    def train(self, iter=100):
        # iterStep: 迭代次数, 超过设置次数还未收敛则强制停止
        # parameterChanged: 单次迭代中有参数改变则增加 1
        iterStep = 0;
        parameterChanged = 1

        # 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        # parameterChanged == 0 时表示上次迭代没有参数改变, 如果遍历了一遍都没有参数改变, 说明达到了收敛状态, 可以停止
        while (iterStep < iter) and (parameterChanged > 0):
            # 打印当前迭代轮数
            print('iter:%d:%d' % (iterStep, iter))

            # 迭代步数加 1
            iterStep += 1

            # 新的一轮将参数改变标志位重新置 0
            parameterChanged = 0

            # 大循环遍历所有样本, 用于找 SMO 中第一个变量
            for i in range(self.m):
                # 查看第一个遍历是否满足 KKT 条件, 如果不满足则作为 SMO 中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    # 如果下标为 i 的 a 不满足 KKT 条件, 则进行优化
                    # 第一个变量 a 的下标 i 已经确定, 接下来按照 7.4.2 选择变量 2
                    # 由于变量 2 的选择中涉及到 |E1 - E2|, 因此先计算 E1
                    E1 = self.calcEi(i)

                    # 选择第 2 个变量
                    E2, j = self.getAlphaJ(E1, i)

                    # 获得两个变量的标签
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]

                    # 复制 a 值作为 old 值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]

                    # 依据标签是否一致来生成不同的 L 和 H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)

                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)

                    # 如果两者相等, 说明该变量无法再优化, 直接跳到下一次循环
                    if L == H:
                        continue

                    # 计算 a 的新值
                    # 更新 a2 值
                    # 先获得几个 k 值, 用于计算 7.106 中的分母 η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]

                    # 依据式 7.106 更新 a2, 该 a2 还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)

                    # 剪切a2
                    if alphaNew_2 < L:
                        alphaNew_2 = L

                    elif alphaNew_2 > H:
                        alphaNew_2 = H

                    # 依据式 7.109 更新 a1
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    # 依据 7.4.2 变量的选择方法 第三步式 7.115 和 7.116 计算 b1 和 b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据 a1 和 a2 的值范围确定新 b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New

                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New

                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入, 进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew
                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    # 如果 a2 的改变量过于小, 就认为该参数未改变, 不增加 parameterChanged 值
                    # 反之则自增 1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                # 打印迭代轮数, i 值, 该迭代轮数修改 a 数目
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))

        # 全部计算结束后, 重新遍历一遍 a, 查找里面的支持向量
        for i in range(self.m):
            # 如果 a > 0, 说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        '''
        计算核函数
        :param x1: 向量 1
        :param x2: 向量 2
        :return: 核函数结果
        '''
        # 按照 7.90 计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))

        # 返回结果
        return np.exp(result)

    def predict(self, x):
        '''
        对样本的标签进行预测
        公式依据 7.3.4 非线性支持向量分类机 中的式 7.94
        :param x: 要预测的样本 x
        :return: 预测结果
        '''
        result = 0

        for i in self.supportVecIndex:
            # 遍历所有支持向量, 计算求和式
            # 如果是非支持向量, 求和子式必为 0, 没有必须进行计算
            # 先单独将核函数计算出来
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], np.mat(x))

            # 对每一项子式进行求和, 最终计算得到求和项的值
            result += self.alpha[i] * self.trainLabelMat[i] * tmp

        # 求和项计算结束后加上偏置 b
        result += self.b

        # 使用 sign 函数返回预测结果
        return np.sign(result)

    def test(self, testDataList, testLabelList):
        '''
        测试
        :param testDataList: 测试数据集
        :param testLabelList: 测试标签集
        :return: 正确率
        '''
        # 错误计数值
        errorCnt = 0

        # 遍历测试集所有样本
        for i in range(len(testDataList)):
            # 打印目前进度
            print('test:%d:%d' % (i, len(testDataList)))

            # 获取预测结果
            result = self.predict(testDataList[i])

            # 如果预测与标签不一致, 错误计数值加一
            if result != testLabelList[i]:
                errorCnt += 1

        # 返回正确率
        return 1 - errorCnt / len(testDataList)

if __name__ == '__main__':
    # 起始时间
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainDataList, trainLabelList = loadData('D:\Mnist\mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testDataList, testLabelList = loadData('D:\Mnist\mnist_test.csv')

    # 初始化SVM类
    print('start init SVM')
    svm = SVM(trainDataList[:1000], trainLabelList[:1000], 10, 200, 0.001)

    # 开始训练
    print('start to train')
    svm.train()

    # 开始测试
    print('start to test')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('the accuracy is:%d' % (accuracy * 100), '%')

    # 打印时间
    print('time span:', time.time() - start)