# coding = utf-8
# Author: Wiger
# Date: 2020-11-10
# Email: wiger@mail.ustc.edu.cn

'''
例:
C1 C2 ... Ck: 决定是否贷款
    C1: 贷款
    C2: 不贷款
D1 D2 ... Dn: 某一特征上的分类
    A1: 年龄
        D1: 青年
        D2: 中年
        D3: 老年
    A2: 工作
        D1: 有工作
        D2: 无工作
    A3: 房屋
        D1: 有房
        D2: 无房
'''

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

def majorClass(labelArr):
    '''
    寻找当前标签集合中数目最多的标签
    :param labelArr: 标签集合
    :return: 数目最多的标签
    '''

    # 建立字典, 统计各类标签
    classDict = {}

    # 遍历标签
    for i in range(len(labelArr)):
        # 字典中存在该标签
        if labelArr[i] in classDict.keys():
            # 标签数目加一
            classDict[labelArr[i]] += 1
        # 无该标签
        else:
            # 创建该标签
            classDict[labelArr[i]] = 1

    # 字典进行降序排序
    classSort = sorted(classDict.items(), key = lambda x:x[1], reverse = True)
    
    # 返回第一项(数目最多)的标签
    return classSort[0][0]

def calc_H_D(trainLabelArr):
    '''
    计算数据集 D 的经验熵, 熵只与分布概率分布有关, 与数据本身的值无关
    :param trainLabelArr: 训练标签集合
    :return: 经验熵
    '''

    # 经验熵初始化
    H_D = 0

    # 标签放入集合中
    trainLabelSet = set([label for label in trainLabelArr])

    # 计算每一个出现过的标签
    for i in trainLabelSet:
        # 计算概率
        # trainLabelArr == i: 当前标签集中为该标签的的位置
        # 例: a = [1, 0, 0, 1], c = (a == 1): c == [True, false, false, True]
        # trainLabelArr[trainLabelArr == i]: 获得指定标签的样本
        # trainLabelArr[trainLabelArr == i].size: 获得指定标签的样本的大小, 即 |Ck|
        # trainLabelArr.size: 整个标签集的数量(样本集的数量), 即 |D|
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size

        # 经验熵累加求和
        H_D += -1 * p * np.log2(p)

    # 返回经验熵
    return H_D

def calcH_D_A(trainDataArr_DevFeature, trainLabelArr):
    '''
    计算经验条件熵
    :param trainDataArr_DevFeature: 切割后只有 feature 那列数据的数组
    :param trainLabelArr: 标签集数组
    :return: 经验条件熵
    '''

    # 初始为 0
    H_D_A = 0

    # 在 featue 那列放入集合中, 是为了根据集合中的数目知道该 feature 目前可取值数目是多少
    trainDataSet = set([label for label in trainDataArr_DevFeature])

    # 对于每一个特征取值遍历计算条件经验熵的每一项
    for i in trainDataSet:
        # 计算 H(D|A)
        # trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size: |Di| / |D|
        # calc_H_D(trainLabelArr[trainDataArr_DevFeature == i]): H(Di)
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size * calc_H_D(trainLabelArr[trainDataArr_DevFeature == i])

    #返回得出的条件经验熵
    return H_D_A

def calcBestFeature(trainDataList, trainLabelList):
    '''
    计算信息增益最大的特征
    :param trainDataList: 当前数据集
    :param trainLabelList: 当前标签集
    :return: 信息增益最大的特征和最大信息增益值
    '''

    # 数据集转为数组
    trainDataArr = np.array(trainDataList)

    # 标签集转为数组
    # 标签集需要转置
    # 借用 dodo 一个很好的例子:
    # a = np.array([1, 2, 3]); b = np.array([1, 2, 3]).T
    # a[0] = [1, 2, 3]; b[0] = 1, b[1] = 2
    # 为了调用每一个元素, 标签集需要转置
    trainLabelArr = np.array(trainLabelList).T

    # 特征数目
    # shape[0]: 样本数目; shape[1]: 特征数目
    featureNum = trainDataArr.shape[1]

    # 初始化最大信息增益
    maxG_D_A = -1

    # 初始化最大信息增益的特征
    maxFeature = -1

    # 对所有特征遍历计算
    for feature in range(featureNum):

        # 1. 计算数据集 D 的经验熵 H(D)
        H_D = calc_H_D(trainLabelArr)

        # 2. 提取列表中 feature 列的数据
        # trainDataArr[:, feature]: 所有样本第 feature 个特征那的一列数据
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)

        # 3. 计算条件经验熵 H(D|A)
        H_D_A = calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)

        # 4. 计算信息增益率: G(D,A) = H(D) - H(D|A)
        G_D_A = H_D - H_D_A

        # 5. 更新最大的信息增益与对应的特征
        if G_D_A > maxG_D_A:
            # 更新最大信息增益
            maxG_D_A = G_D_A

            # 更新对应特征
            maxFeature = feature

    # 返回最大信息增益的特征和最大信息增益
    return maxFeature, maxG_D_A

def getSubDataArr(trainDataArr, trainLabelArr, feature_index, feature_value):
    '''
    创建决策树时, 保留数据集中值为指定特征值的样本, 但是去除该特征值
    例如:60000 * 784 数据中, 有 45000 人买车, 有 15000 人不买车
    trainDataArr[i][feature_index]: 第 i 个样本, 特征为 “是否买车”
    feature_value: 买车
    在操作一次后, 返回的新样本矩阵变为: 45000 * 783
    :param trainDataArr: 未更新的数据集
    :param trainLabelArr: 未更新的标签集
    :param feature_index: 去除的特征索引
    :param feature_value: 新的数据集和标签集, data[feature_index] = feature_value 时, 该行样本需要保留
    :return:
    '''

    # 返回的数据集
    retDataArr = []

    # 返回的标签集
    retLabelArr = []

    # 对当前数据的每一个样本进行遍历
    for i in range(len(trainDataArr)):
        # 如果当前样本的特征为指定特征值 feature_value
        if trainDataArr[i][feature_index] == feature_value:
            # 那么将该样本的第 feature_index 个特征切割掉, 其他的放入返回的数据集中
            retDataArr.append(trainDataArr[i][0:feature_index] + trainDataArr[i][feature_index + 1:])

            # 将该样本的标签放入返回标签集中
            retLabelArr.append(trainLabelArr[i])

    # 返回新的数据集和标签集
    return retDataArr, retLabelArr

def createTree(*dataSet):
    '''
    递归创建决策树
    :param dataSet: 元组: (trainDataList, trainLabelList), 在递归中更加简便
    :return: 新的子节点
    '''

    # 信息增益阈值
    Epsilon = 0.1

    # 数据集
    trainDataList = dataSet[0][0]

    # 标签集
    trainLabelList = dataSet[0][1]

    # 开始创建: 打印当前特征向量数目与剩余样本数目
    print('START A NODE: ', len(trainDataList[0]), len(trainLabelList))

    # 标签字典, 与当前样本标签中的类别相同
    classDict = {i for i in trainLabelList}

    # 当 D 中所有样本均属于同一类 Ck, 无需再分割, 则 Ck 作为该结点类, 返回该叶子结点
    if len(classDict) == 1:
        # 仅需返回当前标签的类别, 由于从 0, 1, 2 ... 均为同一类, 故可以直接返回第 0 个
        return trainLabelList[0]

    # 若当前特征个数已经为 0, 没有可以用于分割的标签, 用于叶子结点
    # 例如:
    # n - 1 步: 根据是否买车划分, 此时特征仅剩这一个, 45000 人买车
    # n 步: 此时样本数据为 45000 * 0, 标签为 45000 * 1
    # 此时直接对标签进行最大类划分即可
    if len(trainDataList[0]) == 0:
        # 返回占大多数的类别
        return majorClass(trainLabelList)

    # 其他情况: 计算出样本数据中最大信息增益的特征 Ag
    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)

    # 如果 Ag 的信息增益小于阈值 Epsilon, 则置 T 为单节点树
    if EpsilonGet < Epsilon:
        # 将 D 中实例数最大的类 Ck 作为该节点的类, 返回 T
        return majorClass(trainLabelList)

    # 否则, 对 Ag 的每一可能值 ai, 依 Ag = ai 将 D 分割为若干非空子集 Di, 将 Di 中实例数最大的类作为标记, 构建子节点, 由节点及其子节点构成树 T, 返回T
    treeDict = {Ag: {}}

    # getSubDataArr(trainDataList, trainLabelList, Ag, 0): 在当前数据集中切割当前 feature, 返回新的数据集和标签集
    # 此处已经根据 128 划分 0 1 化
    # 特征值为 0 时, 进入 0 分支
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))

    # 特征值为 1 时, 进入 1 分支
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))

    # 返回树
    return treeDict

def predict(testDataList, tree):
    '''
    预测标签
    :param testDataList:样本
    :param tree: 决策树
    :return: 预测结果
    '''

    # 死循环, 直到找到一个有效的分类
    while True:
        # 因为有时候当前字典只有一个节点
        # 例如: {73: {0: {74: 6}}} 看起来节点很多, 但是对于字典的最顶层来说, 只有 73 一个 key, 其余都是 value
        # 若还是采用 for 来读取的话不太合适, 所以使用下行这种方式读取 key 和 value
        (key, value), = tree.items()

        # 如果当前的 value 是字典, 说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':

            # 获取目前所在节点的 feature 值, 需要在样本中删除该 feature
            dataVal = testDataList[key]

            # 因为在创建树的过程中, feature 的索引值永远是对于当时剩余的 feature 来设置的
            # 所以需要不断地删除已经用掉的特征, 保证索引相对位置的一致性
            del testDataList[key]

            # 将 tree 更新为其子节点的字典
            tree = value[dataVal]

            # 如果当前节点的子节点的值是 int, 就直接返回该 int 值
            # 例如: {403: {0: 7, 1: {297: 7}}, dataVal = 0
            # 此时上一行 tree = value[dataVal], 将 tree 定位到了 7, 而 7 不再是一个字典了
            # 这里就可以直接返回 7 了, 如果 tree = value[1], 那就是一个新的子节点, 需要继续遍历下去
            if type(tree).__name__ == 'int':
                # 返回该节点值，也就是分类值
                return tree

        # 如果当前value不是字典
        else:
            # 返回分类值
            return value

def test(testDataList, testLabelList, tree):
    '''
    测试准确率
    :param testDataList: 待测试数据集
    :param testLabelList: 待测试标签集
    :param tree: 训练集生成的树
    :return: 准确率
    '''

    # 错误次数计数
    errorCnt = 0

    # 遍历测试集中每一个测试样本
    for i in range(len(testDataList)):
        # 判断预测与标签中结果是否一致
        if testLabelList[i] != predict(testDataList[i], tree):
            # 不一致时错误计数加一
            errorCnt += 1

    #返回准确率
    return 1 - errorCnt / len(testDataList)

if __name__ == '__main__':
    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取训练数据
    trainDataList, trainLabelList = loadData("D:\\Mnist\\mnist_train.csv")

    # 获取测试数据
    testDataList, testLabelList = loadData("D:\\Mnist\\mnist_test.csv")

    # 创建决策树
    print('START TO CREATE TREE')

    # 决策树
    tree = createTree((trainDataList, trainLabelList))

    # 打印决策树
    print('Tree:', tree)

    # 测试准确率
    print('START TO TEST')

    # 返回测试率
    accur = test(testDataList, testLabelList, tree)

    # 显示测试率
    print('The accuracy is:', accur)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示用时时长
    print('Time Span:', endTime - startTime)