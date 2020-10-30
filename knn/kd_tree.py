# coding = utf-8
# Author: Wiger
# Date: 2020-10-30
# Email: wiger@mail.ustc.edu.cn

'''
kd 树: 用于 KNN 算法, 快速获得距离当前样本最近的 topK 个样本
'''

import numpy as np
import time

class Node(object):
    '''
    结点类
    '''

    def __init__(self, item = None, label = None, dim = None, parent = None, left_child = None, right_child = None):
        '''
        结点初始化
        :param item: 结点数据(样本信息)
        :param label: 结点标签
        :param dim: 结点切分的维度(特征)
        :param parent: 父结点
        :param left_child: 左子树
        :param right_child: 右子树
        '''

        self.item = item
        self.label = label
        self.dim = dim
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

class KDTree(object):
    '''
    kd 树类
    '''

    def __init__(self, dataMat, labelMat):
        '''
        kd 树初始化
        :param dataMat: 数据矩阵
        :param labelMat: 标签矩阵
        '''

        # 长度不可手动修改
        self.__length = 0

        # 根节点, 私有属性, 不可手动修改
        self.__root = self.__create(dataMat, labelMat)

    def __create(self, dataMat, labelMat, parentNode = None):
        '''
        创建 kd 树
        :param dataMat: 数据矩阵: 行数代表样本数, 列数表示特征数
        :param labelMat: 标签矩阵
        :param parentNode: 父结点
        :return: 根节点
        '''

        # 数组尺寸: m: 样本数 n: 特征数
        m, n = dataMat.shape

        # 样本集为空
        if m == 0:
            return None

        # 根节点的切分超平面选择: 特征方差最大
        # 计算所有特征方差: n * 1
        varList = [np.var(dataMat[:, col]) for col in range(n)]

        # 最大特征方差索引
        maxIndex = varList.index(max(varList))

        # 该特征值升序排序: m * 1
        sortedVarList = np.argsort(dataMat[:, maxIndex])

        # 选取该特征中位数样本点: 1 * 1
        midItemIndex = sortedVarList[m // 2]

        # 样本数目为 1
        if m == 1:

            # 数据总数 + 1
            self.__length += 1

            # 返回 midItemIndex 的数据, 标签和特征, 父结点和左右子树均为空
            return Node(item = dataMat[midItemIndex], label= labelMat[midItemIndex], dim = maxIndex, parent = parentNode, left_child = None, right_child = None)

        # 样本数目 > 1, 生成结点
        node = Node(item = [midItemIndex], label= labelMat[midItemIndex], dim = maxIndex, parent = parentNode, left_child = None, right_child = None)

        # 左子树数据: (m // 2) * n
        # 该特征量上, 值小于等于该结点的样品 index 列表
        left_data = dataMat[sortedVarList[:m // 2]]

        # 左子树标签: (m // 2) * n
        left_label = labelMat[sortedVarList[:m // 2]]

        # 左子树结点
        left_child = self.__create(left_data, left_label, node)

        # 仅有左子树, 无右子树
        if m == 2:

            # 右子树为空
            right_child = None

        # 右子树
        else:

            # 右子树数据: (m // 2) * n
            # 该特征量上, 值大于等于该结点的样品 index 列表
            right_data = dataMat[sortedVarList[m // 2 + 1:]]

            # 右子树标签: (m // 2) * n
            right_label = labelMat[sortedVarList[m // 2 + 1:]]

            # 右子树结点
            right_child = self.__create(right_data, right_label, node)

        # 设置结点左子树
        node.left_child = left_child

        # 设置结点右子树
        node.right_child = right_child

        # 树结点数目 + 1
        self.__length += 1

        # 返回结点
        return node

    # 方法作为属性调用

    # 树的结点个数
    @property
    def length(self):
        return self.__length

    # 树的根节点
    @property
    def root(self):
        return self.__root

    def find_nearest_neighbour(self, item):
        '''
        寻找最近邻点
        :param item: 预测样本向量
        :return: 距离最近的样本向量
        '''

        print(item)