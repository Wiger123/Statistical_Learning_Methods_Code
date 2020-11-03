# coding = utf-8
# Author: Wiger
# Date: 2020-10-30
# Email: wiger@mail.ustc.edu.cn

'''
kd 树: 用于 KNN 算法, 快速获得距离当前样本最近的 topK 个样本
'''

import numpy as np
import time


def calcDist(x1, x2):
    '''
    计算两样本向量之间距离
    :param x1: 向量一
    :param x2: 向量二
    :return: 向量之间距离
    '''

    # 欧式距离
    return np.sqrt(np.sum(np.square(x1 - x2)))

    # 曼哈顿距离
    # return np.sum(np.abs(x1 - x2))


def calcOutDist(x1, x2):
    '''
    计算测试点 x1 与样本点 x2 分割线的距离
    :param x1: 测试点
    :param x2: 样本点
    :return: 测试点到分割线的距离
    '''
    return abs(x1[x2.dim] - x2.item[x2.dim])


def search_node_region(root, item):
    '''
    寻找当前结点在 kd 树中位于哪个叶子结点的区域中
    :param root: 开始搜索的起始根结点
    :param item: 需要预测的样本向量
    :return: 所在区域的的叶子结点
    '''

    # item 需要转为数组
    itemArray = np.array(item)

    # 空树
    if root == None:
        return None

    # 递归找距离测试点最近叶结点
    node = root

    # 递归搜索该样本所在的叶结点区域
    while True:
        # 仅一个样本, 无左右子树
        if node.left_child == None and node.right_child == None:
            return node

        # 当前结点特征
        cur_dim = node.dim

        # 样本向量的该特征值 < 结点的该特征值
        if item[cur_dim] <= node.item[cur_dim]:

            # 若左子树为空
            if node.left_child == None:
                # 返回自身结点
                return node

            # 若左子树非空, 进入左子树
            node = node.left_child

        # 样本向量的该特征值 >= 结点的该特征值, 当特征值相等时, 由于未遍历到叶子结点, 仍需要继续向下遍历
        else:

            # 若右子树为空
            if node.right_child == None:
                # 返回自身结点
                return node

            # 若右子树非空, 进入右子树
            node = node.right_child


def search_k_node(tree_root, root, item, k, node_list, node_visited):
    '''
    当树的总结点数多于 k 个, 寻找前 k 个样本
    :param tree_root: 树的根结点
    :param root: 搜索起始根结点
    :param item: 测试样本
    :param k: 距离最近的样本个数
    :param node_list: 距离最近的 k 个样本列表
    :param node_visited: 访问过的 结点集合
    :return: 返回距离样本最近的前 k 个样本标签
    '''

    # 测试样本转为数组类型
    item = np.array(item)

    # 找到最近叶子结点
    leaf_node = search_node_region(root, item)

    # 加入访问过的集合
    node_visited.add(leaf_node)

    # 测试点与叶子结点之间距离
    dist_now = calcDist(item, leaf_node.item)

    # 判断结点列表是否已满, 在列表数目未到 k 个的状态下, 需要添加当前结点
    if len(node_list) < k:

        # 结点列表添加父结点
        node_list.append([dist_now, tuple(leaf_node.item), leaf_node.label])

        # 结点列表按照距离排序
        node_list.sort()

    # 列表已经达到 k 个点
    else:
        # 父结点与测试点距离小于列表里最长距离时
        if dist_now < node_list[-1][0]:
            # 列表去掉最长距离
            node_list.pop()

            # 添加到结点列表
            node_list.append([dist_now, tuple(leaf_node.item), leaf_node.label])

            # 结点列表按照距离排序
            node_list.sort()

    # 不断进行回溯
    while True:
        # 不断向上爬层, 直到结点未被访问过
        while True:
            # 已经到达 kd 树根结点
            if leaf_node == tree_root:
                # 返回结点列表
                return node_list

            # 向上爬一层
            leaf_node = leaf_node.parent

            # 父结点未被访问过
            if leaf_node not in node_visited:
                # 加入访问集合
                node_visited.add(leaf_node)

                # 跳出循环
                break

        # 测试点与叶子结点之间距离
        dist_now = calcDist(item, leaf_node.item)

        # 若此时结点列表未达到 k 个
        if len(node_list) < k:
            # 添加到结点列表
            node_list.append([dist_now, tuple(leaf_node.item), leaf_node.label])

            # 结点列表按照距离排序
            node_list.sort()

        # 结点列表已达到 k 个( 虽然不会出现 > k 的情况, 但是防止极其特殊的 bug, 还是加上大于号)
        # 且结点到测试点距离大于列表中最大值
        elif len(node_list) >= k and dist_now <= node_list[-1][0]:
            # 列表去掉最长距离
            node_list.pop()

            # 添加到结点列表
            node_list.append([dist_now, tuple(leaf_node.item), leaf_node.label])

            # 结点列表按照距离排序
            node_list.sort()

        # 该结点分割线到样本点的距离
        dist_out = calcOutDist(item, leaf_node)

        # 该距离大于等于结点列表中最远的距离并且列表中已有 k 个点, 则在切分线另一边不会有更近的点, 返回不断进行回溯
        if len(node_list) >= k and dist_out >= node_list[-1][0]:
            # 返回不断进行回溯
            continue

        # 该距离小于列表中最远的距离, 或列表中不足 k 个点, 则切分线另一边可能有更近的点
        # 因此在当前结点的另一个兄弟结点, 开始执行新的递归
        elif len(node_list) < k or dist_out < node_list[-1][0]:
            # 若该点从左结点回溯上来
            if leaf_node.left_child in node_visited:
                # 若右节点不存在
                if leaf_node.right_child == None:
                    # 跳过本次循环, 向上爬一层
                    continue

                # 则另一子树为右结点
                another_child = leaf_node.right_child

            # 若该点从右结点回溯上来
            else:
                # 若左节点不存在
                if leaf_node.left_child == None:
                    # 跳过本次循环, 向上爬一层
                    continue

                # 则另一子树为左结点
                another_child = leaf_node.left_child

            # 兄弟结点递归调用寻找
            search_k_node(tree_root, another_child, item, k, node_list, node_visited)

class Node(object):
    '''
    结点类
    '''

    def __init__(self, item=None, label=None, dim=None, parent=None, left_child=None, right_child=None):
        '''
        结点初始化
        :param item: 结点数据(样本信息)
        :param label: 结点标签
        :param dim: 特征
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

    def __init__(self, dataArr, labelArr):
        '''
        kd 树初始化
        :param dataArr: 数据数组
        :param labelArr: 标签数组
        '''

        # 长度不可手动修改
        self.__length = 0

        # 根结点, 私有属性, 不可手动修改
        self.__root = self.__create(dataArr, labelArr)

    def __create(self, dataArr, labelArr, parentNode=None):
        '''
        创建 kd 树
        :param dataArr: 数据数组: 行数代表样本数, 列数表示特征数
        :param labelArr: 标签数组
        :param parentNode: 父结点
        :return: 根结点
        '''

        # 数据数组
        dataArray = np.array(dataArr)

        # 数组尺寸: m: 样本数 n: 特征数
        m, n = dataArray.shape

        # 标签数组
        labelArray = np.array(labelArr).reshape(m, 1)

        # 样本集为空
        if m == 0:
            return None

        # 根结点的切分超平面选择: 特征方差最大
        # 计算所有特征方差: n * 1
        varList = [np.var(dataArray[:, col]) for col in range(n)]

        # 最大特征方差索引
        maxIndex = varList.index(max(varList))

        # 该特征值升序排序: m * 1
        sortedVarList = dataArray[:, maxIndex].argsort()

        # 选取该特征中位数样本点: 1 * 1
        # 转为 int 型
        midItemIndex = int(sortedVarList[m // 2])

        # 样本数目为 1
        if m == 1:
            # 数据总数 + 1
            self.__length += 1

            # 返回 midItemIndex 的数据, 标签和特征, 父结点和左右子树均为空
            return Node(item=dataArray[midItemIndex], label=int(labelArray[midItemIndex]), dim=maxIndex,
                        parent=parentNode, left_child=None, right_child=None)

        # 样本数目 > 1, 生成结点
        node = Node(item=dataArray[midItemIndex], label=int(labelArray[midItemIndex]), dim=maxIndex, parent=parentNode,
                    left_child=None, right_child=None)

        # 左子树数据: (m // 2) * n
        # 该特征量上, 值小于等于该结点的样品 index 列表
        left_data = dataArray[sortedVarList[:m // 2]]

        # 左子树标签: (m // 2) * n
        left_label = labelArray[sortedVarList[:m // 2]]

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
            right_data = dataArray[sortedVarList[m // 2 + 1:]]

            # 右子树标签: (m // 2) * n
            right_label = labelArray[sortedVarList[m // 2 + 1:]]

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

    # 树的根结点
    @property
    def root(self):
        return self.__root

    def transfer_list(self, node, kdList=[]):
        '''
        通过遍历 kd 树, 获取所有结点, 转化为列表嵌套字典
        :param node: 传入的根结点
        :param kdList: 返回嵌套字典的列表
        :return:
        '''

        # 树为空
        if node == None:
            # 返回空列表
            return None

        # 列表新的一项
        element_dict = {}

        # 结点向量
        element_dict['item'] = tuple(node.item)

        # 结点标签
        element_dict['label'] = node.label

        # 结点特征
        element_dict['dim'] = node.dim

        # 父结点, 没有时传回 None
        element_dict['parent'] = tuple(node.parent.item) if node.parent else None

        # 左子结点, 没有时传回 None
        element_dict['left_child'] = tuple(node.left_child.item) if node.left_child else None

        # 右子结点, 没有时传回 None
        element_dict['right_child'] = tuple(node.right_child.item) if node.right_child else None

        # 列表添加新项
        kdList.append(element_dict)

        # 左子树递归
        self.transfer_list(node.left_child, kdList)

        # 右子树递归
        self.transfer_list(node.right_child, kdList)

        # 返回列表
        return kdList

    def get_brother_node(self, node):
        '''
        获取兄弟结点
        :param node: 当前结点
        :return: 当前结点兄弟结点
        '''

        # 根结点无兄弟结点
        if node == self.root:
            # 返回空
            return None

        # 父结点
        parent_node = node.parent

        # 该结点是父结点的左子树
        if node == parent_node.left_child:

            # 返回右子树
            return parent_node.right_child

        # 该结点是父结点的右子树
        else:

            # 返回左子树
            return parent_node.left_child

    def search_nearest_node(self, item, k):
        '''
        寻找距离测试样本距离最近的前 k 个样本
        :param item: 测试样本
        :param k: 距离最近的样本个数
        :return: 返回距离样本最近的前 k 个样本标签
        '''

        # 树的总结点数少于等于 k 个
        if self.length <= k:

            # 标签集合
            label_dict = {}

            # 获取所有结点列表
            kdList = self.transfer_list(self.root)

            # 获取所有标签
            for element in kdList:

                # 当前标签已存在时
                if element['label'] in label_dict:

                    # 标签数目加一
                    label_dict[element['label']] += 1

                # 当前标签不存在时
                else:

                    # 创建该标签
                    label_dict[element['label']] = 1

            # 根据标签数目排序
            sorted_label = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)

            # 返回标签列表
            return sorted_label[0][0]

        # 树的总结点数多于 k 个
        else:
            # 最近的 k 个结点列表
            node_list = []

            # 访问过的结点集合
            node_visited = set()

            # 迭代寻找 k 个最近结点
            return search_k_node(self.root, self.root, item, k, node_list, node_visited)

# 主函数
if __name__ == '__main__':
    # 测试数据
    dataArray = np.array([[19, 2], [7, 0], [13, 5], [3, 15], [3, 4], [3, 2], [8, 9], [9, 3], [17, 15], [11, 11]])

    # 测试标签
    labelArray = np.array([[0], [2], [0], [5], [1], [5], [0], [1], [1], [1]])

    # 获取当前时间
    Time1 = time.time()

    # 构造 kd 树
    kd_tree = KDTree(dataArray, labelArray)

    # 获取当前时间
    Time2 = time.time()

    # 最近的 k 个元素的投票结果
    k_nodes = kd_tree.search_nearest_node([12, 7], 3)
    print(k_nodes)
    # 获取当前时间
    Time3 = time.time()

    # 显示用时时长
    print('Create kd-Tree:', Time2 - Time1)

    # 显示用时时长
    print('Search kd-Tree:', Time3 - Time2)
