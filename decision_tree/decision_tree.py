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
    正确率: 85.89%
    运行时长: 494.27s
'''