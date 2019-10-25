#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
This code was done by zhangyuan818
which is published in https://github.com/zhangyuan818/Statistical_Learning_method
and my blog is https://blog.csdn.net/u013731486
everyone is welcome to contact me via zhangyuan18g@ict.ac.cn
"""
import numpy as np


class KdNode:
    def __init__(self, data, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild


class KdTree:
    def __init__(self):
        self.kdTree = None
        self.nearest = []
        self.nearestDis = []

    def create(self, dataSet, depth=0):
        if len(dataSet) > 0:
            m, n = np.shape(dataSet)
            midIndex = m // 2
            axis = depth % n
            sortedDataSet = self.sort(dataSet, axis)
            node = KdNode(sortedDataSet[midIndex])
            leftDataSet = sortedDataSet[: midIndex]
            rightDataSet = sortedDataSet[midIndex+1:]
            node.lchild = self.create(leftDataSet, depth+1)
            node.rchild = self.create(rightDataSet, depth+1)
            return node
        else:
            return None

    def sort(self, dataSet, axis):
        sortDataSet = dataSet[:]
        m, n = np.shape(sortDataSet)
        for i in range(m):
            for j in range(0, m-i-1):
                if sortDataSet[j][axis] > sortDataSet[j+1][axis]:
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j+1]
                    sortDataSet[j+1] = temp
        return sortDataSet

    def preOrder(self, node):
        if node is not None:
            print("tree node->%s" % node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    # 用kd树的k近邻搜索算法,k缺省为1
    def search(self, node, aim, k=1, depth=0):
        if node is not None:
            n = len(aim)  # 求维度
            axis = depth % n
            if aim[axis] < node.data[axis]:
                self.search(node.lchild, aim, k, depth+1)
            else:
                self.search(node.rchild, aim, k, depth+1)

            dis = self.dist(aim, node.data)  # 欧式距离
            if len(self.nearest) < k:
                self.nearest.append(node.data)
                self.nearestDis.append(dis)
            elif max(self.nearestDis) > dis:
                maxIndex = self.nearestDis.index(max(self.nearestDis))
                self.nearest[maxIndex] = node.data
                self.nearestDis[maxIndex] = dis

            # 算法3.3(3)(b)判断是否需要去另一子结点搜索
            if abs(node.data[axis] - aim[axis]) <= max(self.nearestDis):
                if aim[axis] < node.data[axis]:
                    self.search(node.rchild, aim, k, depth + 1)
                else:
                    self.search(node.lchild, aim, k, depth + 1)

    def dist(self, x1, x2):
        return (np.sum((np.array(x1)-np.array(x2))**2))**0.5

    
if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    x = [5, 3]
    # data = [[2,4], [6,1.5], [4,7], [8,2.5], [7.5,4.5],[1,1],[1.5,7]]
    # x = [6,3]
    kdtree = KdTree()
    tree = kdtree.create(data)
    # kdtree.preOrder(tree)
    kdtree.search(tree, x, 3)
    print(kdtree.nearest)
    print(kdtree.nearestDis)
