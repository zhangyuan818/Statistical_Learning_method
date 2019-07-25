#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
This code was done by zhangyuan818
which is published in https://github.com/zhangyuan818/code-realization-in-Statistical-Learning-method
and my blog is https://blog.csdn.net/u013731486
everyone is welcome to contact me via 1453534820@qq.com
"""

import numpy as np
# from collections import Counter


class Bayes:
    def __init__(self, data, label, lamda=0):
        self.priorP = {}
        self.condP = {}
        self.rangeOfFeature = {}  # 保存每个特征的取值个数
        self.train(data, label, lamda)

    # 给一个向量，返回字典，包含不同元素的比例。可以引用collections中的Counter函数来实现
    # 这个函数可以改进，懒得弄了
    def count(self, vec, classNum, lamda=0):
        keys = set(vec)
        p = {}
        for key in keys:
            n = np.sum(np.isin(vec, key) + 0)
            p[key] = (n+lamda)/(len(vec)+classNum*lamda)  # 直接计算频率
        return p

    def train(self, data, label, lamda=0):
        m, n = np.shape(data)
        for j in range(n):
            self.rangeOfFeature[j] = len(set([x[j] for x in data]))
        classes = set(label)
        # 计算先验概率
        self.priorP = self.count(label, len(classes), lamda)
        print("priorP:", self.priorP)
        # 计算条件概率
        for c in classes:
            subset = [data[i] for i in range(m) if label[i] == c]
            for j in range(n):
                self.condP[str(c)+" "+str(j)] = self.count([x[j] for x in subset], self.rangeOfFeature[j], lamda)
        print("condP:", self.condP)

    def predict(self, x):
        preP = {}
        for c in self.priorP.keys():
            preP[c] = self.priorP[c]
            for i, features in enumerate(x):
                preP[c] *= self.condP[str(c)+" "+str(i)][features]
        print("probability: ", preP)
        print("prediction: ", max(preP, key=preP.get))


if __name__ == "__main__":
    dataSet = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
               [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
               [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    bayes = Bayes(dataSet, labels, 1)
    bayes.predict([2, "S"])
