#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
This code was done by zhangyuan818
which is published in https://github.com/zhangyuan818/code-realization-in-Statistical-Learning-method
and my blog is https://blog.csdn.net/u013731486
everyone is welcome to contact me via 1453534820@qq.com
"""
import numpy as np


class DecisionTree:
    def cal_H_D(self, data):
        # data = np.array(data)
        classes = set(data)
        H_D = 0
        for i in classes:
            p = data[data == i].size / data.size
            H_D += -1 * p * np.log2(p)
        return H_D

    def gain(self, D, A, k=0):
        # D = np.array(D)
        # A = np.array(A)
        Aclasses = set(A)
        H_D_A = 0
        for i in Aclasses:
            H_D_A += A[A == i].size / A.size * self.cal_H_D(D[A == i])
        # k==1时表示信息增益率
        if k == 1:
            return 1 - H_D_A/self.cal_H_D(D)
        return self.cal_H_D(D) - H_D_A

    def selectFeature(self, data, A, k=0):
        data = np.array(data).T
        A = np.array(A)
        gains = [self.gain(item, A, k) for item in data]
        # print("gain:", gains)
        return np.argmax(np.array(gains)), max(gains)

    def mostInstance(self, label):
        if isinstance(label,list):
            return max(label, key=label.count)
        return max(label, key=label.tolist().count)

    def subset(self, data, label, I, ai):
        dataT = np.array(data).T
        label = np.array(label)
        temp = []
        for i in range(len(dataT)):
            temp.append(dataT[i][dataT[I] == ai])
        # temp = dataT[dataT[I] == ai]
        subdata = np.array(temp[:I]+temp[I+1:])
        sublabel = label[dataT[I] == ai]
        return subdata.T, sublabel

    def creat(self, data, label, k=0):
        epsilon = 0.01
        if len(set(label)) <= 1:
            return self.mostInstance(label)
        featureId, featureGain = self.selectFeature(data, label, k)
        # print("selected feature:",featureList[featureId])
        if featureGain < epsilon:
            return self.mostInstance(label)
        dataT = np.array(data).T
        tree = {}
        for ai in set(dataT[featureId]):
            subdata, sublabel = self.subset(data,label,featureId,ai)
            tree[str(featureId)+"|"+str(ai)] = self.creat(subdata,sublabel,k)
            # print(tree)
        return tree

    def predict(self, tree, x):
        if isinstance(tree, dict):
            currentFeatureStr = list(tree.keys())
            currentFeaturt = currentFeatureStr[0].split("|")[0]
            return self.predict(tree[str(currentFeaturt)+"|"+str(x[int(currentFeaturt)])], x)
        return tree


if __name__ == "__main__":
    # featureList = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 年龄，有工作，有自己的房子，信贷情况
    example = [[1,0,0,1],[1,0,0,2],[1,1,0,2],[1,1,1,1],[1,0,0,1],
               [2,0,0,1],[2,0,0,2],[2,1,1,2],[2,0,1,3],[2,0,1,3],
               [3,0,1,3],[3,0,1,2],[3,1,0,2],[3,1,0,3],[3,0,0,1]]
    labels = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
    # print("feature：", selectFeature(example, label))
    DT = DecisionTree()
    Dtree = DT.creat(example, labels, 0)
    print("Decision Tree:", Dtree)
    print("[2,1,1,3] is class ", DT.predict(Dtree, [2,1,1,3]))