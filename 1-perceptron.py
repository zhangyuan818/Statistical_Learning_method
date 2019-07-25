#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# train = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
train = [[(0, 0), 1], [(0, 1), 1], [(1, 0), -1], [(1, 1), -1]]
w = [0, 0]
b = 0


def update(data):
    global w, b
    for i in range(len(data[0])):
        w[i] = w[i] + 1 * data[1] * data[0][i]
    b = b + 1 * data[1]
    # print(w, b)


def cal(data):
    global w, b
    res = 0
    for i in range(len(data[0])):
        res += data[0][i] * w[i]
    res += b
    res *= data[1]
    return res


def check():
    flag = False
    for data in train:
        if cal(data) <= 0:
            flag = True
            update(data)
    if not flag:
        print("w: " + str(w) + " b: " + str(b))
        return True
    return False


for times in range(1000):
    if check():
        break

