# coding = utf-8
# @Time : 2022/1/7 21:39
# @Author : hky
# @File : t.py
# @Software : PyCharm

import numpy as np
import struct
import matplotlib.pyplot as plt
import aes as AES
from SCAUtil import SBOX, HW, correlation


def pearson(X: np.ndarray, Y: np.ndarray):
    if X.shape[0] != Y.shape[0]:
        print("X and Y have wrong dimension")
        return
    # X: N*1, Y:N*M
    mean_X = X.mean(axis=0)
    # print(mean_X)
    mean_Y = Y.mean(axis=0)  # mean of Y by column
    # print(mean_Y)

    XX = (X - mean_X).reshape(X.shape[0])
    YY = Y - mean_Y
    # print(XX)
    # print(YY)

    r_X = np.sqrt(np.sum(np.square(XX), axis=0))
    print(r_X.shape)
    r_Y = np.sqrt(np.sum(np.square(YY), axis=0))

    print(r_Y.shape)


    print(YY.T.shape)
    print(XX.shape)
    sum_XY = np.matmul(YY.T, XX)
    print(sum_XY.shape)
    print((r_X * r_Y).shape)
    r = sum_XY / (r_X * r_Y)
    # print(r)
    return r

A = np.array([[1], [2], [3]])
B = np.array([[1, 2, 2, 4], [3, 4, 4, 4], [5, 6, 6, 6]])
print(A.shape)
print(B.shape)
pearson(A, B)
# print(s1_corr_rank)

