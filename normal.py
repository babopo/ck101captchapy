# -*- coding: utf-8 -*-
# 将字符图片归一化至28*28，邻近插值，返回处理后的数组
import numpy as np

def normal(I):
    w_n = 28
    h_n = 28
    m, n = I.shape
    w = w_n / n
    h = h_n / m
    imgn = np.zeros([28, 28]).astype(int)
    rot = np.array([[h, 0, 0], [0, w, 0], [0, 0, 1]])  # 变换矩阵 x = h*u, y = w*v
    rotI = np.mat(rot).I  # rot矩阵求逆
    for row in range(h_n):
        for col in range(w_n):
            pix = np.mat([row, col, 1])
            pix = pix * rotI
            pix = np.array(pix).astype(int)  # 矩阵索引不太方便，转为数组
            if pix[0][0] < 1:
                pix[0][0] = 1
            if pix[0][1] < 1:
                pix[0][1] = 1
            imgn[row][col] = I[round(pix[0][0])][round(pix[0][1])]
    return imgn