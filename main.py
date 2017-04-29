# -*- coding: utf-8-*-

import os
import xlrd, xlwt
import math
import numpy as np
import cv2
from scipy import misc
from skimage import morphology
from matplotlib import pyplot as plt


def normal(I):  # 将字符图片归一化至28*28，邻近插值，返回处理后的数组
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


def filename(file):  # 读取目录下特定顺序文件
    path = ".\\temp\\"
    all_files = os.listdir(path)
    return all_files[file]


img = cv2.imread('0.png')

cv2.imshow('origin image', img)

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # 转LAB空间
m, n, z = lab_img.shape

# 转为kmeans算法的数据样本
samples = lab_img.reshape(-1, 3)
samples = np.float32(samples)

cv2.imshow('lab image', lab_img)

# kmeans
K = 45  # 簇数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 迭代终止条件，满足迭代10次或精度为1.0则停止
flags = cv2.KMEANS_RANDOM_CENTERS  # 随机初始中心
compactness, labels, centers = cv2.kmeans(samples, K, None, criteria, 10, flags)
K_img = labels.reshape(m, n)

for i in range(K):
    LAB_BW = np.zeros([m, n]).astype(int)  # 类图 防止浅拷贝
    for row in range(m):
        for col in range(n):
            if K_img[row][col] == i:
                LAB_BW[row][col] = 1  # 反色图方便连通域标记去噪，因为0表示亮度最低
    windowname = str(i)
    # cv2.imshow(windowname, LAB_BW)
    LAB_BW = morphology.remove_small_objects(LAB_BW.astype(bool), min_size=70, connectivity=1, in_place=False)  # 4连通域去噪
    Z = 0  # 统计类图中黑点数
    for row in range(m):
        for col in range(n):
            Z = Z + LAB_BW[row][col]
    ver_num = np.zeros(n).astype(int)  # 竖直方向投影黑点数
    for row in range(m):
        for col in range(n):
            ver_num[col] = ver_num[col] + LAB_BW[row][col]
    width = np.where(ver_num > 0)
    lel_num = np.zeros(m).astype(int)  # 水平方向投影黑点数
    for row in range(m):
        for col in range(n):
            lel_num[row] = lel_num[row] + LAB_BW[row][col]
    high = np.where(lel_num > 0)
    for row in range(m):
        for col in range(n):
            LAB_BW[row][col] = 1 - LAB_BW[row][col]  # 反色为正常颜色
    if width[0].any() and high[0].any():  # 类图非空
        w = width[0][-1] - width[0][0]
        h = high[0][-1] - high[0][0]
        if w < 25 and h < 28 and w > 7 and h > 8 and Z < 500 and Z > 70:  # 像素点数和长宽是否符合要求
            # plt.imshow(LAB_BW, cmap=plt.cm.gray_r)
            # plt.show()
            cut = np.zeros([h, w])  # 将字符从类图中分割
            for row in range(h):
                for col in range(w):
                    cut[row][col] = LAB_BW[high[0][0] + row][width[0][0] + col]
            pos = width[0][0] / 120
            cut_n = normal(cut).astype(float)  # 转为float类型以便opencv显示
            cv2.imshow(windowname, cut_n)
            savename = ".\\temp\\" + str(pos) + ".png"  # 相对路径，方便修改
            misc.imsave(savename, cut_n)  # 将array保存为图像
for i in range(0, len(os.listdir(".\\temp\\"))):
    temp = os.path.join(".\\temp\\", filename(i))
    I = cv2.imread(temp, 0)  # 单个字符图像，单通道读取
    m_s, n_s = I.shape  # 提取4*4粗网格特征，统计每个网格黑点数
    numB = np.zeros([4, 4]).astype(int)
    for row in range(m_s):
        for col in range(n_s):
            rowB = math.floor(row / 7)
            colB = math.floor(col / 7)
            if I[row][col] == 0:
                numB[rowB][colB] = numB[rowB][colB] + 1
    fea = numB.reshape(1, 16)  # 16位向量作为特征储存







cv2.waitKey(0)
cv2.destroyAllWindows()
