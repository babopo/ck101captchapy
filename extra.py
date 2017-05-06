# -*- coding: utf-8 -*-
import numpy as np
import xlrd
from xlutils.copy import copy


# 清空excel工作表中的数据
def xls_clear(filename):
    book = xlrd.open_workbook(filename)
    sheets = book.sheet_by_index(0)
    rows = sheets.nrows
    cols = sheets.ncols
    xls_w = copy(book)
    sh_r = xls_w.get_sheet(0)  # 只有get_sheet有write方法
    for i in range(rows):
        for j in range(cols):
            sh_r.write(i, j, '')
    xls_w.save(filename)


# 向excel文件添加一行数据, values是一个二维数组且为float类型
def xls_append(filename, values):
    book = xlrd.open_workbook(filename)
    sheets = book.sheet_by_index(0)
    rows = sheets.nrows
    xls_w = copy(book)
    sh_r = xls_w.get_sheet(0)
    for i in range(0, len(values[0])):
        sh_r.write(rows, i, values[0][i])
    xls_w.save(filename)


# 将字符图片归一化至28*28，邻近插值，返回处理后的数组
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


# 对字符分类
def sort(argument):
    switcher = {
        1: "2",
        2: "3",
        3: "4",
        4: "6",
        5: "7",
        6: "8",
        7: "9",
        8: "B",
        9: "C",
        10: "E",
        11: "F",
        12: "G",
        13: "H",
        14: "J",
        15: "K",
        16: "M",
        17: "P",
        18: "Q",
        19: "R",
        20: "T",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
    }
    return switcher.get(argument, "")
