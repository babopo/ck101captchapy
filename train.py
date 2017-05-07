# -*- coding: utf-8 -*-
# 读取用于模式识别的单字符样本数据
import xlrd
import numpy as np
import os


# 返回所有样本特征rawdata及对应labels
def samples():
    path = 'C:\\Users\Limbo\Desktop\毕设\验证码\卡提诺\CharactorSrc\\'
    for d in range(len(os.listdir(path))):
        filename = os.path.join(path, os.listdir(path)[d], 'train.xls')
        book = xlrd.open_workbook(filename)
        sheets = book.sheet_by_index(0)
        rows = sheets.nrows
        cols = sheets.ncols
        sam = np.zeros([rows, 16])
        lab = np.zeros([rows, 5])  # 样本中的标签用五位二进制储存
        for i in range(rows):
            for j in range(0, 16):
                sam[i][j] = sheets.cell_value(i, j)
        for i in range(rows):
            for j in range(16, cols):
                lab[i][j-16] = sheets.cell_value(i, j)
        if not d:
            rawdata = sam
            labels = lab
        else:
            rawdata = np.append(rawdata, sam, axis=0)
            labels = np.append(labels, lab, axis=0)
    return rawdata, labels