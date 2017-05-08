# xls test

import os
import xlrd
import xlwt
from xlutils.copy import copy
import extra
import numpy as np
import train
import cv2
import math


def filename(file):  # 读取目录下特定顺序文件
    path = ".\\temp\\"
    all_files = os.listdir(path)
    return all_files[file]
# values = [1, 2, 3, 4]
# book = xlrd.open_workbook("feature.xls")
# br = copy(book)
# a = book.sheet_by_index(0)
# sh = br.get_sheet(0)  # sheet_by_index没有write方法
# rows = a.nrows
#
# for i in range(0, len(values)):
#     sh.write(rows, i, values[i])
#
# br.save("feature.xls")
# #extra.xls_clear("feature.xls")
#
#
# inx = np.array([100,3,4,5])
# data = np.array([[2,3,4,5],[5,6,7,8]])
# labels = ['a','b']
# print(extra.knn(inx,data,labels,1))


# book = xlrd.open_workbook('feature.xls')
# sheets = book.sheet_by_index(0)
# print(sheets.cell_value(0,0))
#
# path = 'C:\\Users\Limbo\Desktop\毕设\验证码\卡提诺\CharactorSrc\\'
# for i in range(len(os.listdir(path))):
#     print(os.listdir(path)[i])
#
# m,n = train.samples()
#
#
# a = 0
# extra.xls_clear("feature.xls")
num = len(os.listdir(".\\temp\\"))  # 由实验得分离出的字符数不一定为4个
fea = np.zeros([num, 16]).astype(int)
for i in range(num):
    temp = os.path.join(".\\temp\\", filename(i))
    I = cv2.imread(temp, 0)  # 单个字符图像，单通道读取
    m_s, n_s = I.shape  # 提取4*4粗网格特征，统计每个网格黑点数
    numB = np.zeros([4, 4])
    for row in range(m_s):
        for col in range(n_s):
            rowB = math.floor(row / 7)
            colB = math.floor(col / 7)
            if I[row][col] == 0:
                numB[rowB][colB] = numB[rowB][colB] + 1
    feature = numB.reshape(1, 16)  # 16位向量作为特征储存
    # extra.xls_append("feature.xls", fea)
    fea[i] = feature[0]
#################

# 字符识别

#################
dataX, dataY = train.samples()
dataY = extra.b2d(dataY)
Kn = 3  # knn中的k值
for i in range(num):
    resultlabel = extra.knn(fea[i], dataX, dataY, Kn)
    res = extra.sort(resultlabel)
    print(res)
