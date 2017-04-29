# just for test
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filename(file):  # 读取目录下特定顺序文件
    path = ".\\temp\\"
    all_files = os.listdir(path)
    return all_files[file]

for i in range(0, len(os.listdir(".\\temp\\"))):
    temp = os.path.join(".\\temp\\", filename(i))
    I = cv2.imread(temp, 0)
    windowname = str(i)
    cv2.imshow(windowname, I)

img = cv2.imread('1.png')
# cv2.imshow('origin image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(img)
plt.show()

img = np.array(img)
if img.ndim == 3:
    img = img[:, :, 0]
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img, cmap ='gray')
# plt.subplot(223)
# plt.imshow(img, cmap = plt.cm.gray)
# plt.subplot(224)
# plt.imshow(img, cmap = plt.cm.gray_r)
# plt.show()

cv2.imshow('origin image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()