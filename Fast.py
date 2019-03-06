# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
# 自己实现FAST角点检测算法:不依赖OpenCV库


def rgb2gray(image):
    rows, cols = image.shape[:2]
    grayscale = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            red, green, blue = image[row][col]
            gray = int(0.3 * red + 0.59 * green + 0.11 * blue)
            grayscale[row][col] = gray
    return grayscale


def bgr2gray(image):
    # 转换图片空间BGR->gray
    rows, cols = image.shape[:2]
    grayscale = image.copy()
    for row in range(0, rows):
        for col in range(0, cols):
            blue, green, red = image[row][col]
            gray = int(0.3 * red + 0.59 * green + 0.11 * blue)
            grayscale[row][col] = gray
    return grayscale


def medianBlur(image, ksize=3, ):
    '''
    中值滤波，去除椒盐噪声
    输入:
        image：输入图片数据,要求为灰度图片
        ksize：滤波窗口大小
    返回：
        中值滤波之后的图片
    '''
    rows, cols = image.shape[:2]
    half = ksize // 2
    startSearchRow = half
    endSearchRow = rows - half - 1
    startSearchCol = half
    endSearchCol = cols - half - 1
    dst = np.zeros((rows, cols), dtype=np.uint8)
    # 中值滤波
    for y in range(startSearchRow, endSearchRow):
        for x in range(startSearchCol, endSearchCol):
            window = []
            for i in range(y - half, y + half + 1):
                for j in range(x - half, x + half + 1):
                    window.append(image[i][j])
            window = np.sort(window, axis=None)
            if len(window) % 2 == 1:
                medianValue = window[len(window) // 2]  # 取中间值
            else:
                medianValue = int((window[len(window) // 2] + window[len(window) // 2 + 1]) / 2)
            dst[y][x] = medianValue
    return dst


def circle(row, col):
    '''
    从像素点位置(row,col)获取其邻域圆上16个像素点坐标，圆由16个像素点组成
    输入:
        row：行坐标 注意row要大于等于3
        col：列坐标 注意col要大于等于3
    '''
    if row < 3 or col < 3:
        return
    point1 = (row - 3, col)
    point2 = (row - 3, col + 1)
    point3 = (row - 2, col + 2)
    point4 = (row - 1, col + 3)
    point5 = (row, col + 3)
    point6 = (row + 1, col + 3)
    point7 = (row + 2, col + 2)
    point8 = (row + 3, col + 1)
    point9 = (row + 3, col)
    point10 = (row + 3, col - 1)
    point11 = (row + 2, col - 2)
    point12 = (row + 1, col - 3)
    point13 = (row, col - 3)
    point14 = (row - 1, col - 3)
    point15 = (row - 2, col - 2)
    point16 = (row - 3, col - 1)
    return [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12, point13,
            point14, point15, point16]


def is_corner(image, row, col, threshold):
    '''
    输入:
        image：输入图片数据,要求为灰度图片
        row：行坐标 注意row要大于等于3
        col：列坐标 注意col要大于等于3
        threshold：阈值
    输出:
        返回True或者False
    '''
    rows, cols = image.shape[:2]
    if row < 3 or col < 3:
        return False
    if row >= rows - 3 or col >= cols - 3:
        return False
    intensity = int(image[row][col])
    ROI = circle(row, col)
    # 获取位置1,9,5,13的像素值
    row1, col1 = ROI[0]
    row9, col9 = ROI[8]
    row5, col5 = ROI[4]
    row13, col13 = ROI[12]
    intensity1 = int(image[row1][col1])
    intensity9 = int(image[row9][col9])
    intensity5 = int(image[row5][col5])
    intensity13 = int(image[row13][col13])
    # 统计上面4个位置中满足  像素值  >  intensity + threshold点的个数
    countMore = 0
    # 统计上面4个位置中满足 像素值  < intensity - threshold点的个数
    countLess = 0
    if intensity1 - intensity > threshold:
        countMore += 1
    elif intensity1 + threshold < intensity:
        countLess += 1
    if intensity9 - intensity > threshold:
        countMore += 1
    elif intensity9 + threshold < intensity:
        countLess += 1
    if intensity5 - intensity > threshold:
        countMore += 1
    elif intensity5 + threshold < intensity:
        countLess += 1
    if intensity13 - intensity > threshold:
        countMore += 1
    elif intensity13 + threshold < intensity:
        countLess += 1

    return countMore >= 3 or countLess >= 3


def areAdjacent(point1, point2):
    # 通过欧拉距离来确定两个点是否相邻,如果它们在彼此的四个像素内，则两个点相邻
    row1, col1 = point1
    row2, col2 = point2
    xDist = row1 - row2
    yDist = col1 - col2
    return (xDist ** 2 + yDist ** 2) ** 0.5 <= 4


def calculateScore(image, point):
    '''
    计算特征点响应大小，得分V定义为p和它周围16个像素点的绝对偏差之，通过两个相邻的特征点比较,V值较小的点移除
    输入:
        image：输入图片数据,要求为灰度图片
        point: 角点坐标
    '''
    col, row = point
    intensity = int(image[row][col])
    ROI = circle(row, col)
    values = []
    for p in ROI:
        values.append(int(image[p]))
    score = 0
    for value in values:
        score += abs(intensity - value)
    return score


def suppress(image, corners):
    '''
    输入:
        image： 灰度值数组
        corners ： list类型
    '''
    i = 1
    while i < len(corners):
        currPoint = corners[i]
        prevPoint = corners[i - 1]  # 由于相邻的角点在corners列表中彼此相邻，所以我们写成下面形式
        if areAdjacent(prevPoint, currPoint):
            currScore = calculateScore(image, currPoint)  # 计算响应分数
            prevScore = calculateScore(image, prevPoint)
            if (currScore > prevScore):
                # 移除较小分数的点
                del (corners[i - 1])
            else:
                del (corners[i])
        else:
            i += 1
            continue
    return


def detect(image, threshold=10, nonMaximalSuppress=True):
    '''
    输入：
        image： 灰度值数组
        threshold：int类型，用于过滤非角点
    返回：
        corners：角点的坐标
    '''
    corners = []
    rows, cols = image.shape[:2]
    image = medianBlur(image, 3)  # 中值滤波
    cv2.imshow('medianBlur', image)
    cv2.waitKey(20)
    # 开始搜寻角点
    for row in range(rows):
        for col in range(cols):
            if is_corner(image, row, col, threshold):
                corners.append((col, row))
    if nonMaximalSuppress:
        suppress(image, corners)# 非极大值抑制
    return corners


def test():
    image = cv2.imread('test0.jpg')
    image = cv2.resize(image, dsize=(600, 400))
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = detect(imgray)
    print('检测到的角点个数为：', len(corners))
    for point in corners:
        cv2.circle(image, point, 1, (0, 255, 0), 1)
    cv2.imshow('FAST', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()