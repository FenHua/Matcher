# -*- coding: utf-8 -*-
# WLD 韦伯局部描述符
import cv2
import numpy as np


class WLD(object):
    def __init__(self):
        # 差分激励由中心像素与周围像素强度的差异以及中心像素强度组成，分别由f1和f2滤波器得出。
        self.__f1 = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])
        self.__f2 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]])
        # 方向反映局部窗内灰度变化的空间分布信息。通过局部窗内水平方向与垂直方向上邻域像素点
        # 的灰度差值比值的反正切变换来描述。 方向分为竖直方向和水平方向，由f3和f4滤波器得出。
        self.__f3 = np.array([[0, -1, 0],
                              [0, 0, 0],
                              [0, 1, 0]])
        self.__f4 = np.array([[0, 0, 0],
                              [1, 0, -1],
                              [0, 0, 0]])
        self.__T = 12  # 方向量化个数，如果修改，也需要修改__classify_fai函数
        self.__M = 8  # 差分激励量化个数，如果修改，也需要修改__classify_epc函数
        self.__S = 4  # 每个频段上将差分激励均匀地量化为S格，可以修改

    def calc_hist(self, image, concatenate=True):
        '''
        输出统计直方图 形状[M,T,S]或[MxTxS,]
        输入:
            image：输入灰度图
            concatenate：表明输出直方图是否合并为一维
        '''
        his = np.zeros((self.__M, self.__T, self.__S), np.float32)
        rows, cols = image.shape[:2]
        sum_pix = rows * cols  # 计算像素点个数
        epc, theta = self.__compute(image)  # 计算每个像素点的差分激励和方向
        for i in range(rows):
            for j in range(cols):
                m = self.__classify_epc(epc[i][j])  # 把差分激励分为8个区间，分别对应一个类别
                t = theta[i][j]
                s = self.__classify_s(epc[i][j], m)
                his[m - 1][t - 1][s - 1] += 1
        his /= sum_pix  # 归一化
        if concatenate:
            his = np.reshape(his, -1)
        return his

    def __compute(self, image):
        # 计算每个像素点的差分激励和方向
        rows, cols = image.shape[:2]
        epc = np.zeros((rows, cols), dtype=np.float32)  # 用于保存每个像素点对应的差分激励算子
        theta = np.zeros((rows, cols), dtype=np.float32)  # 用于保存每个像素点对应的方向算子
        # 计算差分激励ξ
        v1 = cv2.filter2D(image, cv2.CV_16SC1, self.__f1)
        v2 = cv2.filter2D(image, cv2.CV_16SC1, self.__f2)
        for i in range(rows):
            for j in range(cols):
                epc[i][j] = np.arctan(v1[i][j] / (v2[i][j] + 0.0001))  # -π/2~π/2
        # 计算每个像素点的方向Φ，把方向[0,2pi]均匀地量化为12个区间
        v3 = cv2.filter2D(image, cv2.CV_16SC1, self.__f3)
        v4 = cv2.filter2D(image, cv2.CV_16SC1, self.__f4)
        for i in range(rows):
            for j in range(cols):
                theta[i][j] = np.arctan(v3[i][j] / ((v4[i][j]) + 0.0001))
                if v3[i][j] > 0 and v4[i][j] > 0:
                    pass
                elif v3[i][j] > 0 and v4[i][j] < 0:
                    theta[i][j] = theta[i][j] + 2 * np.pi
                else:
                    theta[i][j] = theta[i][j] + np.pi
                theta[i][j] = self.__classify_fai(theta[i][j])
        theta = theta.astype(np.uint8)
        return epc, theta

    def __classify_fai(self, value):
        '''
        把方向值value分类  类别为1、2、3、4、5、6、7、8、9、10、11、12
        输入value：数值 0~2π之间
        '''
        if value >= 0 and value < 0.15 * np.pi:
            return 1
        elif value >= np.pi * 0.15 and value < 0.35 * np.pi:
            return 2
        elif value >= np.pi * 0.35 and value < 0.5 * np.pi:
            return 3
        elif value >= np.pi * 0.5 and value < 0.65 * np.pi:
            return 4
        elif value >= 0.65 * np.pi and value < 0.85 * np.pi:
            return 5
        elif value >= np.pi * 0.85 and value < np.pi:
            return 6
        elif value >= np.pi and value < 1.15 * np.pi:
            return 7
        elif value >= 1.15 * np.pi and value < 1.35 * np.pi:
            return 8
        elif value >= np.pi * 1.35 and value < 1.5 * np.pi:
            return 9
        elif value >= 1.5 * np.pi and value < 1.65 * np.pi:
            return 10
        elif value >= 1.65 * np.pi and value < 1.85 * np.pi:
            return 11
        else:
            return 12

    def __classify_epc(self, value):
        '''
        把差分激励值value分类，划分为8个区间  类别为1、2、3、4、5、6、7、8
        输入value：数值 -π/2~π/2之间
        '''
        if value >= np.pi * (-0.5) and value < (-0.3) * np.pi:
            return 1
        elif value >= np.pi * (-0.3) and value < (-0.15) * np.pi:
            return 2
        elif value >= np.pi * (-0.15) and value < (-0.05) * np.pi:
            return 3
        elif value >= np.pi * (-0.05) and value < 0:
            return 4
        elif value >= 0 and value < 0.05 * np.pi:
            return 5
        elif value >= np.pi * 0.05 and value < 0.15 * np.pi:
            return 6
        elif value >= np.pi * 0.15 and value < 0.3 * np.pi:
            return 7
        else:
            return 8

    def __classify_s(self, value, label):
        '''
        将每个区间的差分激励再次划分为S格
        输入：
            value：差分激励值   -π/2~π/2之间
            label：当前所属区间  1,2,3,...,8
        '''
        if label == 1:
            space = ((-0.3) * np.pi - (-0.5) * np.pi) / self.__S
            return int((value - (-0.5) * np.pi) / space) + 1
        elif label == 2:
            space = ((-0.15) * np.pi - (-0.3) * np.pi) / self.__S
            return int((value - (-0.3) * np.pi) / space) + 1
        elif label == 3:
            space = ((-0.05) * np.pi - (-0.15) * np.pi) / self.__S
            return int((value - (-0.15) * np.pi) / space) + 1
        elif label == 4:
            space = 0 - (-0.05) * np.pi / self.__S
            return int((value - (-0.05) * np.pi) / space) + 1
        elif label == 5:
            space = 0.05 * np.pi / self.__S
            return int(value / space) + 1
        elif label == 6:
            space = (0.15 * np.pi - 0.05 * np.pi) / self.__S
            return int((value - 0.05 * np.pi) / space) + 1
        elif label == 7:
            space = (0.3 * np.pi - 0.15 * np.pi) / self.__S
            return int((value - 0.15 * np.pi) / space) + 1
        else:
            space = (0.5 * np.pi - 0.3 * np.pi) / self.__S
            n = int((value - 0.3 * np.pi) / space) + 1
            if n == self.__S + 1:
                n = self.__S
            return n


def draw_hist(hist):
    # 首先先创建一个黑底的图像，图像高为100，宽为直方图的长度
    width = len(hist)
    height = 100
    draw_image = np.zeros((height, width, 3), dtype=np.uint8)  # 绘制图像是一个8位的3通道图像
    max_value = np.max(hist)  # 获取最大值
    value = np.asarray((hist * 0.9 / max_value * height), dtype=np.int8)  # 数值量化
    for i in range(width):
        cv2.line(draw_image, (i, height - 1), (i, height - 1 - value[i]), (255, 0, 0))
    cv2.imshow('hist', draw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('test0.jpg')
    image = cv2.resize(image, dsize=(600, 400))
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wld = WLD()
    hist = wld.calc_hist(imgray)
    print(hist.shape)
    print(hist)
    print(np.sum(hist))
    draw_hist(hist)