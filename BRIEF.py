# -*- coding: utf-8 -*-
# BRIEF特征描述符
import cv2
import numpy as np
import functools


class BriefDescriptorExtractor(object):
    # BRIEF描述符实现，这里只实现了16字节
    def __init__(self, byte=16):
        self.__patch_size = 48  # 邻域范围
        self.__kernel_size = 9  # 平均积分核大小
        self.__bytes = byte  # 占用字节数16，对应描述子长度16*8=128

    def compute(self, image, keypoints):
        # 计算特征描述符，keypoints：图像的关键点集合，函数返回特征点，特征描述符元组
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.clone()
        self.__image_sum = cv2.integral(gray_image, sdepth=cv2.CV_32S)  # 计算积分图像
        print(type(self.__image_sum), self.__image_sum.shape)
        # 移除接近边界的关键点
        keypoints_res = []
        rows, cols = image.shape[:2]
        for keypoint in keypoints:
            point = keypoint.pt
            if point[0] > (self.__patch_size + self.__kernel_size) / 2 and point[0] < cols - (
                    self.__patch_size + self.__kernel_size / 2):
                if point[1] > (self.__patch_size + self.__kernel_size) / 2 and point[1] < rows - (
                        self.__patch_size + self.__kernel_size) / 2:
                    keypoints_res.append(keypoint)
        return keypoints_res, self.pixelTests16(keypoints_res)  # 计算特征点描述符

    def pixelTests16(self, keypoints):
        # 创建BRIEF描述符，返回描述符
        descriptors = np.zeros((len(keypoints), self.__bytes), dtype=np.uint8)
        for i in range(len(keypoints)):
            # 固定默认参数
            SMOOTHED = functools.partial(self.smoothed_sum, keypoint=keypoints[i])
            '''
            functools.partial返回的是一个可调用的partial对象，使用方法是partial(func,*args,**kw),
            func是必须要传入的，而且至少需要一个args或是kw参数。
            '''
            descriptors[i][0] = (((SMOOTHED(-2, -1) < SMOOTHED(7, -1)) << 7) +
                                 ((SMOOTHED(-14, -1) < SMOOTHED(-3, 3)) << 6) +
                                 ((SMOOTHED(1, -2) < SMOOTHED(11, 2)) << 5) +
                                 ((SMOOTHED(1, 6) < SMOOTHED(-10, -7)) << 4) +
                                 ((SMOOTHED(13, 2) < SMOOTHED(-1, 0)) << 3) +
                                 ((SMOOTHED(-14, 5) < SMOOTHED(5, -3)) << 2) +
                                 ((SMOOTHED(-2, 8) < SMOOTHED(2, 4)) << 1) +
                                 ((SMOOTHED(-11, 8) < SMOOTHED(-15, 5)) << 0))
            descriptors[i][1] = (((SMOOTHED(-6, -23) < SMOOTHED(8, -9)) << 7) +
                                 ((SMOOTHED(-12, 6) < SMOOTHED(-10, 8)) << 6) +
                                 ((SMOOTHED(-3, -1) < SMOOTHED(8, 1)) << 5) +
                                 ((SMOOTHED(3, 6) < SMOOTHED(5, 6)) << 4) +
                                 ((SMOOTHED(-7, -6) < SMOOTHED(5, -5)) << 3) +
                                 ((SMOOTHED(22, -2) < SMOOTHED(-11, -8)) << 2) +
                                 ((SMOOTHED(14, 7) < SMOOTHED(8, 5)) << 1) +
                                 ((SMOOTHED(-1, 14) < SMOOTHED(-5, -14)) << 0))
            descriptors[i][2] = (((SMOOTHED(-14, 9) < SMOOTHED(2, 0)) << 7) +
                                 ((SMOOTHED(7, -3) < SMOOTHED(22, 6)) << 6) +
                                 ((SMOOTHED(-6, 6) < SMOOTHED(-8, -5)) << 5) +
                                 ((SMOOTHED(-5, 9) < SMOOTHED(7, -1)) << 4) +
                                 ((SMOOTHED(-3, -7) < SMOOTHED(-10, -18)) << 3) +
                                 ((SMOOTHED(4, -5) < SMOOTHED(0, 11)) << 2) +
                                 ((SMOOTHED(2, 3) < SMOOTHED(9, 10)) << 1) +
                                 ((SMOOTHED(-10, 3) < SMOOTHED(4, 9)) << 0))
            descriptors[i][3] = (((SMOOTHED(0, 12) < SMOOTHED(-3, 19)) << 7) +
                                 ((SMOOTHED(1, 15) < SMOOTHED(-11, -5)) << 6) +
                                 ((SMOOTHED(14, -1) < SMOOTHED(7, 8)) << 5) +
                                 ((SMOOTHED(7, -23) < SMOOTHED(-5, 5)) << 4) +
                                 ((SMOOTHED(0, -6) < SMOOTHED(-10, 17)) << 3) +
                                 ((SMOOTHED(13, -4) < SMOOTHED(-3, -4)) << 2) +
                                 ((SMOOTHED(-12, 1) < SMOOTHED(-12, 2)) << 1) +
                                 ((SMOOTHED(0, 8) < SMOOTHED(3, 22)) << 0))
            descriptors[i][4] = (((SMOOTHED(-13, 13) < SMOOTHED(3, -1)) << 7) +
                                 ((SMOOTHED(-16, 17) < SMOOTHED(6, 10)) << 6) +
                                 ((SMOOTHED(7, 15) < SMOOTHED(-5, 0)) << 5) +
                                 ((SMOOTHED(2, -12) < SMOOTHED(19, -2)) << 4) +
                                 ((SMOOTHED(3, -6) < SMOOTHED(-4, -15)) << 3) +
                                 ((SMOOTHED(8, 3) < SMOOTHED(0, 14)) << 2) +
                                 ((SMOOTHED(4, -11) < SMOOTHED(5, 5)) << 1) +
                                 ((SMOOTHED(11, -7) < SMOOTHED(7, 1)) << 0))
            descriptors[i][5] = (((SMOOTHED(6, 12) < SMOOTHED(21, 3)) << 7) +
                                 ((SMOOTHED(-3, 2) < SMOOTHED(14, 1)) << 6) +
                                 ((SMOOTHED(5, 1) < SMOOTHED(-5, 11)) << 5) +
                                 ((SMOOTHED(3, -17) < SMOOTHED(-6, 2)) << 4) +
                                 ((SMOOTHED(6, 8) < SMOOTHED(5, -10)) << 3) +
                                 ((SMOOTHED(-14, -2) < SMOOTHED(0, 4)) << 2) +
                                 ((SMOOTHED(5, -7) < SMOOTHED(-6, 5)) << 1) +
                                 ((SMOOTHED(10, 4) < SMOOTHED(4, -7)) << 0))
            descriptors[i][6] = (((SMOOTHED(22, 0) < SMOOTHED(7, -18)) << 7) +
                                 ((SMOOTHED(-1, -3) < SMOOTHED(0, 18)) << 6) +
                                 ((SMOOTHED(-4, 22) < SMOOTHED(-5, 3)) << 5) +
                                 ((SMOOTHED(1, -7) < SMOOTHED(2, -3)) << 4) +
                                 ((SMOOTHED(19, -20) < SMOOTHED(17, -2)) << 3) +
                                 ((SMOOTHED(3, -10) < SMOOTHED(-8, 24)) << 2) +
                                 ((SMOOTHED(-5, -14) < SMOOTHED(7, 5)) << 1) +
                                 ((SMOOTHED(-2, 12) < SMOOTHED(-4, -15)) << 0))
            descriptors[i][7] = (((SMOOTHED(4, 12) < SMOOTHED(0, -19)) << 7) +
                                 ((SMOOTHED(20, 13) < SMOOTHED(3, 5)) << 6) +
                                 ((SMOOTHED(-8, -12) < SMOOTHED(5, 0)) << 5) +
                                 ((SMOOTHED(-5, 6) < SMOOTHED(-7, -11)) << 4) +
                                 ((SMOOTHED(6, -11) < SMOOTHED(-3, -22)) << 3) +
                                 ((SMOOTHED(15, 4) < SMOOTHED(10, 1)) << 2) +
                                 ((SMOOTHED(-7, -4) < SMOOTHED(15, -6)) << 1) +
                                 ((SMOOTHED(5, 10) < SMOOTHED(0, 24)) << 0))
            descriptors[i][8] = (((SMOOTHED(3, 6) < SMOOTHED(22, -2)) << 7) +
                                 ((SMOOTHED(-13, 14) < SMOOTHED(4, -4)) << 6) +
                                 ((SMOOTHED(-13, 8) < SMOOTHED(-18, -22)) << 5) +
                                 ((SMOOTHED(-1, -1) < SMOOTHED(-7, 3)) << 4) +
                                 ((SMOOTHED(-19, -12) < SMOOTHED(4, 3)) << 3) +
                                 ((SMOOTHED(8, 10) < SMOOTHED(13, -2)) << 2) +
                                 ((SMOOTHED(-6, -1) < SMOOTHED(-6, -5)) << 1) +
                                 ((SMOOTHED(2, -21) < SMOOTHED(-3, 2)) << 0))
            descriptors[i][9] = (((SMOOTHED(4, -7) < SMOOTHED(0, 16)) << 7) +
                                 ((SMOOTHED(-6, -5) < SMOOTHED(-12, -1)) << 6) +
                                 ((SMOOTHED(1, -1) < SMOOTHED(9, 18)) << 5) +
                                 ((SMOOTHED(-7, 10) < SMOOTHED(-11, 6)) << 4) +
                                 ((SMOOTHED(4, 3) < SMOOTHED(19, -7)) << 3) +
                                 ((SMOOTHED(-18, 5) < SMOOTHED(-4, 5)) << 2) +
                                 ((SMOOTHED(4, 0) < SMOOTHED(-20, 4)) << 1) +
                                 ((SMOOTHED(7, -11) < SMOOTHED(18, 12)) << 0))
            descriptors[i][10] = (((SMOOTHED(-20, 17) < SMOOTHED(-18, 7)) << 7) +
                                  ((SMOOTHED(2, 15) < SMOOTHED(19, -11)) << 6) +
                                  ((SMOOTHED(-18, 6) < SMOOTHED(-7, 3)) << 5) +
                                  ((SMOOTHED(-4, 1) < SMOOTHED(-14, 13)) << 4) +
                                  ((SMOOTHED(17, 3) < SMOOTHED(2, -8)) << 3) +
                                  ((SMOOTHED(-7, 2) < SMOOTHED(1, 6)) << 2) +
                                  ((SMOOTHED(17, -9) < SMOOTHED(-2, 8)) << 1) +
                                  ((SMOOTHED(-8, -6) < SMOOTHED(-1, 12)) << 0))
            descriptors[i][11] = (((SMOOTHED(-2, 4) < SMOOTHED(-1, 6)) << 7) +
                                  ((SMOOTHED(-2, 7) < SMOOTHED(6, 8)) << 6) +
                                  ((SMOOTHED(-8, -1) < SMOOTHED(-7, -9)) << 5) +
                                  ((SMOOTHED(8, -9) < SMOOTHED(15, 0)) << 4) +
                                  ((SMOOTHED(0, 22) < SMOOTHED(-4, -15)) << 3) +
                                  ((SMOOTHED(-14, -1) < SMOOTHED(3, -2)) << 2) +
                                  ((SMOOTHED(-7, -4) < SMOOTHED(17, -7)) << 1) +
                                  ((SMOOTHED(-8, -2) < SMOOTHED(9, -4)) << 0))
            descriptors[i][12] = (((SMOOTHED(5, -7) < SMOOTHED(7, 7)) << 7) +
                                  ((SMOOTHED(-5, 13) < SMOOTHED(-8, 11)) << 6) +
                                  ((SMOOTHED(11, -4) < SMOOTHED(0, 8)) << 5) +
                                  ((SMOOTHED(5, -11) < SMOOTHED(-9, -6)) << 4) +
                                  ((SMOOTHED(2, -6) < SMOOTHED(3, -20)) << 3) +
                                  ((SMOOTHED(-6, 2) < SMOOTHED(6, 10)) << 2) +
                                  ((SMOOTHED(-6, -6) < SMOOTHED(-15, 7)) << 1) +
                                  ((SMOOTHED(-6, -3) < SMOOTHED(2, 1)) << 0))
            descriptors[i][13] = (((SMOOTHED(11, 0) < SMOOTHED(-3, 2)) << 7) +
                                  ((SMOOTHED(7, -12) < SMOOTHED(14, 5)) << 6) +
                                  ((SMOOTHED(0, -7) < SMOOTHED(-1, -1)) << 5) +
                                  ((SMOOTHED(-16, 0) < SMOOTHED(6, 8)) << 4) +
                                  ((SMOOTHED(22, 11) < SMOOTHED(0, -3)) << 3) +
                                  ((SMOOTHED(19, 0) < SMOOTHED(5, -17)) << 2) +
                                  ((SMOOTHED(-23, -14) < SMOOTHED(-13, -19)) << 1) +
                                  ((SMOOTHED(-8, 10) < SMOOTHED(-11, -2)) << 0))
            descriptors[i][14] = (((SMOOTHED(-11, 6) < SMOOTHED(-10, 13)) << 7) +
                                  ((SMOOTHED(1, -7) < SMOOTHED(14, 0)) << 6) +
                                  ((SMOOTHED(-12, 1) < SMOOTHED(-5, -5)) << 5) +
                                  ((SMOOTHED(4, 7) < SMOOTHED(8, -1)) << 4) +
                                  ((SMOOTHED(-1, -5) < SMOOTHED(15, 2)) << 3) +
                                  ((SMOOTHED(-3, -1) < SMOOTHED(7, -10)) << 2) +
                                  ((SMOOTHED(3, -6) < SMOOTHED(10, -18)) << 1) +
                                  ((SMOOTHED(-7, -13) < SMOOTHED(-13, 10)) << 0))
            descriptors[i][15] = (((SMOOTHED(1, -1) < SMOOTHED(13, -10)) << 7) +
                                  ((SMOOTHED(-19, 14) < SMOOTHED(8, -14)) << 6) +
                                  ((SMOOTHED(-4, -13) < SMOOTHED(7, 1)) << 5) +
                                  ((SMOOTHED(1, -2) < SMOOTHED(12, -7)) << 4) +
                                  ((SMOOTHED(3, -5) < SMOOTHED(1, -5)) << 3) +
                                  ((SMOOTHED(-2, -2) < SMOOTHED(8, -10)) << 2) +
                                  ((SMOOTHED(2, 14) < SMOOTHED(8, 7)) << 1) +
                                  ((SMOOTHED(3, 9) < SMOOTHED(8, 2)) << 0))
        return descriptors

    def smoothed_sum(self, y, x, keypoint):
        '''
        这里我们采用随机点平滑，采用随机点邻域内积分和代替
        输入:
            self.__image_sum:图像积分图
            keypoint：其中一个关键点
            y,x：x和y表示点对中某一个像素相对于特征点的坐标
        返回:
            函数返回滤波的结果
        '''
        half_kernel = self.__kernel_size // 2
        img_y = int(keypoint.pt[1] + 0.5) + y  # 计算点对中某一个像素的绝对坐标
        img_x = int(keypoint.pt[0] + 0.5) + x
        # 计算以该像素为中心，以KERNEL_SIZE为边长的正方形内所有像素灰度值之和，本质上是均值滤波
        ret = self.__image_sum[img_y + half_kernel + 1][img_x + half_kernel + 1] \
              - self.__image_sum[img_y + half_kernel + 1][img_x - half_kernel] \
              - self.__image_sum[img_y - half_kernel][img_x + half_kernel + 1] \
              + self.__image_sum[img_y - half_kernel][img_x - half_kernel]
        return ret


def brief_test():
    img1 = cv2.imread('test0.jpg')
    img1 = cv2.resize(img1, dsize=(600, 400))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('test1.jpg')
    img2 = cv2.resize(img2, dsize=(600, 400))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image1 = gray1.copy()
    image2 = gray2.copy()
    image1 = cv2.medianBlur(image1, 5)  # 中值滤波将图像的每个像素用邻域像素的中值代替
    image2 = cv2.medianBlur(image2, 5)

    '''
    1.使用SURF算法检测关键点
    '''

    surf = cv2.xfeatures2d.SURF_create(3000)# 创建一个SURF对象，阈值越高，能识别的特征就越少，因此可以采用试探法来得到最优检测。
    keypoints1 = surf.detect(image1)
    keypoints2 = surf.detect(image2)
    # 在图像上绘制关键点
    image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv2.drawKeypoints(image=image2, keypoints=keypoints2, outImage=image2, color=(255, 0, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    '''
    2.计算特征描述符
    '''
    brief = BriefDescriptorExtractor(16)
    keypoints1, descriptors1 = brief.compute(image1, keypoints1)
    keypoints2, descriptors2 = brief.compute(image2, keypoints2)
    print(descriptors1[:5])
    print(descriptors2[:5])
    print('descriptors1:', len(descriptors1), descriptors1.shape, 'descriptors2', len(descriptors2), descriptors2.shape)

    '''
    3.匹配  汉明距离匹配特征点
    '''
    matcher = cv2.BFMatcher_create(cv2.HAMMING_NORM_TYPE)
    matchePoints = matcher.match(descriptors1, descriptors2)
    print('matchePoints', type(matchePoints), len(matchePoints), matchePoints[0])

    # 提取强匹配特征点
    minMatch = 1
    maxMatch = 0
    for i in range(len(matchePoints)):
        if minMatch > matchePoints[i].distance:
            minMatch = matchePoints[i].distance
        if maxMatch < matchePoints[i].distance:
            maxMatch = matchePoints[i].distance
    print('最佳匹配值是:', minMatch)
    print('最差匹配值是:', maxMatch)
    # 获取排雷在前边的几个最优匹配结果
    goodMatchePoints = []
    for i in range(len(matchePoints)):
        if matchePoints[i].distance < minMatch + (maxMatch - minMatch) / 3:
            goodMatchePoints.append(matchePoints[i])
    # 绘制最优匹配点
    outImg = None
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, goodMatchePoints, outImg, matchColor=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow('matche', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    brief_test()