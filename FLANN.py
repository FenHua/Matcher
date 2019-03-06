# coding: utf-8
# 特征描述符匹配算法 暴力匹配，KNN匹配，FLANN匹配

import cv2 
import numpy as np

# FLANN匹配
def flann_test():
    img1 = cv2.imread('test0.jpg')    
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('test1.jpg')
    img2 = cv2.resize(img2,dsize=(450,300))
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    queryImage = gray1.copy()
    trainImage = gray2.copy()
    sift = cv2.xfeatures2d.SIFT_create(100)  # 创建SIFT对象
    keypoints1,descriptor1 = sift.detectAndCompute(queryImage,None)  # SIFT对象会使用DoG检测关键点，并且对每个关键点周围的区域计算特征向量
    keypoints2,descriptor2 = sift.detectAndCompute(trainImage,None)  # 该函数返回关键点的信息和描述符
    print('descriptor1:',descriptor1.shape,'descriptor2',descriptor2.shape)
    # 在图像上绘制关键点
    queryImage = cv2.drawKeypoints(image=queryImage,keypoints = keypoints1,outImage=queryImage,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    trainImage = cv2.drawKeypoints(image=trainImage,keypoints = keypoints2,outImage=trainImage,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('sift_keypoints1',queryImage)
    cv2.imshow('sift_keypoints2',trainImage)
    cv2.waitKey(20)
    
    # FLANN匹配 
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
    searchParams = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(indexParams,searchParams)
    matches = flann.knnMatch(descriptor1,descriptor2,k=2)
    print(type(matches),len(matches),matches[0])
    dMatch0 = matches[0][0]  # 获取queryImage中的第一个描述符在trainingImage中最匹配的一个描述符
    dMatch1 = matches[0][1]  # 获取queryImage中的第一个描述符在trainingImage中次匹配的一个描述符
    print('knnMatches',dMatch0.distance,dMatch0.queryIdx,dMatch0.trainIdx)
    print('knnMatches',dMatch1.distance,dMatch1.queryIdx,dMatch1.trainIdx)
    matchesMask = [[0,0] for i in range(len(matches))]  # 设置mask,过滤匹配点
    minRatio = 3/4
    for i,(m,n) in enumerate(matches):
        if m.distance/n.distance < minRatio:
            matchesMask[i] = [1,0]  #只绘制最优匹配点
    drawParams = dict(#singlePointColor=(255,0,0),matchColor=(0,255,0),
                      matchesMask = matchesMask,
                      flags = 0)
    resultImage = cv2.drawMatchesKnn(queryImage,keypoints1,trainImage,keypoints2,matches,None,**drawParams)
    cv2.imshow('matche',resultImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    flann_test()