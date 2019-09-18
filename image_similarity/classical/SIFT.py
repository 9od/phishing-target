import cv2
from matplotlib import pyplot as plt
import os
import math

def getMatchNum(matches, ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  #将距离比率小于ratio的匹配点删选出来
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)

def SIFT():
    # ImgPaths
    path = 'D:/yuxuan/ImgSim/targetlist/'
    queryPath = path+'targetlist/s'  #图库路径
    samplePath = path + 'phish/s/2shrt.png'  #样本图片

    # 记录比较结果
    comparisonImageList = []
    #创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()
    #创建FLANN匹配对象
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    sampleImage = cv2.imread(samplePath, 0)
    kp1, des1 = sift.detectAndCompute(sampleImage, None)  #提取样本图片的特征
    for root, dirnames, filenames in os.walk(queryPath):
        for p in filenames:
            p = os.path.join(root, p)
            queryImage = cv2.imread(p, 0)
            # 提取比对图片的特征
            kp2, des2 = sift.detectAndCompute(queryImage, None)
            # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
            matches = flann.knnMatch(des1, des2, k=2)
            # 通过比率条件，计算出匹配程度
            (matchNum, matchesMask) = getMatchNum(matches, 0.9)
            matchRatio = matchNum * 100 / len(matches)
            drawParams = dict(matchColor=(0, 255, 0),
                              singlePointColor=(255, 0, 0),
                              matchesMask=matchesMask,
                              flags=0)
            comparisonImage = cv2.drawMatchesKnn(sampleImage, kp1, queryImage, kp2,
                                                 matches, None, **drawParams)
            # 记录下结果
            comparisonImageList.append((comparisonImage, matchRatio))
    # 按照匹配度排序
    comparisonImageList.sort(key=lambda x: x[1], reverse=True)
    count = len(comparisonImageList)

    #绘图显示
    plt.figure()
    for index, (image, ratio) in enumerate(comparisonImageList):
        ax = plt.subplot(5, 4, index+1)
        ax.set_title('%.2f%%' % ratio)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    SIFT()