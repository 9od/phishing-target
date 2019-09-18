# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:34:15 2019

@author: Yuxuan
"""

import os
import cv2 as cv

def Divide():
    rootpath = "D:/yuxuan/ImgSim/PhishingCheck/ImgSim/Siamese Network/Dataset/dataset-resize/database"
    newpath = "D:/yuxuan/ImgSim/PhishingCheck/ImgSim/Siamese Network/Dataset/dataset-resize/"
    for root, dirs, files in os.walk(rootpath):
        c = 0
        for dir_ in dirs:      
            path_ = os.path.join(root, dir_)
            for diroot, _, imgs in os.walk(path_):
                count = 0                
                for img in imgs:
                    count += 1
                    c += 1
                    oldpath = os.path.join(diroot, img)
                    if count < int(len(imgs)/4):
                        imgpath = newpath+"mix_test/"+str(dir_)+"/"
                    else:
                        imgpath = newpath+"mix_train/"+str(dir_)+"/"
                    if not os.path.exists(imgpath):
                        os.makedirs(imgpath)  
                    imgpic = cv.imread(oldpath)
                    newimgpath = imgpath+str(c)+".png"
                    cv.imwrite(newimgpath, imgpic)
 
if __name__ == '__main__': 
    Divide()