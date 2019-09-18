# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:48:47 2019

@author: yuxuan
"""

import os
import random
import numpy as np
import cv2 as cv
import tensorflow as tf

    
def imgResize(img, new_width, new_height, interp=0):
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)
    img = cv.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 255, np.uint8)
    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)
    image_padded[dh: resize_h+dh, dw: resize_w + dw, :] = img
    return image_padded


def randomCrop(img, imgpath, num):
    height, width = img.shape[:2]    
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        file_contents = tf.io.read_file(imgpath)
        image = tf.image.decode_image(file_contents, channels=3)
        image_crop_en_list = []
        for i in range(num):
            image_crop = tf.image.random_crop(image, [int(height*0.8), int(width*0.8), 3])
            image_crop_en_list.append(tf.image.encode_png(image_crop))
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            results = sess.run(image_crop_en_list)
            return results


def targetList():
    listpath = "./targetlist/phish_collect"
    newpath = "./targetlist/phish_collect/"
    for root, dirs, files in os.walk(listpath):
        for file in files:
            imgname = file[:file.find(".")]+".png"
            imgpath = os.path.join(root, file) 
            img = cv.imread(imgpath)
            imgresize = imgResize(img, 100, 100, interp=0)
            newimgpath = newpath+imgname
            cv.imwrite(newimgpath, imgresize)            
            
        
def dataAugment():
    path = "./dataset/faces/testing"
#    path = "./dataset/logo_collection"
    for root, dirs, files in os.walk(path):
        for dir_ in dirs:
            dirpath = os.path.join(root, dir_)       
            for dirroot, chdir, imgfile in os.walk(dirpath):
                count = 0
                newpath = "./dataset/faces/"+dir_
                if not os.path.exists(newpath):
                    os.makedirs(newpath)            
                for img in imgfile:
                    count += 1
                    
                    # Resize
                    imgpath = dirpath+"/"+img
                    imgpic = cv.imread(imgpath)  
                    imgresize = imgResize(imgpic, 100, 100, interp=0)
      #              imgresize = cv.GaussianBlur(imgresize,(7,7),0)
      #              imgresize = cv.cvtColor(imgresize,cv.COLOR_RGB2GRAY)
                    newimgpath = newpath+"/"+img[:img.find(".")]+".png"
                    cv.imwrite(newimgpath, imgresize)
                
# =============================================================================
#                     # Crop
#                     num = random.randint(0,3)
#                     imgcrop = randomCrop(imgpic, imgpath, num)
#                     for idx, imge in enumerate(imgcrop):
#                         count += 1
#                         newimgpath = newpath+"/"+str(count)+".png"
#                         with open(newimgpath,'wb') as f:
#                             f.write(imge)
#                         img = cv.imread(newimgpath)    
#                         imgresize = imgResize(img, 224, 224, interp=0)
#                         newimgpath = newpath+"/"+str(count)+".png"
#                         cv.imwrite(newimgpath, imgresize) 
# =============================================================================

def phishCrop():
    path = "./targetlist/phish_collect/"
    for root, dirs, img in os.walk(path):
        for img_ in img:
            imgpath = path+img_
            image = cv.imread(imgpath)
            height, width = image.shape[:2]
            clogo = image[0:width][0:int(2*height/3)]  # (left, upper, right, lower)
            clogo = imgResize(clogo, 100, 100, interp=0)
            cv.imwrite(imgpath, clogo)


if __name__ == '__main__': 
    dataAugment()