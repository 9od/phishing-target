# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:05:46 2019

@author: yuxuan
"""

from PIL import Image
import glob
import os

for j in range(1, 30):
    name = glob.glob(".\\data\\logo_collection\\s"+str(j)+"\\*.jpg")
#print(type(name))#name is a list
#print(len(name))#2425
#print(name[2420])
    for i in range(len(name)):
        img = Image.open(name[i])
        if img.mode == "P":
            img = img.convert('RGB')
        name[i] = name[i][:-3]
        if not os.path.exists(".\\data\\logos\\training\\S" + str(j)):
            os.makedirs(".\\data\\logos\\training\\S" + str(j))  
        savepath=".\\data\\logos\\training\\S"+str(j)+"\\"+str(i+1)+".pgm"
        img.save(savepath)
        