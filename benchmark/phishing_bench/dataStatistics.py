# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:54:35 2019

@author: yuxuan
"""

import os
from collections import Counter
import matplotlib.pyplot as plt


def Statistics(path):
    dirnames=[]
    for root, dirs, files in os.walk(path):
        for dir_ in dirs:
            dirname = dir_[:dir_.find("+")]
            dirname = dirname.replace(",", "/")
            dirnames.append(dirname)
    di = Counter(dirnames)        
    dirname_ = sorted(di.items(), key=lambda item:item[1], reverse=True)
    return dirname_, len(dirnames)

if __name__ == '__main__': 
    dir_, count = Statistics("D:/yuxuan1/phishwebsGet/Database")
    fileObject = open('TargetList.txt', 'w')
    for tar in dir_:
        tar = str(tar)
        tar = tar.replace("(", "")
        tar = tar.replace(")", "")
        fileObject.write(tar)
        fileObject.write('\n')
    fileObject.close()
    
    tarname=[]
    tarcount=[]
    for d in dir_:
        tarname.append(d[0])
        tarcount.append(d[1])
    plt.figure()
    plt.bar(range(len(tarcount)), tarcount, tick_label=tarname)
    plt.savefig("PhishingWebs.png")
    plt.show()