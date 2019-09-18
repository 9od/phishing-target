# -*- coding: utf-8 -*-

import os
import shutil
import random

def dataTest(path):
    for root, dirs, files in os.walk(path):
        dataSample = random.sample(dirs, 200)
        for dir_ in dataSample:
            imgpath = os.path.join(root, dir_) + "/shot.png"
            htmlpath=os.path.join(root, dir_) + "/html.txt"
            if os.path.exists(htmlpath) and os.path.exists(imgpath):
                with open(htmlpath) as file_object:
                    content = file_object.read()
                flag = (os.path.getsize(imgpath)/1024 > 10)\
                       and (not "NUS" in content)\
                       and (not "domain" in content)\
                       and (not "sorry" in content)\
                       and (not "Suspended" in content)\
                       and (not "The website you were trying to reach is temporarily unavailable" in content)\
                       and ("logo" in content)
                if flag:
                    imgnewpath = "./phishTest/" + str(dir_) + ".png"
                    shutil.copy(imgpath, imgnewpath)
        break
if __name__ == '__main__':
    path = "./Database"
    dataTest(path)