from PIL import Image
import pytesseract
import cv2 as cv
import os
import Levenshtein
import shutil

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def imgPadding(imgpath):
    img = cv.imread(imgpath)
    shape = img.shape
    img = cv.copyMakeBorder(img, int(0.5 * shape[0]), int(0.5 * shape[0]), int(0.5 * shape[1]),
                            int(0.5 * shape[1]), cv.BORDER_CONSTANT, value=[255, 255, 255])
    return img

#------------------Phish Identification------------------#
def ocrIdentification(brand, flag, imgpath, true_):
    img = cv.imread(imgpath)
    text = pytesseract.image_to_string(Image.fromarray(img)).lower()
    if Levenshtein.ratio(brand, text) >= 0.3:
        flag = False
        true_ += 1
    else:
        padimg = imgPadding(imgpath)
        text = pytesseract.image_to_string(Image.fromarray(padimg)).lower()
        if Levenshtein.ratio(brand, text) >= 0.3:
            flag = False
            true_ += 1
    return flag, true_

#------------------Phish Detection-----------------------#
def targetList():
    target = []
    if os.path.exists("target_domain.txt"):
        with open("target_domain.txt") as f:
            for line in f:
                brand = line[:line.find("+")]
                targetdomain = line[line.find("+") + 1:]
                target.append((brand, targetdomain))
    else:
        tmp = []
        with open("targetList.txt", "r") as f:
            for line in f:
                brand = line[line.find("'") + 1: line.find(",") - 1]
                if "inc" in brand.lower():
                    brand = brand[:brand.lower().find("inc")]
                if "/" in brand.lower():
                    brand = brand[:brand.lower().find("/")]
                targetdomain = domainGet(brand)
                tmp.append(brand + "+" + targetdomain + "\n")
                target.append((brand, targetdomain))
        with open("target_domain.txt", "w") as f:
            for temp in tmp:
                f.write(temp)
    return target
def domainGet(query):
    page = requests.get("http://www.google.com/search?hl=en&q=" + query).content
    soup = BeautifulSoup(page, features="html.parser")
    for link in soup.find_all("a", href=re.compile("(?<=/url\?q=)(htt.*://.*)")):
        url = re.split(":(?=http)", link["href"].replace("/url?q=", ""))
        print(urlparse(url[0])[1])
        return urlparse(url[0])[1]
def ocrDetection(urld, flag, imgpath, true_, flag_detec):
    img = cv.imread(imgpath)
    text = pytesseract.image_to_string(Image.fromarray(img)).lower()
    for (tar, phishdomain) in target:
        if Levenshtein.ratio(tar, text) >= 0.4:
            flag_detec = True
            break
    if flag_detec:
        if phishdomain != urld+"\n":
            flag = False
            true_ += 1
    else:
        padimg = imgPadding(imgpath)
        text = pytesseract.image_to_string(Image.fromarray(padimg)).lower()
        for (tar, phishdomain) in target:
            if Levenshtein.ratio(tar, text) >= 0.4:
                flag_detec = True
                break
        if flag_detec:
            if phishdomain != urld+"\n":
                flag = False
                true_ += 1
    return flag, true_, flag_detec

#------------------Benign Detection----------------------#
def benignDetection(urld, flag, imgpath, true_, flag_detec):
    img = cv.imread(imgpath)
    text = pytesseract.image_to_string(Image.fromarray(img)).lower()
    for (tar, phishdomain) in target:
        if Levenshtein.ratio(tar, text) >= 0.9:
            flag_detec = True
            break
    if flag_detec:
        if "google" in phishdomain and not "google" in urld:
            flag = False
            true_ += 1
        else:
            if "www." in phishdomain:
                phishdomain = phishdomain[4:]
            if phishdomain != urld + "\n":
                flag = False
                true_ += 1
    else:
        padimg = imgPadding(imgpath)
        text = pytesseract.image_to_string(Image.fromarray(padimg)).lower()
        for (tar, phishdomain) in target:
            if Levenshtein.ratio(tar, text) >= 0.9:
                flag_detec = True
                break
        if flag_detec:
            if "google" in phishdomain and not "google" in urld:
                flag = False
                true_ += 1
            else:
                if "www." in phishdomain:
                    phishdomain = phishdomain[4:]
                if phishdomain != urld + "\n":
                    flag = False
                    true_ += 1
    return flag, true_, flag_detec


def testPhish_Identi(path):
    count = 0
    true_yolo = 0
    true_ocr = 0
    true_ocr_yolo = 0
    try:
        for root, dirs, files in os.walk(path):
            for dir_ in dirs:
                brand = dir_[:dir_.find("+")].lower()
                dirpath = os.path.join(root, dir_)
                count += 1
                print(count)
                if count < 10000:
                    continue
                # if count > 10000:
                #     break
                for dirroot, chdir, imgfile in os.walk(dirpath):
                    flag_ocr = True
                    flag_yolo = True
                    flag_ocr_yolo = True
                    countimg = 0
                    for img_ in imgfile:
                        countimg += 1
                        imgpath = os.path.join(dirroot, img_)
                        if "ocr" in imgpath and flag_ocr:
                            flag_ocr, true_ocr = ocrIdentification(brand, flag_ocr, imgpath, true_ocr)
                        elif flag_yolo:
                            flag_yolo, true_yolo = ocrIdentification(brand, flag_yolo, imgpath, true_yolo)
                        if flag_ocr_yolo:
                            flag_ocr_yolo, true_ocr_yolo = ocrIdentification(brand, flag_ocr_yolo, imgpath,
                                                                             true_ocr_yolo)
                        if not (flag_ocr or flag_yolo or flag_ocr_yolo):
                            break
                        if flag_ocr_yolo and countimg == len(imgfile):
                            copypath = "ocr+yolo_error_gentle/" + dir_
                            shutil.copytree(dirpath, copypath)
    except Exception as e:
        print(e)

    print("Identification :")
    print("urlCount: %d" % count)
    print("test_OCR: " + str(true_ocr))
    print("test_YOLO: " + str(true_yolo))
    print("test_OCR+YOLO: " + str(true_ocr_yolo))


def testPhish_Detec(path):
    count = 0
    true_yolo = 0
    true_ocr = 0
    true_ocr_yolo = 0
    detec = 0
    try:
        for root, dirs, files in os.walk(path):
            for dir_ in dirs:
                oridir = "D:\\Yiwen\\yolov3_image_proces_mild\\newDatabase_gentle\\" + dir_ + "\\info.txt"
                with open(oridir) as f:
                    content = f.readlines()
                url = eval(content[0])['url']
                urld = urlparse(url)[1]
                dirpath = os.path.join(root, dir_)
                count += 1
                print(count)
                if count < 11000:
                    continue
                # if count > 11000:
                #     break
                for dirroot, chdir, imgfile in os.walk(dirpath):
                    flag_ocr = True
                    flag_yolo = True
                    flag_ocr_yolo = True
                    flag_detec = False
                    flag_detec_once = True
                    countimg = 0
                    for img_ in imgfile:
                        countimg += 1
                        imgpath = os.path.join(dirroot, img_)
                        if "ocr" in imgpath and flag_ocr:
                            flag_ocr, true_ocr, flag_detec = ocrDetection(
                                urld, flag_ocr, imgpath, true_ocr, flag_detec)
                        elif flag_yolo:
                            flag_yolo, true_yolo, flag_detec = ocrDetection(
                                urld, flag_yolo, imgpath, true_yolo, flag_detec)
                        if flag_ocr_yolo:
                            flag_ocr_yolo, true_ocr_yolo, flag_detec = ocrDetection(
                                urld, flag_ocr_yolo, imgpath, true_ocr_yolo, flag_detec)
                        if flag_detec_once and flag_detec:
                            detec += 1
                            flag_detec_once = False
                        if not (flag_ocr or flag_yolo or flag_ocr_yolo):
                            break
                        if flag_ocr_yolo and countimg == len(imgfile):
                            copypath = "ocr+yolo_error_Detect_gentle/" + dir_
                            shutil.copytree(dirpath, copypath)
    except Exception as e:
        print(e)

    print("Detection :")
    print("urlCount: %d" % count)
    print("test_OCR: " + str(true_ocr))
    print("test_YOLO: " + str(true_yolo))
    print("test_OCR+YOLO: " + str(true_ocr_yolo))
    print("Detect: " + str(detec))


def testBenign_Detec(path):
    count = 0
    detec = 0
    detec_to_phish = 0
    for root, dirs, files in os.walk(path):
        for dir_ in dirs:
            dirpath = os.path.join(root, dir_)
            urld = dir_
            count += 1
            print(count)
            if count < 15000:
                continue
            # if count > 15000:
            #     break
            for dirroot, chdir, imgfile in os.walk(dirpath):
                flag = True
                flag_detec = False
                flag_detec_once = True
                countimg = 0
                for img_ in imgfile:
                    countimg += 1
                    imgpath = os.path.join(dirroot, img_)
                    if flag:
                        try:
                            flag, detec_to_phish, flag_detec = benignDetection(
                                urld, flag, imgpath, detec_to_phish, flag_detec)
                        except Exception as e:
                            print(e)
                    else:
                        try:
                            copypath = "benign_error/" + dir_
                            shutil.copytree(dirpath, copypath)
                        except Exception as e:
                            print(e)
                        break
                    if flag_detec_once and flag_detec:
                        detec += 1
                        flag_detec_once = False

    print("Benign :")
    print("urlCount: %d" % (count-1))
    print("detec_to_phish: " + str(detec_to_phish))
    print("Detect: " + str(detec))

if __name__ == '__main__':
    path = "D:\\Yiwen\\yolov3_image_process\\detection"
    global target
    target = targetList()
    testBenign_Detec(path)
