import os
import cv2 as cv
import Levenshtein
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options


def imgCrop(imgpath):
    img = cv.imread(imgpath)
    shape = img.shape
    img1 = img[0:shape[1], 0:int(0.6 * shape[0])]
    img_crop1 = imgpath[:imgpath.find(".")] + "_crop1.png"
    cv.imwrite(img_crop1, img1)
    img2 = img[0:int(0.6 * shape[1]), 0:shape[0]]
    img_crop2 = imgpath[:imgpath.find(".")] + "_crop2.png"
    cv.imwrite(img_crop2, img2)
    return img_crop1, img_crop2


def imgPadding(imgpath):
    img = cv.imread(imgpath)
    shape = img.shape
    img = cv.copyMakeBorder(img, int(0.4 * shape[0]), int(0.4 * shape[0]), int(0.4 * shape[1]),
                            int(0.4 * shape[1]), cv.BORDER_CONSTANT, value=[255, 255, 255])
    img_padding = imgpath[:imgpath.find(".")] + "_padding.png"
    cv.imwrite(img_padding, img)
    return img_padding


def googleSearch(filePath, brand):
    searchUrl = 'http://www.google.hr/searchbyimage/upload'
    multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
    response = requests.post(searchUrl, files=multipart, allow_redirects=False)
    fetchUrl = response.headers['Location']
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(fetchUrl)
    pagesource = driver.page_source
    for link in driver.find_elements_by_xpath("//input[contains(@class, 'gLFyf gsfi')]"):
        value = link.get_attribute('value')
    driver.quit()
    brand_name = str.lower(brand)
    return value, brand_name


def testYOLO():
    true_ = 0
    count = 0
    path = "D:/yuxuan/yolo/testing_result/08192019"
    erroroot = "D:/yuxuan/yolo/testing_result/08192019/_error"
    for root, dirs, files in os.walk(path):
        for dir_ in dirs:
            if dir_ == "_error":
                continue
            dirpath = os.path.join(root, dir_)
            for dirroot, chdir, imgfile in os.walk(dirpath):
                count += 1
                for img in imgfile:
                    imgpath = os.path.join(dirroot, img)
                    search_result, brand_name = googleSearch(imgpath, dir_)
                    if Levenshtein.ratio(brand_name, search_result) >= 0.3:
                        true_ += 1
                        break
                    img_padding = imgPadding(imgpath)
                    search_result, brand_name = googleSearch(img_padding, dir_)
                    if Levenshtein.ratio(brand_name, search_result) >= 0.3:
                        true_ += 1
                        break
                    img_crop1, img_crop2 = imgCrop(imgpath)
                    search_result, brand_name = googleSearch(img_crop1, dir_)
                    if Levenshtein.ratio(brand_name, search_result) >= 0.3:
                        true_ += 1
                        break
                    search_result, brand_name = googleSearch(img_crop2, dir_)
                    if Levenshtein.ratio(brand_name, search_result) >= 0.3:
                        true_ += 1
                        break
                    else:
                        errorpath = erroroot + "/" + dir_
                        if not os.path.exists(errorpath):
                            os.makedirs(errorpath)
                        errorimgpath = errorpath + "/" + str(search_result) + str(count) + ".png"
                        img = cv.imread(imgpath)
                        cv.imwrite(errorimgpath, img)
    print(count)
    print(true_)
    return (true_ / count)


if __name__ == '__main__':
    accur = testYOLO()
    print(accur)
