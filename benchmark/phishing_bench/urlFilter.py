import os
import cv2
import pytesseract
from PIL import Image
from selenium import webdriver
from urllib3.exceptions import MaxRetryError
from selenium.common.exceptions import TimeoutException


# Get the urls
def urlsGet():
    readfile = "remained.txt"
    f = open(readfile, "r", encoding="utf8")
    lines = f.readlines()
    i = 0
    targets = []
    urls = []
    for line in lines:
        i += 1
        if i % 2 == 0:
            urls.append(line)
        else:
            targets.append(line.strip('\n'))
    return urls, targets


# Recursively delete a folder
def delDirs(path_):
    if (os.path.isdir(path_)):
        for p in os.listdir(path_):
            delDirs(os.path.join(path_, p))
        if (os.path.exists(path_)):
            os.rmdir(path_)
    else:
        if (os.path.exists(path_)):
            os.remove(path_)


# Main function
def main():
    urls, targets = urlsGet()
    for k in range(0, len(urls)):
        try:
            print(str(k))
            url = urls[k]
            tag = targets[k]

            driver = webdriver.PhantomJS()
            driver.set_page_load_timeout(10)
            driver.set_script_timeout(10)
            driver.maximize_window()
            driver.get(url)

            ospath = './remaining_url/' + tag
            if not os.path.exists(ospath):
                os.makedirs(ospath)
            imgpath = ospath + '/' + str(tag) + '.png'
            driver.save_screenshot(imgpath)
            img = cv2.imread(imgpath)
            text = pytesseract.image_to_string(Image.fromarray(img))
            if "404" in text or text == "" or "not found" in text:
                delDirs(ospath)
            driver.quit()

        except TimeoutException:
            driver.quit()
        except MaxRetryError:
            driver.quit()
        except Exception as e:
            driver.quit()


if __name__ == '__main__':
    main()
