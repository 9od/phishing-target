import os
import shutil
import cv2 as cv
from PIL import Image
import pytesseract


def dataWash_gentle(path):
    for root, dirs, files in os.walk(path):
        count = 0
        for dir_ in dirs:
            print(str(dir_))
            count+=1
            # if count<18000:
            #     continue
            # if count>2000:
            #     break
            imgpath = os.path.join(root, dir_) + "/shot.png"
            htmlpath = os.path.join(root, dir_) + "/html.txt"
            if os.path.exists(htmlpath) and os.path.exists(imgpath):
                with open(htmlpath) as file_object:
                    content = file_object.read()
                flag = ((os.path.getsize(imgpath) / 1024 > 10)
                        and (not "NUS" in content)
                        and (not "domain" in content)
                        and (not "sorry" in content)
                        and (not "Suspended" in content)
                        and (not "The website you were trying to reach is temporarily unavailable" in content))
                if flag:
                    img = cv.imread(imgpath)
                    text = pytesseract.image_to_string(Image.fromarray(img))
                    if not ("404" in text or text == "" or "not found" in text):
                        oldpath = "D:\\yuxuan\\phishwebsGet\\Database\\" + str(dir_)
                        newpath = "D:\\yuxuan\\phishwebsGet\\newDatabase_gentle\\" + str(dir_)
                        try:
                            shutil.copytree(oldpath, newpath)
                            break
                        except Exception as e:
                            print(e)


if __name__ == '__main__':
    path = "D:\\yuxuan\\phishwebsGet\\Database"
    dataWash_gentle(path)