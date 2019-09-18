from PIL import Image
import pytesseract
import cv2 as cv

img_path = 'images/netflex.png'
img = cv.imread(img_path)
text = pytesseract.image_to_string(Image.fromarray(img))
print(text)