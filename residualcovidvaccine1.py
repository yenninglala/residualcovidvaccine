# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:07:30 2021

@author: s1064
"""
#%%
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# import pytesseract as tess
import numpy as np

options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920x1080')
options.add_argument('--disable-gpu')  # Last I checked this was necessary.
options.add_argument("--disable-notifications")

chrome = webdriver.Chrome(
    'C:/Users/s1064/Downloads/chromedriver_win32/chromedriver',
    chrome_options=options)
chrome.get('https://rmsvc.sph.org.tw/Residue.aspx')
# 身分證
idno = chrome.find_element_by_id("idno")

# 姓名
pname = chrome.find_element_by_id("pname")

# 出生日期
rmsdata = chrome.find_element_by_id("rmsdata")

# 手機號碼
cellphone = chrome.find_element_by_id("cellphone")

# 聯絡電話
lineid = chrome.find_element_by_id("lineid")

# 疫苗種類
#typelist = chrome.find_element_by_id("typelist")

# 驗證碼
# txt_input = chrome.find_element_by_id("txt_input")

# applybtn = chrome.find_element_by_id("applybtn")

imgCode = chrome.find_element_by_id('imgCode')
print(imgCode.location)
print(imgCode.size)

left = imgCode.location['x']
right = imgCode.location['x'] + imgCode.size['width']
top = imgCode.location['y']
bottom = imgCode.location['y'] + imgCode.size['height']

print(left, right, top, bottom)

chrome.save_screenshot('test.png')
img = Image.open('test.png')
img = img.convert('RGB')
img = img.crop((left, top, right, bottom))

img.save('captcha.jpg', 'jpeg')

Image.open('captcha.jpg')
img = cv2.imread('captcha.jpg')

dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(dst)
# kernel_first = np.ones((2,2),np.uint8)
# erosion = cv2.erode(dst,kernel_first,iterations=1)
# # 中值濾波
# blurred = cv2.medianBlur(erosion,5)
# blurred = cv2.GaussianBlur(blurred,(5,5),0)
# edged = cv2.Canny(blurred,30,80)
# #cv2.dilate() 的第一個參數為二值化的影像， 第二個參數為使用的捲積 kernel，第三個參數為迭代次數(預設為1)，
# dilation = cv2.dilate(img,kernel_first,iterations=1)

# dilation2 = dilation.copy()
# dilation2 = cv2.cvtColor(dilation2,cv2.COLOR_BGR2GRAY)
# plt.imshow(dilation2)

# def recognize_text(src):
#     # 转成灰度图像
#     gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#     # 二值化
#     ret, binary = cv2.threshold(
#         gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     # 结构元素  去掉竖直的线
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
#     # 开操作
#     bin1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#     # 去掉横线
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#     # 开操作
#     open_out = cv2.morphologyEx(bin1, cv2.MORPH_OPEN, kernel)
#     cv2.imshow("binary-image", open_out)
#     # 黑色背景 变成白色背景
#     cv2.bitwise_not(open_out, open_out)
#     # fromarray 二维数组
#     textImage = Image.fromarray(open_out)
#     print(textImage)
#     # 图片转成字符串
#     text = tess.image_to_string(textImage)
#     print("识别结果: %s" % text)

kernel = np.ones((2, 2), np.uint8)
erosion = cv2.erode(dst, kernel, iterations=1)
# 中值濾波
blurred = cv2.medianBlur(erosion, 5)
blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 80)
#cv2.dilate() 的第一個參數為二值化的影像， 第二個參數為使用的捲積 kernel，第三個參數為迭代次數(預設為1)，
dilation = cv2.dilate(img, kernel, iterations=1)

dilation2 = dilation.copy()
dilation2 = cv2.cvtColor(dilation2, cv2.COLOR_BGR2GRAY)
plt.imshow(dilation2)

contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours],
              key=lambda x: x[1])

ary = []
for (c, _) in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    print(x, y, w, h)
    if w > 15 and h > 15:
        ary.append((x, y, w, h))

fig = plt.figure()
for id, (x, y, w, h) in enumerate(ary):
    roi = dilation2[y:y + h, x:x + w]
    thresh = roi.copy()
    a = fig.add_subplot(1, len(ary), id + 1)
    # res = cv2.resize(thresh,(50,50))
    # cv2.imwrite('%d.png'(id),res)
    plt.imshow(thresh)
# src = img

# recognize_text(src)
# cv2.waitKey(0)

idno.send_keys('')
pname.send_keys('')
rmsdata.send_keys('')
cellphone.send_keys('')
lineid.send_keys('')
lineid.submit()
time.sleep(3)

# %%
