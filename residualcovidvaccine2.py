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
import pytesseract as tess

import numpy as np

options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920x1080')
options.add_argument('--disable-gpu')  # Last I checked this was necessary.
options.add_argument("--disable-notifications")

chrome = webdriver.Chrome(
    'C:/Users/s1064/Downloads/chromedriver_win32/chromedriver',
    chrome_options=options)
chrome.get(
    'http://www.tcmg.com.tw/register/register_1_detail_2_covid19remainder.php')
# 身分證
idNumber = chrome.find_element_by_id("idNumber")
# 性別
sexradioF = chrome.find_element_by_id("sexradioF")
# 姓名
ptName = chrome.find_element_by_id("ptName")

# 出生日期(民國)
bYear = chrome.find_element_by_id("bYear")

# 出生日期(月)
bMonth = chrome.find_element_by_id("bMonth")

# 出生日期(日)
bDay = chrome.find_element_by_id("bDay")

# 手機號碼
telNumber = chrome.find_element_by_id("telNumber")

# 疫苗種類
#typelist = chrome.find_element_by_id("typelist")

# 驗證碼
# txt_input = chrome.find_element_by_id("txt_input")

# applybtn = chrome.find_element_by_id("applybtn")

imgCode = chrome.find_element_by_xpath(
    '//*[@id="content_d"]/div[2]/div[4]/div[2]/div[1]/img')
print(imgCode.location)
print(imgCode.size)

left = imgCode.location['x']
right = imgCode.location['x'] + imgCode.size['width']
top = imgCode.location['y']
bottom = imgCode.location['y'] + imgCode.size['height']

print(left, right, top, bottom)

chrome.save_screenshot('test1.png')
img = Image.open('test1.png')
img = img.convert('RGB')
img = img.crop((left, top, right, bottom))

img.save('captcha1.jpg', 'jpeg')

Image.open('captcha1.jpg')
img = cv2.imread('captcha1.jpg')

img2 = img.copy()
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.imshow(img2)

ret, binary = cv2.thre
contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE,
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
    roi = img2[y:y + h, x:x + w]
    thresh = roi.copy()
    a = fig.add_subplot(1, len(ary), id + 1)
    # res = cv2.resize(thresh, (50, 50))
    # cv2.imwrite('%d.png' (id), res)
    plt.imshow(thresh)

idNumber.send_keys('H224923457')
ptName.send_keys('蘇姸寧')
bYear.send_keys('88')
bMonth.send_keys('04')
bDay.send_keys('10')
telNumber.send_keys('0981998161')
sexradioF.send_keys('F')
sexradioF.submit()
time.sleep(3)

# %%
