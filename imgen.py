from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import random
import time
import requests
import os

def imgen(inp,s):
    browser = webdriver.Chrome(fr"{os.getcwd()}/chromedriver.exe")

    clss = pipeline("zero-shot-classification", model="typeform/mobilebert-uncased-mnli", tokenizer="typeform/mobilebert-uncased-mnli")
    js=clss(inp, ["angry","happy","funny","free","relax","positive","classic","modern"])
    style=""
    for l, s in zip(js["labels"], js['scores']):
        if s>0.2: style=style+l+", "
        else:
            style=style[:-2]+" art"
            break
    browser.get(f"https://www.google.com/search?q={style}&tbm=isch")
    time.sleep(2)
    browser.find_element_by_xpath('//a[@jsname="sTFXNd"]').click()
    time.sleep(6)
    with open("style.png", "wb")as f:
        f.write(requests.get(browser.find_elements_by_xpath('//img[@jsname="HiaYvf"]')[0].get_attribute("src")).content)
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/mobilebert-finetuned-pos")
    model = AutoModelForTokenClassification.from_pretrained("mrm8488/mobilebert-finetuned-pos")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    inp=nlp(inp)

    no=0
    for ip in inp:
        if ip["entity"]=="NN": 
            no+=1
            klm=ip["word"]
            browser.get(f"https://www.google.com/search?q={klm} transparent&tbm=isch")
            time.sleep(2)
            browser.find_element_by_xpath('//a[@jsname="sTFXNd"]').click()
            time.sleep(6)
            with open((["test", "bg"][no-1]+".png"), "wb")as f:
                f.write(requests.get(browser.find_elements_by_xpath('//img[@jsname="HiaYvf"]')[0].get_attribute("src")).content)
        if no>1: break
    browser.close()


    img = cv2.imread('test.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    mask = 255 - mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    cv2.imwrite('rebg.png', result)

    background = Image.open("bg.png")
    foreground = Image.open("rebg.png")

    background.paste(foreground, (0, 0), foreground)
    background.save("rebg.png")

    def load_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
       
    content_image = load_image('rebg.png')
    style_image = load_image('style.png')
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    cv2.imwrite(f'rebg{s}.png', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

print("I will try to picture your sentence")
for i in range(1,10,2):#i love odd numbers
    imgen(input("your sentence"),i)
