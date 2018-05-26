# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:47:46 2018

@author: Kevin D'Cruz
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd
import urllib
import cv2

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19, preprocess_input


#------------Model Loading---------------------------------
model = ResNet50(weights='imagenet')
target_size = (224, 224)
#model.summary()

#---------------------Image if given from Local path--------------------
"""
img_path = "C:/Users/Kevin D'Cruz/Desktop/Maisie.jpg"
img = image.load_img(img_path, target_size=(224, 224))
#img

"""
#---------------------URL Image---------------------------------------
url = input('Enter a URL: ')
#url = "https://image.freepik.com/free-photo/hrc-siberian-tiger-2-jpg_21253111.jpg"
url_response = urllib.request.urlopen(url) #extract the contents of the URL
image_url=image.load_img(url_response, target_size=(224,224))#img = image.load_img(img, target_size=(224, 224))

#url_response = urllib.request.urlopen(load) #extract the contents of the URL
#img_array = np.array(bytearray(url_response.read()), dtype=np.uint8) #convert it into a numpy array
#img_array

#------------------Prediction function----------------------------
def predict(model, image_url, target_size, top_n=3):

  x = image.img_to_array(image_url)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)  
  return decode_predictions(preds, top=top_n)[0]
 
    
#------------------------Plotting--------------------------------------

prediction = predict(model, image_url, target_size)
prediction=pd.DataFrame(np.array(prediction).reshape(3,3), columns = list("abc"))
print("This picture has the highest possibility of a "+'\033[1m' '\033[4m'+prediction.b[0])

plt.barh(prediction.b, prediction.c, align='center', color='gray', edgecolor='black', height=0.4)
#order = list((range(len(prediction.c))))
#plt.yticks(order)
plt.ylabel("Predicted Outcomes", color='blue')
plt.xlabel("Output Probabilities", color='blue')
image_url
plt.show()

