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
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19, preprocess_input


#------------Model Loading---------------------------------
model = ResNet50(weights='imagenet')
target_size = (224, 224)
# model.summary()

img_path = "C:/Users/Kevin D'Cruz/Desktop/Maisie.jpg"
img = image.load_img(img_path, target_size=(224, 224))


def predict(model, img, target_size, top_n=3):
    """Run model prediction on image
    Args:
      model: keras model
      img: PIL format image
      target_size: (w,h) tuple
      top_n: # of top predictions to return
    Returns:
      list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=top_n)[0]


prediction = predict(model, img, target_size)
prediction = pd.DataFrame(np.array(prediction).reshape(3, 3), columns=list("abc"))
print("This picture has the highest possibility of a "+'\033[1m'+'\033[4m'+prediction.b[0])

plt.barh(prediction.b, prediction.c, align='center', color='gray', edgecolor='black', height=0.4)
#order = list((range(len(prediction.c))))
# plt.yticks(order)
plt.ylabel("Predicted Outcomes", color='blue')
plt.xlabel("Probabilities of Outcomes", color='blue')
plt.show()
