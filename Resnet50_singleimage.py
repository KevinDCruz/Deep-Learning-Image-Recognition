# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:09:53 2018

@author: Kevin D'Cruz
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:32:02 2018

@author: Kevin D'Cruz
"""

from keras.preprocessing import image
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from keras.preprocessing import image

from IPython.display import display
from PIL import Image

model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.summary()

img_path = "C:/Users/Kevin D'Cruz/Keras_Image_Classification/train/cat.0.jpg"

# path array to all images

img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)
resnet50_feature = model.predict(img_data)
