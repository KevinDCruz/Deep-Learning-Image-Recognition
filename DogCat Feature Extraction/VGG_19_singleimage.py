# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:41:49 2018

@author: Kevin D'Cruz
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:32:02 2018

@author: Kevin D'Cruz
"""

from keras.preprocessing import image
from keras.models import Model
from keras import applications
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np
import os
from keras.preprocessing import image
from IPython.display import display
from PIL import Image


model = VGG19(weights='imagenet')  # , include_top=True)
model.summary()
model_extractfeatures = Model(input=model.input, output=model.get_layer('fc1').output)


img_path = "C:/Users/Kevin D'Cruz/Keras_Image_Classification/train/cat.0.jpg"
img = image.load_img(img_path, target_size=(224, 224))

img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg19_features = model_extractfeatures.predict(img_data)
