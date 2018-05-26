from keras.preprocessing import image
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from keras.preprocessing import image

from IPython.display import display 
from PIL import Image

#model = applications.resnet50.ResNet50(weights='imagenet', include_top=True, pooling='avg')

model = VGG16(weights='imagenet', include_top=True)
model.summary()
#model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)


rootDir = "C:/Users/Kevin D'Cruz/Keras_Image_Classification/train"

## path array to all images
path = []

for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        ## crreating image path
        imagepath = os.path.join(rootDir, fname) ## concatenating
        path.append(imagepath)
        #print('\t%s' % fname)

#img_path = "C:/Users/Kevin D'Cruz/Keras_Image_Classification/train/cat.0.jpg"

allfeature = []
    
for i in path:
    print(i)
    img = image.load_img(i, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    #k = np.squeeze(vgg16_feature)
    allfeature.append(vgg16_feature)
    
    
    
    
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
