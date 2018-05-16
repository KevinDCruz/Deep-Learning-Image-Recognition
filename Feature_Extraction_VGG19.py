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

model = VGG19(weights='imagenet')#, include_top=True)
model.summary()
model_extractfeatures = Model(input=model.input, output=model.get_layer('fc1').output)

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

allfeature = []
    
for i in path:
    print(i)
    img = image.load_img(i, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg19_feature = model_extractfeatures.predict(img_data)
    allfeature.append(vgg19_feature)
    #allfeature
    allfeature_list_np = np.array(allfeature)
    allfeature_list_np


fw = open("all_features.txt",'w')
for feature in allfeature:
    fw.write(str(feature))
    
fr = open('all_features.txt', 'r')
print(fr.read())

fr.close()

type(allfeature_list_np)

import numpy as np
np.savetxt("all_features.csv", allfeature_list_np, delimiter=",")


import pandas as pd
df = pd.DataFrame(allfeature)

df[0].str.strip('[[]]')
df

df.str.get(0)
#np.savetxt(r'all_features.txt', allfeature_list_np.values, fmt='%d')
df.to_csv("all_features.csv", index=False, sep='\t', encoding='utf-8')
    

with open("all_features.csv", "wb") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(allfeature)

    
with open("all features.txt", 'r') as file:
    allfeature = [line.rstrip('\n') for line in file]

print(allfeature)    
    #k = np.squeeze(vgg16_feature)
    
    
import csv

with open('all_features2.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(allfeature)   