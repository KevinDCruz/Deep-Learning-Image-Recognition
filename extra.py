# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:57:51 2018

@author: Kevin D'Cruz
"""

for i in kl:
    print(i)
    img = image.load_img(i, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg19_feature = model_extractfeatures.predict(img_data)
    flat = vgg19_feature.flatten()
    allfeature2.append(vgg19_feature)
    

