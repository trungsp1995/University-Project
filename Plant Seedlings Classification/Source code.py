#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"


# In[2]:


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2

from keras.utils import np_utils
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# # 1. Data Analysis

# In[3]:


#Store the image from train dataset
path_train = 'train/*/*.png' 
file_train = glob(path_train)
train_Image = []
train_Label = []
i = 0
for image in file_train:
    train_Image.append(cv2.resize(cv2.imread(image), (50, 50)))
    train_Label.append(image.split("\\")[1])
    i = i + 1    
print ("Number of image in train dataset is:", i)
train_Image = np.asarray(train_Image) 
train_Label = pd.DataFrame(train_Label) 


# In[18]:


#Print the number of images of each species
print (train_Label[0].value_counts())


# # 2. Data Pre-Processing

# In[4]:


#Remove the background of images on train dataset
train_Image_preprocess = []
for image in train_Image:    
    blur_Image = cv2.GaussianBlur(image, (5, 5), 0)   
    hsv_Image = cv2.cvtColor(blur_Image, cv2.COLOR_BGR2HSV)  
    low_green = (25, 52, 72)
    high_green = (102, 255, 255)
    mask = cv2.inRange(hsv_Image, low_green, high_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   
    boolean_mask = mask > 0    
    empty_image = np.zeros_like(image, np.uint8)
    empty_image[boolean_mask] = image[boolean_mask]         
    train_Image_preprocess.append(empty_image)
train_Image_preprocess = np.asarray(train_Image_preprocess)
train_Image_preprocess = train_Image_preprocess / 255


# In[6]:


#Transform the categorical ouput to numerical output
label_endcoder = preprocessing.LabelEncoder()
label_endcoder.fit(train_Label[0])
transform_train_Label = label_endcoder.transform(train_Label[0])
train_Label_preprocess = np_utils.to_categorical(transform_train_Label)


# In[7]:


#Split the train dataset
train_input, validation_input, train_output, validation_output = train_test_split(train_Image_preprocess, 
                                                                                  train_Label_preprocess, 
                                                                                  test_size = 0.2, random_state = 89, 
                                                                                  stratify = train_Label_preprocess)


# In[8]:


#Create image generator and appling on train_input
image_generator = ImageDataGenerator(rotation_range = 180, horizontal_flip = True, vertical_flip = True)  
image_generator.fit(train_input)


# # 3. Build models

# In[9]:


#Create CNN model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (5, 5), input_shape = (50, 50, 3), activation = 'relu'))
model.add(keras.layers.BatchNormalization(axis = 3))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(filters = 128, kernel_size = (5, 5), activation = 'relu'))
model.add(keras.layers.BatchNormalization(axis = 3))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(filters = 256, kernel_size = (5, 5), activation = 'relu'))
model.add(keras.layers.BatchNormalization(axis = 3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(12, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[10]:


#Summary of the model
model.summary()


# In[11]:


#Fit the model to train set and validation set
model.fit(train_input, train_output, validation_data = (validation_input, validation_output), epochs = 50, batch_size = 50)


# # 4. Applying the model on test dataset

# In[12]:


#Store the image from test dataset
path_test = 'test/*.png' 
file_test = glob(path_test)
test_Image = []
test_file = []
i = 0
for image in file_test:
    test_Image.append(cv2.resize(cv2.imread(image), (50, 50)))
    test_file.append(image.split("\\")[1])
    i = i + 1
test_Image = np.asarray(test_Image)  
print ("Number of image in test dataset is:", i)


# In[13]:


#Remove the background of images on test dataset
test_Image_preprocess = []
for image in test_Image:    
    blur_Image = cv2.GaussianBlur(image, (5, 5), 0)   
    hsv_Image = cv2.cvtColor(blur_Image, cv2.COLOR_BGR2HSV)  
    low_green = (25, 52, 72)
    high_green = (102, 255, 255)
    mask = cv2.inRange(hsv_Image, low_green, high_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   
    boolean_mask = mask > 0    
    empty_image = np.zeros_like(image, np.uint8)
    empty_image[boolean_mask] = image[boolean_mask]         
    test_Image_preprocess.append(empty_image)
test_Image_preprocess = np.asarray(test_Image_preprocess)
test_Image_preprocess = test_Image_preprocess / 255


# In[15]:


#Predict the species of each image
output_predict = model.predict(test_Image_preprocess)
tmp = np.argmax(output_predict, axis=1)
predict_species = label_endcoder.classes_[tmp]


# In[16]:


#Assign these predictions to submission dataframe
submission = pd.DataFrame()
submission['file'] = test_file
submission['species'] = predict_species
submission.head()


# In[17]:


#Convert the submission to .csv file
submission.to_csv('submission.csv', index = False)

