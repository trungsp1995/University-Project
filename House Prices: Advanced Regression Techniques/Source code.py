#!/usr/bin/env python
# coding: utf-8

# Name: The Trung Le
# 
# Student ID: a1784927

# In[1]:


import sys
assert sys.version_info >= (3, 5)


# In[2]:


import sklearn
assert sklearn.__version__ >= "0.20"


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# # 1. Data Analysis

# In[4]:


#Load train dataset and test dataset
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[5]:


#Show information about train dataset
data.info()


# In[6]:


#Show information about test dataset
test_data.info()


# In[7]:


#Show the first five data in train dataset
data.head()


# In[8]:


#Remove the Id column from train dataset
data = data.drop(['Id'], axis=1)


# In[9]:


#Print the name and number of numerical features in train dataset
numerical_feature = []
for feature in data.columns:
    if data[feature].dtypes != 'O':
        numerical_feature.append(feature)

print("Number of numerical features is: ", len(numerical_feature))
print("Numerical features are: ", numerical_feature)


# In[10]:


#Print the numerical features of first five data in train dataset
data[numerical_feature].head()


# In[11]:


#The correlation between Sale Price and other numerical features
correlation = data[numerical_feature].corr()
correlation['SalePrice'].sort_values(ascending = False)


# In[12]:


#Scatter plots about relationship between Sale Price and 5 features which have the highest correlation to Sale Price
data.plot(kind = 'scatter', x = 'OverallQual', y = 'SalePrice')
data.plot(kind = 'scatter', x = 'GrLivArea', y = 'SalePrice')
data.plot(kind = 'scatter', x = 'GarageCars', y = 'SalePrice')
data.plot(kind = 'scatter', x = 'GarageArea', y = 'SalePrice')
data.plot(kind = 'scatter', x = 'TotalBsmtSF', y = 'SalePrice')


# In[13]:


#Print the name and number of categorical features in train dataset
category_feature = []
for feature in data.columns:
    if data[feature].dtypes == 'O':
        category_feature.append(feature)

print("Number of category features is: ", len(category_feature))
print("Category features are: ",category_feature)


# In[14]:


#Print the category features of first five data in train dataset
data[category_feature].head()


# In[15]:


#Print the relationship between categorical features and median value of Sale Price 
for feature in category_feature:
    df = data.groupby(feature)['SalePrice'].median()
    df.plot(kind = 'bar')
    plt.xlabel(feature)
    plt.ylabel('Median Sale Price')
    plt.title(feature)
    plt.show()


# In[16]:


#Print the features which have missing values and the number of missing value
missing_data = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:20])
missing_data.columns = ['Missing Number']
print (missing_data)


# In[17]:


#Information about Sale Price
data.SalePrice.describe()


# In[18]:


#Histogram of Sale Price
data['SalePrice'].hist(bins = 50)
plt.xlabel('Sale Price')
plt.ylabel('House')
plt.title('Sale Price Distribution')
plt.show()


# # 2. Data Pre-processing

# In[19]:


#Copy the Id column in test dataset into different variable
test_Id = test_data['Id'].copy()


# In[20]:


#Delete the Id column in test dataset
test_data = test_data.drop(['Id'], axis=1)


# In[21]:


#Find the numerical features in test dataset
numerical_feature_test = []
for feature in test_data.columns:
    if test_data[feature].dtypes != 'O':
        numerical_feature_test.append(feature)


# In[22]:


#Filling missing values of categorical features by None
data[category_feature] = data[category_feature].fillna('None')

#Filling missing values of numerical features by 0
data[numerical_feature] = data[numerical_feature].fillna(0)

#Re-check the train dataset
data.info()


# In[23]:


#Filling missing values of categorical features by None
test_data[category_feature] = test_data[category_feature].fillna('None')

#Filling missing values of numerical features by 0
test_data[numerical_feature_test] = test_data[numerical_feature_test].fillna(0)

#Re-check the test dataset
test_data.info()


# In[24]:


for feature in category_feature:
    temp = data.groupby(feature)['SalePrice'].count()/len(data)
    temp_df = temp[temp>0.01].index
    data[feature] = np.where(data[feature].isin(temp_df),data[feature],'Rare_category')
    test_data[feature] = np.where(test_data[feature].isin(temp_df),test_data[feature],'Rare_category')


# In[25]:


#Re-ordering the unique values in each categorical features by compare the mean value of Sale Price
#Labelling the unique values in each categorical features by number
for feature in category_feature:
    labels_ordered = data.groupby(feature)['SalePrice'].mean().sort_values().index
    labels_ordered = {k:i for i,k in enumerate(labels_ordered,0)}
    data[feature] = data[feature].map(labels_ordered)
    test_data[feature] = test_data[feature].map(labels_ordered)


# In[26]:


test_data = test_data.fillna(0)


# # 3. Build model and evaluation

# In[27]:


#Log-transformation for Sale Price
SalePrice_log = np.log(data.SalePrice)

#Histogram of Sale Price after log-transformation
SalePrice_log.hist(bins = 50)
plt.xlabel('Sale Price transformation')
plt.ylabel('House')
plt.title('Sale Price Distribution')
plt.show()


# In[28]:


#Split the train dataset into train set and validation set
#First 70% train dataset is train set, last 30% train dataset is validation set
train_set = data[:1022]
validation_set = data[1022:]


# In[29]:


#Set the training features, training labels, validation features and validation labels
training_features = train_set.drop(['SalePrice'], axis=1)
training_labels = np.log(train_set['SalePrice'].copy())

validation_features = validation_set.drop(['SalePrice'], axis=1)
validation_labels = np.log(validation_set['SalePrice'].copy())

training_features.info()
test_data.info()


# In[30]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score

#Build Linear Regression model and fit into the train set
reg = linear_model.LinearRegression()
reg.fit(training_features,training_labels)
scores = cross_val_score(reg, training_features, training_labels, scoring = 'neg_mean_squared_error', cv = 10)
print ("the RMSE of Linear Regression model on train set is: ", np.sqrt(-scores).mean())


# In[31]:


from sklearn.ensemble import RandomForestRegressor
for_reg = RandomForestRegressor(n_estimators = 100)
for_reg.fit(training_features,training_labels)
scores2 = cross_val_score(for_reg, training_features, training_labels, scoring = 'neg_mean_squared_error', cv = 10)
print ("the RMSE of Random Forest Regressor model on train set is: ", np.sqrt(-scores2).mean())


# In[32]:


validation_labels_predict = reg.predict(validation_features)
print ("the RMSE of Linear Regression model on validation set is: ", np.sqrt(mean_squared_error(validation_labels, 
                                                                                           validation_labels_predict)))


# In[33]:


validation_labels_predict2 = for_reg.predict(validation_features)
np.sqrt(mean_squared_error(validation_labels, validation_labels_predict2))
print ("the RMSE of Random Forest Regressor model on validation set is: ", np.sqrt(mean_squared_error(validation_labels, 
                                                                                                      validation_labels_predict2)))


# # 4. Apply on test dataset

# In[34]:


#Fit Random Forest Regressor model on test dataset
test_labels_predict = for_reg.predict(test_data)


# In[35]:


#Exponential the prediction
test_labels_final = np.exp(test_labels_predict)


# In[36]:


#Make a data frame
final = pd.DataFrame()
final['Id'] = test_Id
final['SalePrice'] = test_labels_final
final.head()


# In[37]:


#Convert into csv file
final.to_csv('final.csv', index=False)


# In[ ]:




