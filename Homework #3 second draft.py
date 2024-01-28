#!/usr/bin/env python
# coding: utf-8

# ## Homework 3
# # Dassilva Petit
# CS6682 Machine Learning
# 
# Troy University
# 

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# # Data analysis 

# In[2]:


weatherAUS= pd.read_csv('C:/Users/dassi/Downloads/weatherAUS.CSV')


# In[3]:


weatherAUS.head()


# After review the Data, I realize there was a lot of string in the date:
# 
#     1. I need the prep the data
#     
#     2. I need to Drop the column the I don't need. 

# In[4]:


weatherAUS.info()


# In[5]:


sns.countplot(x='Date', data=weatherAUS)


# In[6]:


sns.countplot(x='Date', hue='RainTomorrow', data=weatherAUS)


# In[7]:


weatherAUS.describe()


# I need to find out which column has missing. if 80% of the data was missing, I was going to Drop the column

# In[8]:


sns.heatmap(weatherAUS.isnull(), cbar=False)


# I run into a problem at the end of the Logistic Regression.
# 
# 1. I drop Date column and run it with everyting else I couldn't get through
# 
# 2. I came back and drop all the column date had missing data, and it didn't work.
# 
# 3. I try to covert the string in all the date, and it didn't work. after exhausted all possibilities. 
# 

# In[9]:


weatherAUS.drop(['MaxTemp', 'WindGustSpeed', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Location', 'RainToday', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Date'], axis=1, inplace=True)


# In[10]:


weatherAUS.head()


# In[11]:


weatherAUS.RainTomorrow = [1 if value == "No" else 0 for value in weatherAUS.RainTomorrow]


# In[12]:


weatherAUS.head()


# In[13]:


weatherAUS["RainTomorrow"] = weatherAUS['RainTomorrow'].astype("category", copy=False)
weatherAUS["RainTomorrow"].value_counts().plot()


# In[14]:


weatherAUS.RainTomorrow


# In[15]:


y = weatherAUS["RainTomorrow"]
X = weatherAUS.drop(["RainTomorrow"], axis = 1)


# In[16]:


y


# In[17]:


X


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


x_training_data = X_train
y_training_data = y_train


# In[20]:


x_training_data.columns = x_training_data.columns.astype(str)


# In[21]:


x_training_data = x_training_data.values


# In[22]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# I try to fix the data:
# 
# 1. from sklearn.impute import SimpleImputer
# 
# 2. sklearn.preprocessing import LabelEncoder
# 
# # I was unable to find the problem

# In[23]:


model.fit(x_training_data, y_training_data)


# In[ ]:


predictions = model.predict(x_test_data)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test_data, predictions


# In[ ]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test_data, predictions))


# In[ ]:





# In[ ]:




