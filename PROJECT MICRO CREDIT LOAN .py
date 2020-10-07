#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib


# # DATASET  ORIGINAL FILE

# In[2]:


df=pd.read_csv('Data file.csv')


# In[3]:


df.shape


# # DATASET FILE AFTER DATA CLEANING

# In[4]:


df1=pd.read_csv('Data file3.csv')


# In[5]:


df1.shape


# # PERCENTAGE OF DATA LOST

# In[36]:


(209593-193576)/209593*100


# In[6]:


df2 = df1.drop(columns=['Unnamed: 0','msisdn','pcircle', 'pdate'])
df2.head()


# # CORRELATION MATRIX AFTER DATA CLEANING

# In[34]:


#correlation matrix
corr = df2.corr()


# In[35]:


#to show the correlation in diagramatic 
import seaborn as sns
plt.figure(figsize=(15,12))     #(column,row)
sns.heatmap(corr,cmap='RdYlGn')


# In[ ]:





# In[ ]:





# In[7]:


from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# # 1. LOGISTIC REGRESSION

# In[8]:


#Importing libraties and classes
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[9]:


# Train - Test split
#just drop the outcome columns
#specify  input and output attributes
#X is the input and y is the output
X = df2.drop(columns=['label'], axis=1)
y = df2['label']


# In[10]:


#Importing libraries and classes
#Dividing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[11]:


#Training the model
model.fit(X_train,y_train)


# In[12]:


#Training Accuracy
model.score(X_train,y_train)


# In[13]:


#Testing Accuracy
model.score(X_test,y_test)


# In[14]:


expected = y_test
predicted = model.predict(X_test)


# In[15]:


#Import librarires and classes
from sklearn import metrics


# In[16]:


#Printing the Report
print(metrics.classification_report(expected,predicted))


# # 2. NAIVE BAYES

# In[17]:


#Importing libraties and classes
from sklearn.naive_bayes import GaussianNB
model2 = GaussianNB()


# In[18]:


#Training the model
model2.fit(X_train,y_train)

expected2 = y_test
predicted2 = model2.predict(X_test)


# In[19]:


#Import librarires and classes
from sklearn import metrics

#Printing the Report
print(metrics.classification_report(expected2,predicted2))

#Out of outcomes,were right and were wrong similarly,For 'N',were right and were wrong
print(metrics.confusion_matrix(expected2,predicted2))


# # 3. STOCHASTIC GRADIENT DESCENT

# In[20]:


#Importing libraties and classes
from sklearn.linear_model import SGDClassifier
model3 = SGDClassifier(loss='modified_huber',shuffle=True,random_state=0)


# In[21]:


#Training the model
model3.fit(X_train,y_train)

expected3 = y_test
predicted3 = model3.predict(X_test)


# In[22]:


#Import librarires and classes
from sklearn import metrics

#Printing the Report
print(metrics.classification_report(expected3,predicted3))

#Out of outcomes,were right and were wrong similarly,For 'N',were right and were wrong
print(metrics.confusion_matrix(expected3,predicted3))


# # 4. K NEIGHBOUR CLASSIFIER

# In[23]:


#Importing libraties and classes
from sklearn.neighbors import KNeighborsClassifier
model4 = KNeighborsClassifier(n_neighbors=15)


# In[24]:


#Training the model
model4.fit(X_train,y_train)


# In[25]:


expected4 = y_test
predicted4 = model4.predict(X_test)


# In[26]:


#Import librarires and classes
from sklearn import metrics

#Printing the Report
print(metrics.classification_report(expected4,predicted4))

#Out of outcomes,were right and were wrong similarly,For 'N',were right and were wrong
print(metrics.confusion_matrix(expected4,predicted4))


# # 5. DECISION TREE CLASSIFIER

# In[30]:


#Importing libraties and classes
from sklearn.tree import DecisionTreeClassifier
model5 = DecisionTreeClassifier(max_depth=10, random_state=101,max_features=None,min_samples_leaf=15)


# In[31]:


#Training the model
model5.fit(X_train,y_train)


# In[32]:


expected5 = y_test
predicted5 = model5.predict(X_test)


# In[33]:


#Import librarires and classes
from sklearn import metrics

#Printing the Report
print(metrics.classification_report(expected5,predicted5))

#Out of outcomes,were right and were wrong similarly,For 'N',were right and were wrong
print(metrics.confusion_matrix(expected5,predicted5))


# # 6. RANDOM FOREST CLASSIFIER

# In[27]:


#Importing libraties and classes
from sklearn.ensemble import RandomForestClassifier
model6 = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)


# In[28]:


#Training the model
model6.fit(X_train,y_train)

expected6 = y_test
predicted6 = model6.predict(X_test)


# In[29]:


#Import librarires and classes
from sklearn import metrics

#Printing the Report
print(metrics.classification_report(expected6,predicted6))

#Out of outcomes,were right and were wrong similarly,For 'N',were right and were wrong
print(metrics.confusion_matrix(expected6,predicted6))


# # RANDOM FOREST CLASSIFIER AND DECISION TREE CLASSIFIER BOTH ARE HAVIG GOOD ACCURACY OF 91%

# In[ ]:




