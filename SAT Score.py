#!/usr/bin/env python
# coding: utf-8

# # ============   Assignment 1  ============
# 
# •	The SAT is considered one of the best estimators of intellectual capacity and capability
# •	Almost all collage (ex: USA) are using the SAT as a proxy for admission
# •	The SAT stood the test of time
# SAT = Critical reading + Mathematics + Writing
# GPA = Grade Point Average
# 
# Your task is to
# Creating a linear regression which predicts GPA based on the SAT score
# Dataset to use: 
# SAT score.csv
# 
# 

# ================================================================================

# # GPA from SAT score

# ## 1.Importing Python Packages

# #### a.Import numpy package for array operations

# In[ ]:


import numpy as np


# #### b.Check if numpy importing is ok 

# In[ ]:


print(np.__version__)


# #### c.Import pandas package for data manipulations

# In[ ]:


import pandas as pd


# #### d.Check if pandas importing is ok 

# In[ ]:


print(pd.__version__)


# #### e.Import sklearn package for machine learning algorithms

# In[ ]:


import sklearn


# #### f.Check if sklearn importing is ok

# In[ ]:


print(sklearn.__version__)


# ================================================================================

# ## 2.Importing Data

# #### a.Import using pandas read_csv function

# In[ ]:


data=pd.read_csv("SAT score.csv")


# #### b.Check if data importing is ok

# In[ ]:


data


# ================================================================================

# ## 3A. Store MinTemp data into input variable x

# In[ ]:


x=np.array(data['SAT']).reshape((-1,1))



# ### a. view input variable x

# In[ ]:


x


# ## 3B. Store MaxTemp data into output variable y

# In[ ]:


y=np.array(data['GPA'])


# 
# a. view output variable y

# In[ ]:


y


# ================================================================================

# ## 4.splitting data into training and testing

# #### a.Importing train test split function

# In[ ]:


from sklearn.model_selection import train_test_split


# #### b.applying train test split function on data

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# ================================================================================

# ## 5.Importing Machine Learning algorithm

# In[ ]:


from sklearn.linear_model import LinearRegression


# ================================================================================

# ## 6.Training using machine learning

# ### a. calling ml algorithm to model variable

# In[ ]:


model=LinearRegression()


# ### b. training ml algorithm 

# In[ ]:


model.fit(x_train,y_train)


# ================================================================================

# ## 7.Making Predictions to test using test data

# In[ ]:


y_pred=model.predict(x_test)


# ================================================================================

# ## 8.Comparing predictions with available original data 

# In[ ]:


print('Predicted GPA:',' '*10,'Original GPA')
for i in range(1,10):
    print(round(y_pred[i]),' '*29,round(y_test[i]))


# ================================================================================

# ## 9.Making new predictions with new input data 

# In[ ]:


new_input=int(input("Enter value to input :"))


# In[ ]:


new_pred=model.predict([[new_input]])


# In[ ]:


print(" GPA of the SAT score of {} is => {} ".format(new_input,round(new_pred[0])))

