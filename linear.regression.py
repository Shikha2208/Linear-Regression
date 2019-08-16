#!/usr/bin/env python
# coding: utf-8

# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[29]:


import pandas as pd


# In[30]:


import matplotlib.pyplot as plt


# In[31]:


plt.rcParams['figure.figsize']=(20.0,10.0)


# In[32]:


data=pd.read_csv('E:/SHIKHA_FOLDER/linearregression/headbrain.csv')


# In[33]:


data.head()


# In[34]:


print(data.shape)


# In[35]:


#collecting X and Y
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values


# In[36]:


#mean of x and y
mean_x=np.mean(X)


# In[37]:


mean_y=np.mean(Y)


# In[38]:


#total number of values
m=len(X)


# In[39]:


#using formula to calculate b1 and b2
numer=0
denom=0
for i in range(m):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
b1=numer/denom
b0=mean_y-(b1*mean_x)
print(b1,b0)


# In[17]:


#plotting values and regression lines
max_x=np.max(X)+100
min_x=np.min(X)-100
#clculating line values x and y
x=np.linspace(min_x,max_x,1000)
y=b0+b1*x

#plotting line

plt.plot (x,y,color='#58b970', label="Regression Line")

#plt scatter point
plt.scatter(X,Y,c='#ef5423', label ='Scatter plot')
plt.xlabel("Head Size in Cm3")
plt.ylabel("Brin weight in  grams")
plt.legend()
plt.show


# In[18]:


print(b1,b0)


# In[19]:


#to check error using Rsquare value
ss_t=0
ss_r=0
for i in range(m):
    y_pred=b0+b1*X[i]
    ss_t+=(Y[i]-mean_y)**2
    ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


from sklearn.metrics import mean_squared_error


# In[22]:


X=X.reshape((m,1))


# In[23]:


#creating model
reg=LinearRegression()


# In[24]:


#fitting training data
reg=reg.fit(X,Y)


# In[25]:


#Yprediction
y_pred=reg.predict(X)


# In[26]:


#calculating R2 score

r2_score=reg.score(X,Y)


# In[27]:


print(r2_score)


# In[ ]:




