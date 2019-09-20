#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# In[80]:


### load the dataset


# In[81]:


dataset=pd.read_csv('train.csv',nrows=14000)
print(dataset.head())


# In[82]:


plt.scatter(dataset['Angle of Rays'],dataset['Eclipse Duration (m)'])


# In[ ]:





# In[83]:


### create a plot with gamma and eclipse duration


# In[84]:


plt.scatter(dataset['Gamma'],dataset['Eclipse Duration (m)'])


# In[85]:


### create a plot with magnitude and eclipse duration


# In[86]:


plt.scatter(dataset['Magnitude 1'],dataset['Eclipse Duration (m)'])


# In[87]:


dataset=dataset.drop('Lunation Number',axis=1)


# In[88]:


dataset.head()


# In[89]:


#dataset=dataset.drop('Date of occurrence',axis=1)
dataset=dataset.drop('Day of Month',axis=1)
dataset=dataset.drop('Time of occurrence',axis=1)
dataset=dataset.drop('Time Difference',axis=1)
dataset=dataset.drop('Latitude',axis=1)
dataset=dataset.drop('Longitude',axis=1)


# In[90]:


### remove all unnecessary coluums of data


# In[91]:


dataset.head(10)


# In[92]:


dataset=dataset.drop('Eclipse Type 1',axis=1)


# In[93]:


dataset.head()


# In[94]:


from sklearn.model_selection import train_test_split
x=dataset[['Angle of Rays','Gamma','Magnitude 1']]
y=dataset['Eclipse Duration (m)']
X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[95]:


X_train


# In[96]:


x_test


# In[97]:


clf=LinearRegression()
clf.fit(X_train,Y_train)


# In[98]:


### applied linear regression


# In[99]:


clf.predict(x_test)


# In[100]:


y_test


# In[101]:


clf.score(x_test,y_test)


# In[102]:


clf1=Lasso(alpha=0.001)
clf1.fit(X_train,Y_train)


# In[103]:


clf.predict(x_test)


# In[104]:


clf1.score(x_test,y_test)


# In[105]:


## got the accuracy 81.7 percent in both linear regression lasso regression


# In[106]:


import seaborn as sns; sns.set(style="darkgrid")
sns.relplot(x='Gamma', y='Eclipse Duration (m)', data=dataset)
sns.lmplot(x='Gamma', y='Eclipse Duration (m)', data=dataset)


# In[107]:


sns.relplot(x='Magnitude 1', y='Eclipse Duration (m)', data=dataset)
sns.lmplot(x='Magnitude 1', y='Eclipse Duration (m)', data=dataset)


# In[108]:


sns.relplot(x='Angle of Rays', y='Eclipse Duration (m)', data=dataset)


# In[109]:


from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = plt.axes(projection='3d')
xline=dataset['Gamma']
yline=dataset['Eclipse Duration (m)']
zline=dataset['Magnitude 1']
ax.plot3D(xline, yline, zline, 'blue')


# In[110]:


ax.scatter3D(xline, yline, zline, c=zline, cmap='hsv')


# In[128]:


X=dataset[['Magnitude 1']]
Y=dataset['Eclipse Duration (m)']
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.3,shuffle=True)
clf.fit(X_train,Y_train)


# In[112]:


y_test


# In[129]:


clf.score(x_test,y_test)


# In[130]:


clf1.fit(X_train,Y_train)


# In[131]:


clf1.predict(x_test)
clf1.score(x_test,y_test)


# In[132]:


x=dataset[['Angle of Rays','Gamma','Magnitude 1']]
y=dataset['Eclipse Duration (m)']
X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.2)
reg = Ridge(alpha=1)
reg.fit(X_train,Y_train)


# In[133]:


reg.predict(x_test)


# In[134]:


reg.score(x_test,y_test)


# In[ ]:





# In[119]:


dataset.head()


# In[120]:


df=pd.read_csv('test.csv')
df.head()


# In[123]:


x_test121=df[['Angle of Rays','Gamma','Magnitude 1']]
x_test121


# In[135]:


data=x_test121 #an array of dim (188,3)

X=data[:,0:2]
y=data[:,2]
m,n=np.shape(X)
y=y.reshape(m,1)
x=np.c_[np.ones((m,1)),X]
theta=np.zeros((n+1,1))


# In[ ]:




