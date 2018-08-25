
# coding: utf-8

# In[37]:

from sklearn import datasets
import pandas as pd
import numpy as np


# In[38]:

iris = datasets.load_iris()
x =pd.DataFrame(iris['data'],columns= iris['feature_names'])
y = pd.DataFrame(iris['target'], columns =['feature_names'])
data = pd.concat([x,y],axis=1)
data.head(3)


# In[39]:

s2 = pd.Series([1,2,3], index=['a','b','c'])
s2['a']


# In[40]:

iris


# In[41]:

np.random.randn(6,4)
df= pd.DataFrame(np.random.randn(6,4), columns=['a','b','c','d'])
df


# In[42]:

data.add(1).head(3)


# In[43]:

data[['sepal length (cm)']].head(5)


# In[44]:

iris.keys()


# In[45]:

print(iris['target_names'])


# In[46]:

x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
x


# In[47]:

iris['target']


# In[48]:

y = pd.DataFrame(iris['target'], columns =['feature_names'])
y


# In[49]:

data = pd.concat([x,y],axis=1)
data


# In[50]:

print(dict(enumerate(iris['target_names'])))


# In[51]:

y1 = pd.DataFrame(pd.Series(iris['target']).map(dict(enumerate(iris['target_names']))), columns=['target_name'])
y1


# In[55]:

data = pd.concat([x,y1],axis=1)
data.head(3)


# In[56]:

data.groupby(by ='target_name').mean()


# In[57]:

data.groupby(by ='target_name').sum()


# In[60]:

data.groupby(by='target_name').apply(lambda x:x.sort_values(by=['sepal length (cm)'],ascending = False)[1:2])


# In[ ]:



