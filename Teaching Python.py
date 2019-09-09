#!/usr/bin/env python
# coding: utf-8

# # Teaching Python

# ## Basics 

# In[28]:


40 + 2 


# See shortcuts in the short cut menu
# 

# In[29]:


#Importing package for computations
import numpy as np


# In[30]:


## Getting the result of first computation
Out[1]
x= Out[1]
x


# In[31]:


np.sin(90)


# In[32]:


#to check help
get_ipython().run_line_magic('pinfo', 'np.sin')


# In[33]:


#to check source code 
get_ipython().run_line_magic('pinfo2', 'np.any')


# ### Magic Commands
# 

# In[34]:


#to see all magic commands 
get_ipython().run_line_magic('magic', '')


# In[35]:


#Current working directory 
get_ipython().run_line_magic('pwd', '')


# In[37]:


get_ipython().run_cell_magic('time', '', 'for i in range(5):\n    print(np.sin(5))')


# In[38]:


get_ipython().run_line_magic('magic', '')


# In[39]:


#to see all files on the var log directory
get_ipython().system('ls /var/log')


# In[40]:


dir_name = '/var/log'
get_ipython().system('ls $dir_name')


# In[41]:


files = get_ipython().getoutput('ls /var/log')
#REPL - read, eval, print and loop
print('We have {} files at {}'.format(len(files),dir_name))


# How to make bullet points in markdown: 
# * This 
# * is 
# * a 
# * list 
# 
# HTML hyperlink: 
# [Pandas](http://pandas.pydata.org) is awesome.
# 
# We can also add images via URL: 
# ![](https://cdn.pixabay.com/photo/2013/04/01/09/07/wink-98461_1280.png)
# 
# And, we can enter LaTex script: 
# 
# $e^x$ 
# $$ \ln(e^x) = x $$

# ## NumPy Basics
# 

# * Building block of most scietific packages in Python 
# * Provides fast arrays, math functions, linear algebra, randomization, and more
# * Not the best option for speed and memoy consumption 

# In[43]:


# power operation 4^2 
4 ** 2


# In[44]:


np.int64(2) ** 1000


# In[46]:


## Creating arrays 
arr = np.array([1,2,3])
arr


# In[47]:


len(arr)


# In[48]:


arr[1]


# In[49]:


type(arr[1])


# In[52]:


# array data type
arr.dtype


# In[56]:


#specifying the data type of an array
arr32 = np.array( [1,2,3] , dtype = np.int32)
arr32


# In[57]:


arr * arr


# In[59]:


v1 = np.random.rand(10000)
v2 = np.random.rand(10000)


# In[61]:


get_ipython().run_line_magic('time', 'v1 * v2')


# In[65]:


#dot product
np.dot(arr,arr)


# In[66]:


arr @ arr


# In[69]:


# making matrix 
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
mat


# In[78]:


#faster way of making the same matrix above
v = np.arange(12)
v.reshape((4,3))


# In[80]:


mat = np.arange(12).reshape((4,3))


# In[81]:


mat.shape


# In[83]:


mat2 = mat.reshape(3,4)
mat2


# In[84]:


mat[1,2] = 17 


# In[86]:


mat2


# In[88]:


#transpose 
mat.T


# In[91]:


#slicing
nums = [1,2,3,4,5]
nums[2:4]
#remember python index starts at 0


# In[92]:


v = np.arange(1,6)
v[2:4]


# In[95]:


arr = np.arange(12).reshape((3,4))
arr


# In[96]:


arr[0]


# In[97]:


arr[1,1]


# In[98]:


arr[:,1]


# In[99]:


arr[:,1].reshape((3,1))


# In[100]:


arr[1:,2:]


# In[101]:


#broadcasting
arr[1:,2:] = 7


# In[102]:


arr


# In[106]:


#boolean indexing 
arr = np.arange(3)
arr


# In[107]:


arr[np.array([True,False,True])]
arr >= 1


# In[108]:


arr[arr>=1]


# In[111]:


arr = np.arange(10)


# In[113]:


arr[(arr>2)&(arr<7)]


# In[114]:


arr[(arr>2)|(arr>7)]


# In[116]:


#returns all values less than or equal to seven
arr[~(arr>7)]


# In[118]:


mat = np.random.rand(5,5)
mat


# In[120]:


#finding all values that are more than 1/2 std devation from the mean
mat[np.abs(mat-mat.mean()) > 1.5*mat.std()]


# In[122]:


#normalizing all these outliers to the mean
mat[np.abs(mat-mat.mean()) > 1.5*mat.std()] = mat.mean()


# In[ ]:




