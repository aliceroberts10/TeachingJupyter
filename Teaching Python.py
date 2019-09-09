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


# In[124]:


arr = np.arange(3)
arr+4


# In[125]:


arr/7


# In[126]:


arr ** 2


# In[128]:


mat = np.arange(9).reshape((3,3))
vec = np.arange(3)


# In[129]:


mat+vec


# In[130]:


v1 = np.arange(3)
v2 = np.arange(3).reshape((3,1))


# In[131]:


v2


# In[134]:


v2.shape


# In[135]:


v1 + v2


# In[138]:


#example of common matrix errors 
# np.arange (3) + np.arange(4)


# dir command 
# * to see what attributes and methods are in an object

# In[139]:


v = np.arange(12).reshape((4,3))


# In[140]:


dir(v)


# In[143]:


v.any()


# In[144]:


v.all()


# In[147]:


import this


# In[149]:


if v.any():
    print('v is True')


# In[150]:


v.prod()


# In[151]:


v.sum(axis=1)


# In[152]:


v.sum(axis=0)


# In[157]:


v1 = v.copy()


# In[158]:


v1[0,0] = 1000


# In[160]:


v1


# In[161]:


data = v.dumps()
data


# In[163]:


v2 = np.loads(data)
v2


# In[164]:


np.prod(v)


# In[165]:


np.sin(np.pi/2)


# In[167]:


v = np.arange(-3,3)
v


# In[168]:


np.sin(v)


# In[169]:


def noneg(n):
    if n <0: 
        return 0
    return n


# In[170]:


noneg(7)


# In[171]:


noneg(-3)


# In[173]:


#example of error
#noneg(v)


# In[174]:


# we fix this problem by the following:
@np.vectorize
def noneg(n):
    if n <0: 
        return 0
    return n


# In[175]:


noneg = np.vectorize(noneg)


# In[176]:


noneg(3)


# In[177]:


noneg(3).shape


# In[178]:


noneg(v)


# In[180]:


nv = np.array([-1,np.nan,1])
np.sin(nv)


# In[181]:


#another example of error
#noneg(nv)


# In[182]:


np.nan>0


# In[183]:


np.nan <0


# In[184]:


np.nan == np.nan


# In[187]:


@np.vectorize
def noneg(n):
    if not np.isnan(n) and n <0: 
        return n.__class__(0)
    return n


# In[189]:


noneg(nv)
#now it works


# In[190]:


#another way 
@np.vectorize
def isneg(n):
    return not np.isnan(n) and n < 0 


# In[191]:


nv[isneg(nv)] = 0


# In[192]:


nv


# ## Pandas

# * Library for real-world data
# * Adopted by Python scientific community as the main tool for working with data
# * Many I/O tools
# * Supports heterogeneous data (mixed types)
# * Time series functionality
# * Efficient with large amounts of data

# In[ ]:




