
# coding: utf-8

# In[91]:


from sklearn.datasets import fetch_openml
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import sparse_random_matrix
svd = TruncatedSVD(n_components=783, algorithm='randomized', n_iter=7, random_state=42)
scaler = StandardScaler()

X,y = fetch_openml('mnist_784', version=1, return_X_y = True)


# In[92]:


# Question 3

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
i=42
plt.imshow(X[i].reshape(28,28))
print(y[i])


# In[93]:


scaler.fit(X)
Xn = scaler.transform(X)


# In[94]:


plt.imshow(Xn[i].reshape(28,28))
print(y[i])


# In[95]:


svd.fit(Xn)  

# print(svd.explained_variance_ratio_)  
print(svd.explained_variance_ratio_.sum())  
# print(svd.singular_values_) 


# In[96]:


plt.plot(svd.singular_values_, 'r')
plt.xlabel('Dimensions')
plt.ylabel('Singular values')


# In[97]:


cum_sum = [0]
for i,var in enumerate(svd.explained_variance_ratio_):
    cum_sum.append(var+cum_sum[i])
#     if cum_sum[-1]>= 0.9:
#         print (i) # 237
#         break
    
plt.plot(cum_sum, 'r') # the top 237 dimensions explain 90% of the variance 
plt.plot(237,0.9,'k.')
plt.xlabel('Dimensions')
plt.ylabel('Cumulative Explained Variance')


# In[98]:


svd2 = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=7, random_state=42)
svd2.fit(Xn)  
print(svd2.explained_variance_ratio_.sum())  


# In[99]:


plt.plot(svd2.singular_values_, 'r')
plt.xlabel('Dimensions')
plt.ylabel('Singular values')


# In[100]:


print(y)
import matplotlib.colors as clrs

x_red = svd2.fit_transform(Xn)
colors = {'1':'red',
          '2':'orange',
          '3':'yellow',
          '4':'green',
          '5':'blue',
          '6':'purple',
          '7':'black',
          '8':'magenta',
          '9':'pink',
          '0':'gold'
         }
colors = ['red','orange','yellow','green','blue','purple','black','magenta','pink','gold']
# x_red.shape
# plt.scatter(x_red[:,0],x_red[:,1],c=y, cmap=clrs.ListedColormap(colors))
x_red[:,0].shape
# len(svd2.components_[0])


# In[104]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
label = [int(j) for j in y]
# print(label)
# x = [4,8,12,16,1,4,9,16]
# y = [1,4,9,16,4,8,12,3]
# label = [0,1,2,3,0,1,2,3]
# colors = ['red','green','blue','purple']

plt.scatter(x_red[:,0], x_red[:,1], c=label, cmap=matplotlib.colors.ListedColormap(colors))


# In[169]:


# Question 5
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

lgrg = LogisticRegression()
scaler_iris = StandardScaler()
svd_iris = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=7, random_state=42)

iris = load_iris()

Xi = iris['data']
yi = iris['target']

scaler_iris.fit(Xi) # fit the scaler
Xin = scaler_iris.transform(Xi) # normalize using the scaler

svd_iris.fit(Xin) # fit the svd
print('Explained variance:',svd_iris.explained_variance_ratio_.sum())  # 95.8% for 2 comp, 99.5% for 3 comp
X_iris_red = svd_iris.fit_transform(Xin) # use the fitter svd to transform the data

#train test split
trainX,testX,trainy,testy = train_test_split(X_iris_red,yi,test_size=0.33, random_state=42)

# fit the logistic regression model
lgrg.fit(trainX,trainy)
# predict and score the predictions
score = lgrg.score(testX, testy)
print('Scroe:',score) # 0.84 for 2-svd, 0.88 for 3-svd

