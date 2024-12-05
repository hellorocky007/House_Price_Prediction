#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import math as m
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score,mean_squared_error


# In[6]:


data=pd.read_csv("G:/bangalore house price prediction OHE-data.csv")


# ### EXPLORATORY DATA ANALYSIS

# In[28]:


data.head(2)


# In[29]:


data.shape


# In[30]:


print(data.columns)
print("-------------------------------------------")
print(data.dtypes)
print("-------------------------------------------")
print(data.info())


# In[8]:


data["balcony"].unique()


# We can see that there are houses with 1.58437574 balconies which appears unusual.

# In[32]:


#data[data["balcony"].between(1.1,1.7)]
data[(data["balcony"]>1.1) & (data["balcony"]<1.7)]


# In[33]:


(202/7120)*100


# we can see there are around 202 records with 1.584376 no of balconies.This 202 records occupy around 2.8 percent of our data. We have total 7120 records,so lets drop this rows with errors.

# In[9]:


df=data[(data["balcony"]==0)|(data["balcony"]==1) |(data["balcony"]==2)|(data["balcony"]==3)]
df.head(2)


# In[35]:


df["balcony"].unique()


# In[36]:


print("Total no of houses are --",len(df))
print("Houses with Super built-up Area--",len(df[df['area_typeSuper built-up  Area']==1]))
print("Houses with Built-up Area--",len(df[df['area_typeBuilt-up  Area']==1]))
print("Houses with Plot Area--",len(df[df['area_typePlot  Area']==1]))
print("-----------------")
print("Upto {} BHK houses are available.".format(df['bhk'].max()))
print("No of Ready to Move Houses are--",len(df[df['availability_Ready To Move']==1]))
print("-----------------")
print("Average price per sqft--",df["price_per_sqft"].mean())


# 
# Checking for missing values--

# In[37]:


df.isnull().sum()


# In[38]:


df.columns


# In[10]:


# Renaming columns--
df1=df.rename({'area_typeSuper built-up  Area':"super_builtup_area",'area_typeBuilt-up  Area':"builtup_area",'area_typePlot  Area':"plot_area",'availability_Ready To Move':"available"},axis=1)
df1.head()


# In[40]:


df1.columns


# In[55]:


detail= [22,40,10,50,70]
s1= pd.Series(detail)
s1


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Exploratory Data Analysis --

# Checking for outliers--

# In[47]:


sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
plt.xticks(rotation=90,fontsize="medium")
sns.boxplot(data=df1.loc[:, ['bath', 'balcony', 'total_sqft_int', 'bhk', 'price_per_sqft',
       'super_builtup_area', 'builtup_area', 'plot_area', 'available']],palette="Oranges")


# Total sqft int area and price per sqft varies for different houses hence, this two columns are showing outliers.

# In[65]:


sns.barplot(x="bhk",y="price_per_sqft",data=df1,palette='CMRmap')
plt.title("Size of BHKs V/S Price per sqft area")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,6)


# In[58]:





# We can see that Price per sqft area is high for 5 and 6 BHK houses.

# In[69]:


sns.countplot(x="bhk",data=df1,palette='BuGn_r')
plt.title("Count of BHKs")
plt.rcParams['figure.figsize']=(8,6)


# In[67]:


sns.lineplot(x="bhk",y=df1["price_per_sqft"],data=df1)
plt.title("Size of BHKs V/S Price per sqft area")
plt.rcParams['figure.figsize']=(6,4)


# Price per sqft area is sharply rising with BHKs however, as we are approaching towards 9 BHKs, there is a downfall in prices.

# In[70]:


dfn=df1[(df1["bhk"]==2)|(data["bhk"]==3)]
sns.barplot(x="bhk",y="price_per_sqft",data=dfn,palette='BuGn')
plt.title("Comparing Price per sqft area of 2 BHK V/S 3 BHK")


# 
# Priorly, we have seen that there are more no of 2 BHKs,in addition, we can say that price per sqft area is also less for 2 BHKs than 3 BHKs.

# In[71]:


sns.lineplot(x='balcony',y='price_per_sqft',data=df1,palette='PuBuGn')
plt.title("No of Balconies V/S Price per sqft area")


# From the above graph, we can conclude that as no of balconies expands from 1 to 2 and from 2 to 3 there is increase in price per sqft area.

# In[72]:


df1[['bath', 'balcony', 'price', 'total_sqft_int', 'bhk', 'price_per_sqft',
     'super_builtup_area','builtup_area', 'plot_area', 'available']].corr()


# In[80]:


sns.heatmap(df1 [['bath', 'balcony', 'price', 'total_sqft_int', 'bhk', 'price_per_sqft','super_builtup_area',
                  'builtup_area', 'plot_area', 'available']].corr(),annot=True,cmap="RdBu")
plt.rcParams['figure.figsize']=(8,6)


# ### Model building-- Using KNN Regressor--

# In[11]:


from sklearn.neighbors import KNeighborsRegressor

X=df1[['bath', 'balcony', 'total_sqft_int', 'bhk','price_per_sqft','super_builtup_area',
       'builtup_area', 'plot_area', 'available']]
y=df1['price']

def train_test(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
    print(X_train.shape),print(y_train.shape)
    print(X_test.shape),print(y_test.shape)
    return X_train,X_test,y_train,y_test

print("Calling the train_test function--")
X_train,X_test,y_train,y_test=train_test(X,y)

def modelling(X_train,y_train,X_test):
    model=KNeighborsRegressor(n_neighbors=22)
    model_train=model.fit(X_train,y_train)
    print("Model training is completed")
    pred_knn=model_train.predict(X_test)
    return pred_knn

print("Calling the modelling function--")
pred_knn=modelling(X_train,y_train,X_test)
print(pred_knn)
r2score=(round(r2_score(y_test,pred_knn)*100,2))
print("--------------------------------------------")
print("KNN Regression--")
print('r2score:',r2score)
rmse = m.sqrt(mean_squared_error(y_test,pred_knn))
print('RMSE:',rmse)


# In[12]:


#To choose k---sqrt(n)
from math import sqrt
import numpy as np
(len(df1))
sqrt(len(df1))


# In[ ]:




