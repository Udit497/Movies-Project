#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure 

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #Adjusts the configuration of the plots we will be creating 


# In[2]:


#Reading the data

df = pd.read_csv(r'C:\Users\Udit Soni\OneDrive\Desktop\movies.csv')


# In[3]:


#Preview of the dateframe

df.head()


# In[4]:


# looking for any blank cells:

for col in df.columns:
    pct_missing =   np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[5]:


# fixing the missing values

df = df.dropna()


# In[6]:


# Again checking for missing values:

for col in df.columns:
    pct_missing =   np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[7]:


# checking for duplicate rows: ( no duplicate row present)

duplicateRows = df[df.duplicated()]
print(duplicateRows)


# In[8]:


# looking at the data types for columns:

df.dtypes


# In[9]:


# cleaning up the data :

df.head()

# removing the zeroes after decimal for Budget, Votes and Gross and rumtime columns:

df.votes = df.votes.astype('int64')
df.budget = df.budget.astype('int64')
df.gross = df.gross.astype('int64')
df.runtime = df.runtime.astype('int64')


# In[10]:


#Reviewing the changes made:

df.head()


# In[11]:


#As we can see that the year in 'year' column and 'released' column is different so let's make those years same:

df['YearCorrected'] = df.released.str.extract(pat = '([0-9]{4})').astype(int)


# In[12]:


#Reviewing the changes

df.head(5)


# In[13]:


#Sorting by gross revenue in descending order:

df.sort_values(by='gross', ascending = False).head(5)


# In[14]:


# scatter plot with budget vs gross

plt.scatter(x = df.budget, y = df.gross)
plt.title('Budget Vs gross')
plt.xlabel('Budget')
plt.ylabel('Gross')
plt.show


# In[15]:


# Plotting Budget vs Gross using seaborn

sns.regplot(x = 'budget', y='gross', data=df, scatter_kws={"color":"red"},
            line_kws={"color":"blue"})


# In[16]:


#looking at the correlations:
#1. pearson  2. kendall, spearman

df.corr(method = 'pearson')


# In[17]:


#showing correlation in visualization:

viz = df.corr(method = 'pearson')
sns.heatmap(viz, annot = True)
plt.title('Viz')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')


# In[18]:


df


# In[19]:


# changing the order of the columns and deleting the redundant columns:
df = df[['name', 'genre', 'YearCorrected', 'rating', 'score', 'released', 'director', 'writer', 'star','country', 'company','runtime','votes','budget','gross','year']]


# In[21]:


df.head()


# In[22]:


del df['year']


# In[23]:


df.head(1)


# In[28]:


# changing all the data to nueremic for correlation analysis:
df_numeric = df


for col_name in df_numeric.columns:
    if(df_numeric[col_name].dtype == 'object'):
        df_numeric[col_name]= df_numeric[col_name].astype('category')
        df_numeric[col_name] = df_numeric[col_name].cat.codes
        
df_numeric.head()


# In[30]:


#Showing correlation on new dataframe:

df_numeric.corr(method='pearson')


# In[36]:


# Visualizing correlation in new dataframe:
Viz1 = df_numeric.corr(method='pearson')

sns.heatmap(Viz1, annot = True)

plt.title("Viz for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[41]:


# Showing correlation between score and gross:

sns.stripplot(x ='score', y ='gross',data = df)

